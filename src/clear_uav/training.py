from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import Trainer

from clear_uav.data import Sample
from clear_uav.ontology import Ontology
from clear_uav.prompts import (
    closed_set_conversation,
    conversation,
    counterfactual_conversation,
    structured_target,
)


def _stable_fraction(value: str, offset: int = 0) -> float:
    digest = hashlib.sha256(f"{value}:{offset}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _find_subsequence(sequence: list[int], pattern: list[int]) -> list[int]:
    if not pattern:
        raise ValueError("Cannot find an empty token pattern")
    starts = []
    for index in range(len(sequence) - len(pattern) + 1):
        if sequence[index : index + len(pattern)] == pattern:
            starts.append(index)
    return starts


def encode_assistant_batch(
    processor,
    conversations: list[list[dict]],
    *,
    max_length: int,
    max_pixels: int,
) -> dict[str, torch.Tensor]:
    image_size = {
        "longest_edge": max_pixels,
        "shortest_edge": min(65_536, max_pixels),
    }
    encoded = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True, "size": image_size},
    )
    prompt_conversations = []
    for messages in conversations:
        if messages[-1]["role"] != "assistant":
            raise ValueError("Training conversation must end with an assistant answer")
        prompt_conversations.append(messages[:-1])
    prompts = processor.apply_chat_template(
        prompt_conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True, "size": image_size},
    )

    labels = torch.full_like(encoded["input_ids"], -100)
    for row in range(len(conversations)):
        full_positions = encoded["attention_mask"][row].bool().nonzero().flatten()
        prompt_positions = prompts["attention_mask"][row].bool().nonzero().flatten()
        full_tokens = encoded["input_ids"][row, full_positions]
        prompt_tokens = prompts["input_ids"][row, prompt_positions]
        if len(full_tokens) > max_length:
            raise ValueError(
                f"Encoded sequence has {len(full_tokens)} tokens; increase --max-length "
                "or decrease --max-pixels"
            )
        if not torch.equal(full_tokens[: len(prompt_tokens)], prompt_tokens):
            raise ValueError("Prompt tokens are not a prefix of the training conversation")
        answer_positions = full_positions[len(prompt_tokens) :]
        if not len(answer_positions):
            raise ValueError("Training conversation has an empty assistant answer")
        labels[row, answer_positions] = encoded["input_ids"][row, answer_positions]
    encoded["labels"] = labels
    return dict(encoded)


@dataclass
class ClearCollator:
    processor: object
    ontology: Ontology
    view: str = "pair"
    max_length: int = 2048
    max_pixels: int = 262_144
    label_weight: float = 1.0
    neighbor_loss: bool = False
    counterfactual_loss: bool = False
    random_negative: bool = False
    random_negative_pool: tuple[str, ...] = ()
    context_dropout: float = 0.0
    evidence_dropout: float = 0.0
    targets: dict[str, str] | None = None
    counterfactual_targets: dict[str, str] | None = None

    def _view_and_uncertainty(self, sample: Sample) -> tuple[str, bool]:
        draw = _stable_fraction(sample.record_uid)
        if draw < self.context_dropout:
            return "evidence", True
        if draw < self.context_dropout + self.evidence_dropout:
            return "context", True
        return self.view, False

    def _negative(self, sample: Sample) -> str:
        if self.random_negative:
            choices = [
                label for label in self.random_negative_pool if label != sample.label
            ]
            if not choices:
                raise ValueError("Frequency-matched random-negative pool is empty")
        else:
            choices = list(self.ontology.neighbors(sample.label))
            if not choices:
                raise ValueError(f"No graph neighbor is defined for {sample.label}")
        index = int(_stable_fraction(sample.record_uid, 1) * len(choices))
        return choices[min(index, len(choices) - 1)]

    def _encode(self, conversations: list[list[dict]]) -> dict[str, torch.Tensor]:
        return encode_assistant_batch(
            self.processor,
            conversations,
            max_length=self.max_length,
            max_pixels=self.max_pixels,
        )

    def __call__(self, samples: list[Sample]) -> dict:
        views_and_uncertainty = [self._view_and_uncertainty(sample) for sample in samples]
        targets = [
            structured_target(sample.label, uncertain=True)
            if uncertain
            else self.targets.get(sample.record_uid, structured_target(sample.label))
            if self.targets is not None
            else structured_target(sample.label)
            for sample, (_, uncertain) in zip(samples, views_and_uncertainty)
        ]
        conversations = [
            conversation(sample.context_path, sample.evidence_path, view, answer=target)
            for sample, (view, _), target in zip(samples, views_and_uncertainty, targets)
        ]
        batch = self._encode(conversations)

        weights = torch.ones_like(batch["labels"], dtype=torch.float32)
        if self.label_weight != 1.0:
            for row, (sample, uncertain) in enumerate(
                zip(samples, (item[1] for item in views_and_uncertainty))
            ):
                if uncertain:
                    continue
                pattern = self.processor.tokenizer.encode(sample.label, add_special_tokens=False)
                positions = _find_subsequence(batch["input_ids"][row].tolist(), pattern)
                positions = [p for p in positions if batch["labels"][row, p] != -100]
                if not positions:
                    raise ValueError(
                        f"No assistant label span found for {sample.record_uid}"
                    )
                for start in positions:
                    weights[row, start : start + len(pattern)] = self.label_weight
        batch["token_weights"] = weights

        if self.neighbor_loss or self.counterfactual_loss:
            negatives = [self._negative(sample) for sample in samples]
        if self.neighbor_loss:
            positive_score_conversations = [
                closed_set_conversation(
                    sample.context_path,
                    sample.evidence_path,
                    self.view,
                    label=sample.label,
                    definition=self.ontology.definitions[sample.label],
                )
                for sample in samples
            ]
            negative_score_conversations = [
                closed_set_conversation(
                    sample.context_path,
                    sample.evidence_path,
                    self.view,
                    label=negative,
                    definition=self.ontology.definitions[negative],
                )
                for sample, negative in zip(samples, negatives)
            ]
            batch["positive_score_inputs"] = self._encode(positive_score_conversations)
            batch["negative_score_inputs"] = self._encode(negative_score_conversations)
        if self.counterfactual_loss:
            statements = [
                json.loads(self.counterfactual_targets[sample.record_uid])["evidence"]
                if self.counterfactual_targets is not None
                else f"Visible evidence supports {negative.replace('_', ' ')}."
                for sample, negative in zip(samples, negatives)
            ]
            if not all(statements):
                raise ValueError("Counterfactual evidence statements must be non-empty")
            counterfactual_conversations = [
                counterfactual_conversation(
                    sample.context_path,
                    sample.evidence_path,
                    self.view,
                    statement=statement,
                )
                for sample, statement in zip(samples, statements)
            ]
            batch["counterfactual_inputs"] = self._encode(counterfactual_conversations)
        return batch


def token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.clamp_min(0).reshape(-1),
        reduction="none",
    ).view_as(shift_labels)
    if weights is None:
        weights = torch.ones_like(losses)
    else:
        weights = weights[:, 1:]
    return (losses * weights * mask).sum() / (weights * mask).sum()


def normalized_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    token_logp = shift_logits.log_softmax(-1).gather(
        -1, shift_labels.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    return (token_logp * mask).sum(-1) / mask.sum(-1)


def unlikelihood_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    token_probability = shift_logits.softmax(-1).gather(
        -1, shift_labels.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    losses = -torch.log1p(-token_probability.clamp(max=1 - 1e-6))
    return (losses * mask).sum() / mask.sum()


class ClearTrainer(Trainer):
    def __init__(
        self, *args, margin: float, lambda_neighbor: float, lambda_cf: float, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.lambda_neighbor = lambda_neighbor
        self.lambda_cf = lambda_cf

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("token_weights")
        positive_score_inputs = inputs.pop("positive_score_inputs", None)
        negative_score_inputs = inputs.pop("negative_score_inputs", None)
        counterfactual_inputs = inputs.pop("counterfactual_inputs", None)
        outputs = model(**inputs)
        loss = token_cross_entropy(outputs.logits, inputs["labels"], weights)

        if self.lambda_neighbor:
            if positive_score_inputs is None or negative_score_inputs is None:
                raise ValueError("Neighbor loss inputs are missing")
            positive_outputs = model(**positive_score_inputs)
            negative_outputs = model(**negative_score_inputs)
            positive_score = normalized_log_likelihood(
                positive_outputs.logits, positive_score_inputs["labels"]
            )
            negative_score = normalized_log_likelihood(
                negative_outputs.logits, negative_score_inputs["labels"]
            )
            neighbor_loss = F.relu(self.margin - positive_score + negative_score).mean()
            loss = loss + self.lambda_neighbor * neighbor_loss
        if self.lambda_cf:
            if counterfactual_inputs is None:
                raise ValueError("Counterfactual loss inputs are missing")
            counterfactual_outputs = model(**counterfactual_inputs)
            loss = loss + self.lambda_cf * unlikelihood_loss(
                counterfactual_outputs.logits, counterfactual_inputs["labels"]
            )
        return (loss, outputs) if return_outputs else loss
