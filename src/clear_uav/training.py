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
    shift_labels = labels[:, 1:]
    shift_weights = weights[:, 1:] if weights is not None else None
    return aligned_token_cross_entropy(logits[:, :-1], shift_labels, shift_weights)


def aligned_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy for aligned logits/labels without expanding masked image tokens."""
    mask = labels.ne(-100)
    if not mask.any():
        raise ValueError("No answer tokens are available for cross entropy")
    losses = F.cross_entropy(
        logits[mask].float(),
        labels[mask],
        reduction="none",
    )
    selected_weights = (
        weights[mask].to(losses.dtype)
        if weights is not None
        else torch.ones_like(losses)
    )
    return (losses * selected_weights).sum() / selected_weights.sum()


def normalized_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return aligned_normalized_log_likelihood(logits[:, :-1], labels[:, 1:])


def aligned_normalized_log_likelihood(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Mean answer-token log likelihood while materializing FP32 only for answer tokens."""
    mask = labels.ne(-100)
    counts = mask.sum(-1)
    if (counts == 0).any():
        raise ValueError("Every scoring row must contain at least one answer token")
    row_indices = mask.nonzero(as_tuple=False)[:, 0]
    selected_logits = logits[mask].float()
    selected_labels = labels[mask]
    token_logp = selected_logits.log_softmax(-1).gather(
        -1, selected_labels.unsqueeze(-1)
    ).squeeze(-1)
    totals = torch.zeros(
        labels.shape[0], device=token_logp.device, dtype=token_logp.dtype
    )
    totals.scatter_add_(0, row_indices, token_logp)
    return totals / counts.to(token_logp.dtype)


def unlikelihood_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return aligned_unlikelihood_loss(logits[:, :-1], labels[:, 1:])


def aligned_unlikelihood_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels.ne(-100)
    if not mask.any():
        raise ValueError("No answer tokens are available for unlikelihood loss")
    selected_logits = logits[mask].float()
    selected_labels = labels[mask]
    token_probability = selected_logits.softmax(-1).gather(
        -1, selected_labels.unsqueeze(-1)
    ).squeeze(-1)
    losses = -torch.log1p(-token_probability.clamp(max=1 - 1e-6))
    return losses.mean()


def forward_answer_logits(model, inputs: dict[str, torch.Tensor]):
    """Run Qwen while projecting only positions that predict supervised answer tokens."""
    labels = inputs["labels"]
    shifted_labels = labels[:, 1:]
    positions = shifted_labels.ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
    if not len(positions):
        raise ValueError("No answer-token logit positions are available")
    model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
    outputs = model(**model_inputs, logits_to_keep=positions)
    selected_labels = shifted_labels.index_select(1, positions)
    return outputs, selected_labels, positions


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
        outputs, answer_labels, answer_positions = forward_answer_logits(model, inputs)
        answer_weights = weights[:, 1:].index_select(1, answer_positions)
        loss = aligned_token_cross_entropy(
            outputs.logits, answer_labels, answer_weights
        )

        if self.lambda_neighbor:
            if positive_score_inputs is None or negative_score_inputs is None:
                raise ValueError("Neighbor loss inputs are missing")
            positive_outputs, positive_labels, _ = forward_answer_logits(
                model, positive_score_inputs
            )
            negative_outputs, negative_labels, _ = forward_answer_logits(
                model, negative_score_inputs
            )
            positive_score = aligned_normalized_log_likelihood(
                positive_outputs.logits, positive_labels
            )
            negative_score = aligned_normalized_log_likelihood(
                negative_outputs.logits, negative_labels
            )
            neighbor_loss = F.relu(self.margin - positive_score + negative_score).mean()
            loss = loss + self.lambda_neighbor * neighbor_loss
        if self.lambda_cf:
            if counterfactual_inputs is None:
                raise ValueError("Counterfactual loss inputs are missing")
            counterfactual_outputs, counterfactual_labels, _ = forward_answer_logits(
                model, counterfactual_inputs
            )
            loss = loss + self.lambda_cf * aligned_unlikelihood_loss(
                counterfactual_outputs.logits, counterfactual_labels
            )
        return (loss, outputs) if return_outputs else loss
