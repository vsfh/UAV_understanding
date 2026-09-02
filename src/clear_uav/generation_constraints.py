from __future__ import annotations

import json
import re

import torch
from transformers import LogitsProcessor


def location_token_strings(count: int = 1000) -> list[str]:
    return [f"<loc_{index}>" for index in range(count)]


def grounding_prefix_allowed_tokens(
    tokenizer,
    labels,
    *,
    location_token_ids: list[int],
    negative_label: str = "no_event",
    prompt_length: int = 0,
    decoder_start_token_id: int | None = None,
):
    """Constrain generation to `no_event` or `label + four <loc_*> tokens`."""
    label_sequences = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"]
        for label in labels
    }
    negative = tokenizer(negative_label, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    loc_ids = set(location_token_ids)

    def allowed_tokens(_batch_id, input_ids):
        prefix = input_ids[prompt_length:].tolist()
        if (
            decoder_start_token_id is not None
            and prefix
            and prefix[0] == decoder_start_token_id
        ):
            prefix = prefix[1:]

        allowed = set()
        if len(prefix) < len(negative) and negative[: len(prefix)] == prefix:
            allowed.add(negative[len(prefix)])
        elif prefix == negative:
            allowed.add(eos)

        for sequence in label_sequences.values():
            if len(prefix) < len(sequence) and sequence[: len(prefix)] == prefix:
                allowed.add(sequence[len(prefix)])
                continue
            if prefix[: len(sequence)] != sequence:
                continue
            coordinates = prefix[len(sequence) :]
            if len(coordinates) < 4 and all(token in loc_ids for token in coordinates):
                allowed.update(loc_ids)
            elif len(coordinates) == 4 and all(token in loc_ids for token in coordinates):
                allowed.add(eos)

        return sorted(allowed) if allowed else [eos]

    return allowed_tokens


def label_prefix_allowed_tokens(
    tokenizer,
    labels,
    *,
    prompt_length: int = 0,
    decoder_start_token_id: int | None = None,
):
    sequences = [
        tokenizer(label, add_special_tokens=False)["input_ids"]
        + [tokenizer.eos_token_id]
        for label in labels
    ]

    def allowed_tokens(_batch_id, input_ids):
        prefix = input_ids[prompt_length:].tolist()
        if (
            decoder_start_token_id is not None
            and prefix
            and prefix[0] == decoder_start_token_id
        ):
            prefix = prefix[1:]
        allowed = {
            sequence[len(prefix)]
            for sequence in sequences
            if len(sequence) > len(prefix) and sequence[: len(prefix)] == prefix
        }
        return sorted(allowed) if allowed else [tokenizer.eos_token_id]

    return allowed_tokens


class JsonCategoryConstraint(LogitsProcessor):
    def __init__(self, tokenizer, labels, prompt_length: int = 0):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.full_values = [
            tokenizer(json.dumps(label), add_special_tokens=False)["input_ids"]
            for label in labels
        ] + [tokenizer("null", add_special_tokens=False)["input_ids"]]
        self.quoted_values = [
            tokenizer(label + '"', add_special_tokens=False)["input_ids"]
            for label in labels
        ]
        self.states = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for row_index, row_ids in enumerate(input_ids):
            generated = row_ids[self.prompt_length :].tolist()
            state = self.states.get(row_index)

            if state is None:
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if re.search(r'"category"\s*:\s*$', text):
                    state = {"start": len(generated), "sequences": self.full_values}
                    self.states[row_index] = state
                elif re.search(r'"category"\s*:\s*"$', text):
                    state = {"start": len(generated), "sequences": self.quoted_values}
                    self.states[row_index] = state

            if state is None or state.get("done"):
                continue

            consumed = generated[state["start"] :]
            candidates = [
                sequence
                for sequence in state["sequences"]
                if sequence[: len(consumed)] == consumed
            ]
            if any(len(sequence) == len(consumed) for sequence in candidates):
                state["done"] = True
                continue

            allowed = sorted(
                {
                    sequence[len(consumed)]
                    for sequence in candidates
                    if len(sequence) > len(consumed)
                }
            )
            if allowed:
                masked = torch.full_like(scores[row_index], -torch.inf)
                masked[allowed] = scores[row_index, allowed]
                scores[row_index] = masked

        return scores
