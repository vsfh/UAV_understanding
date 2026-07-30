from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You recognize operational events in UAV imagery. Use only visible evidence, do not infer "
    "legal status, and return exactly one compact JSON object."
)


def structured_target(label: str, *, uncertain: bool = False) -> str:
    target = {
        "events": [] if uncertain else [label],
        "factors": {},
        "evidence": "",
        "uncertain": uncertain,
    }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def user_content(context: Path, evidence: Path, view: str) -> list[dict[str, str]]:
    if view == "context":
        return [
            {"type": "text", "text": "CONTEXT"},
            {"type": "image", "path": str(context)},
            {"type": "text", "text": "Identify the event. The EVIDENCE view is unavailable."},
        ]
    if view == "evidence":
        return [
            {"type": "text", "text": "EVIDENCE"},
            {"type": "image", "path": str(evidence)},
            {"type": "text", "text": "Identify the event. The CONTEXT view is unavailable."},
        ]
    if view != "pair":
        raise ValueError(f"Unknown view: {view}")
    return [
        {"type": "text", "text": "CONTEXT"},
        {"type": "image", "path": str(context)},
        {"type": "text", "text": "EVIDENCE"},
        {"type": "image", "path": str(evidence)},
        {"type": "text", "text": "Identify the event using both ordered views."},
    ]


def conversation(
    context: Path,
    evidence: Path,
    view: str,
    *,
    answer: str | None = None,
    instruction: str | None = None,
) -> list[dict]:
    content = user_content(context, evidence, view)
    if instruction:
        content.append({"type": "text", "text": instruction})
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


def closed_set_conversation(
    context: Path,
    evidence: Path,
    view: str,
    *,
    label: str,
    definition: str,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Score whether the proposed canonical event is supported by visible "
                        "UAV evidence."
                    ),
                }
            ],
        },
        {"role": "user", "content": user_content(context, evidence, view)},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"{label}: {definition}"}],
        },
    ]


def counterfactual_conversation(
    context: Path,
    evidence: Path,
    view: str,
    *,
    statement: str,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "State the proposed visible evidence without adding unsupported facts.",
                }
            ],
        },
        {"role": "user", "content": user_content(context, evidence, view)},
        {"role": "assistant", "content": [{"type": "text", "text": statement}]},
    ]
