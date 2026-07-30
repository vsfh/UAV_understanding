from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from clear_uav.data import Sample
from clear_uav.ontology import Ontology


PROMPT_VERSION = "clear-uav-teacher-v5"
TARGET_KEYS = {"events", "factors", "evidence", "uncertain"}
GENERIC_DESCRIPTION_KEYS = {
    "description",
    "context_visible",
    "evidence_visible",
    "cross_view_relations",
    "unclear",
}
REWRITE_KEYS = {
    "target",
    "counterfactual_target",
    "distinction",
    "missing_required_factors",
}
VERDICT_KEYS = {"supported", "support_score", "unsupported_claims", "reason"}
VERIFICATION_KEYS = {"positive", "counterfactual"}


def humanize_label(label: str) -> str:
    return label.replace("_", " ")


def counterfactual_statement(label: str, definition: str) -> str:
    return (
        f"The event {humanize_label(label)} is supported: visible evidence shows {definition}."
    )


def choose_neighbor(record_uid: str, label: str, ontology: Ontology) -> str:
    neighbors = ontology.neighbors(label)
    if not neighbors:
        raise ValueError(f"No confusion neighbor is defined for {label}")
    digest = hashlib.sha256(f"{record_uid}:counterfactual".encode()).digest()
    return neighbors[int.from_bytes(digest[:8], "big") % len(neighbors)]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Teacher response does not contain a valid JSON object")


def _image_content(sample: Sample) -> list[dict[str, str]]:
    return [
        {"type": "text", "text": "CONTEXT (wide UAV image)"},
        {"type": "image", "path": str(sample.context_path)},
        {"type": "text", "text": "EVIDENCE (linked local crop)"},
        {"type": "image", "path": str(sample.evidence_path)},
    ]


def perception_messages(sample: Sample) -> list[dict[str, Any]]:
    schema = {
        "description": "one concise English description, at most 70 words",
        "context_visible": ["visible context observation"],
        "evidence_visible": ["visible crop observation"],
        "cross_view_relations": ["visible relation between the two views"],
        "unclear": ["detail that cannot be established visually"],
    }
    prompt = (
        "Describe only what is visibly supported by the two UAV images. You are not given an "
        "event label: do not infer a business category, legality, intent, ownership, address, "
        "company, exact date, or facts outside the pixels. Treat the crop as linked evidence, "
        "not as an independent scene. Be explicit about material, state, spatial relation, and "
        "scene context when visible; put ambiguous details in `unclear`. Return exactly one JSON "
        f"object with this schema and no extra keys: {canonical_json(schema)}"
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a conservative aerial-image observer. Report pixels, not policy "
                        "conclusions, and output strict JSON only."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": _image_content(sample) + [{"type": "text", "text": prompt}],
        },
    ]


def rewrite_messages(
    sample: Sample,
    *,
    perception: dict[str, Any],
    ontology: Ontology,
    negative_label: str,
    verified_factors: dict[str, Any],
) -> list[dict[str, Any]]:
    label = sample.label
    target_schema = {
        "events": [label],
        "factors": verified_factors,
        "evidence": (
            f"a concise statement beginning with `The event {humanize_label(label)}` and citing "
            "only visible support"
        ),
        "uncertain": False,
    }
    counterfactual_schema = {
        "events": [negative_label],
        "factors": {},
        "evidence": counterfactual_statement(
            negative_label, ontology.definitions[negative_label]
        ),
        "uncertain": True,
    }
    schema = {
        "target": target_schema,
        "counterfactual_target": counterfactual_schema,
        "distinction": (
            f"the visible factor separating {humanize_label(label)} from "
            f"{humanize_label(negative_label)}, or an explicit statement that it is not visible"
        ),
        "missing_required_factors": ["verified factor not visibly established"],
    }
    prompt = (
        "Rewrite the label-agnostic observation into one training evidence statement for the "
        "verified source annotation. The event name is supervision, not permission to invent "
        "visual support. In both `events` arrays, copy the supplied underscore-separated machine "
        "event string exactly; use readable space-separated names only inside prose. Copy "
        "`verified_factors` exactly; never add a factor. `missing_required_factors` may refer only "
        "to keys present in `verified_factors`, so it must be [] when that object is empty. "
        "The `distinction` prose must explicitly name both readable event names. The "
        "counterfactual target is frozen from the neighbor definition: copy all four of its fields "
        "from `required_schema` exactly, without paraphrasing, caveats, or comparison with the "
        "actual image. If the images do not "
        "visibly establish the event or a verified factor, retain the verified event but set "
        "`uncertain` true and name the gap. The evidence must name the canonical event in readable "
        "space-separated form, cite only visible content, and distinguish the specified confusion "
        "neighbor. Also formulate a separate declarative counterfactual hypothesis for that "
        "neighbor; it is an audit/rejection target and must not be presented as verified truth. "
        "Return strict JSON with exactly the requested keys.\n\n"
        f"label_agnostic_observation={canonical_json(perception)}\n"
        f"verified_event={label}\n"
        f"verified_event_definition={ontology.definitions[label]}\n"
        f"verified_factors={canonical_json(verified_factors)}\n"
        f"confusion_neighbor={negative_label}\n"
        f"confusion_neighbor_definition={ontology.definitions[negative_label]}\n"
        f"required_schema={canonical_json(schema)}"
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You construct conservative commercial-event supervision from verified "
                        "labels and visible UAV evidence. Output strict JSON only."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": _image_content(sample) + [{"type": "text", "text": prompt}],
        },
    ]


def verification_messages(
    sample: Sample,
    *,
    positive_statement: str,
    counterfactual_statement: str,
) -> list[dict[str, Any]]:
    verdict = {
        "supported": False,
        "support_score": 0.0,
        "unsupported_claims": ["claim not directly supported by pixels"],
        "reason": "brief visible-evidence rationale",
    }
    schema = {"positive": verdict, "counterfactual": verdict}
    prompt = (
        "Independently verify two candidate statements against the images. Judge literal visible "
        "support only; do not use dataset labels, filename text, policy assumptions, or world "
        "knowledge to rescue a claim. A score of 1 means every material visual claim is clear, "
        "and 0 means no material claim is supported. Each verdict must discuss only its own "
        "candidate; do not write `Statement A`, `Statement B`, or `both statements` in a reason "
        "or unsupported claim. For an unsupported candidate, identify the missing or contradicted "
        "visible content instead of making meta-comments. `unsupported_claims` must list every "
        "invented or unresolvable image claim. Return exactly one strict JSON object.\n\n"
        f"statement_A={canonical_json(positive_statement)}\n"
        f"statement_B={canonical_json(counterfactual_statement)}\n"
        f"required_schema={canonical_json(schema)}"
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a label-blind image-text entailment verifier. Output strict JSON "
                        "only."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": _image_content(sample) + [{"type": "text", "text": prompt}],
        },
    ]


def validate_perception(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != GENERIC_DESCRIPTION_KEYS:
        raise ValueError(
            f"Perception keys must be {sorted(GENERIC_DESCRIPTION_KEYS)}, got {sorted(value)}"
        )
    if not isinstance(value["description"], str) or not value["description"].strip():
        raise ValueError("Perception description must be a non-empty string")
    for key in GENERIC_DESCRIPTION_KEYS - {"description"}:
        if not isinstance(value[key], list) or not all(
            isinstance(item, str) and item.strip() for item in value[key]
        ):
            raise ValueError(f"Perception field {key} must be a list of non-empty strings")
    return value


def validate_target(
    value: dict[str, Any],
    *,
    expected_event: str,
    expected_factors: dict[str, Any],
) -> dict[str, Any]:
    if set(value) != TARGET_KEYS:
        raise ValueError(f"Target keys must be {sorted(TARGET_KEYS)}, got {sorted(value)}")
    if value["events"] != [expected_event]:
        raise ValueError(f"Target events must be exactly [{expected_event!r}]")
    if value["factors"] != expected_factors:
        raise ValueError("Target factors differ from the supplied verified factors")
    if not isinstance(value["evidence"], str) or not value["evidence"].strip():
        raise ValueError("Target evidence must be a non-empty string")
    if not isinstance(value["uncertain"], bool):
        raise ValueError("Target uncertain must be boolean")
    return value


def validate_rewrite(
    value: dict[str, Any],
    *,
    label: str,
    negative_label: str,
    verified_factors: dict[str, Any],
    expected_counterfactual_statement: str | None = None,
) -> dict[str, Any]:
    if set(value) != REWRITE_KEYS:
        raise ValueError(f"Rewrite keys must be {sorted(REWRITE_KEYS)}, got {sorted(value)}")
    validate_target(
        value["target"],
        expected_event=label,
        expected_factors=verified_factors,
    )
    validate_target(
        value["counterfactual_target"],
        expected_event=negative_label,
        expected_factors={},
    )
    counterfactual_evidence = value["counterfactual_target"]["evidence"]
    if (
        expected_counterfactual_statement is not None
        and counterfactual_evidence != expected_counterfactual_statement
    ):
        raise ValueError(
            "Counterfactual evidence must exactly copy the frozen neighbor-definition statement"
        )
    forbidden_counterfactual_patterns = (
        r"\b(?:counterfactual|hypothes(?:is|ized)|however|but|although|instead)\b",
        r"\b(?:is|are|was|were|do|does|did|can|could|would)\s+not\b",
        r"\brather than\b",
    )
    if any(
        re.search(pattern, counterfactual_evidence, flags=re.IGNORECASE)
        for pattern in forbidden_counterfactual_patterns
    ):
        raise ValueError(
            "Counterfactual evidence must be a positive assertion without a disclaimer"
        )
    if not isinstance(value["distinction"], str) or not value["distinction"].strip():
        raise ValueError("Rewrite distinction must be a non-empty string")
    if not _contains_phrase(value["distinction"], label) or not _contains_phrase(
        value["distinction"], negative_label
    ):
        raise ValueError("Rewrite distinction must explicitly name both event labels")
    missing = value["missing_required_factors"]
    if not isinstance(missing, list) or not all(
        isinstance(item, str) and item.strip() for item in missing
    ):
        raise ValueError("missing_required_factors must be a list of non-empty strings")
    if missing and not value["target"]["uncertain"]:
        raise ValueError("A target with missing required factors must be uncertain")
    return value


def _validate_verdict(value: dict[str, Any], name: str) -> None:
    if set(value) != VERDICT_KEYS:
        raise ValueError(f"{name} verdict keys must be {sorted(VERDICT_KEYS)}")
    if not isinstance(value["supported"], bool):
        raise ValueError(f"{name}.supported must be boolean")
    score = value["support_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError(f"{name}.support_score must be a number in [0, 1]")
    claims = value["unsupported_claims"]
    if not isinstance(claims, list) or not all(
        isinstance(item, str) and item.strip() for item in claims
    ):
        raise ValueError(f"{name}.unsupported_claims must be a list of strings")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError(f"{name}.reason must be a non-empty string")
    meta_pattern = r"\b(?:statement\s+[ab]|both statements)\b"
    if re.search(meta_pattern, value["reason"], flags=re.IGNORECASE) or any(
        re.search(meta_pattern, claim, flags=re.IGNORECASE) for claim in claims
    ):
        raise ValueError(f"{name} verdict contains cross-candidate meta-commentary")


def validate_verification(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != VERIFICATION_KEYS:
        raise ValueError(
            f"Verification keys must be {sorted(VERIFICATION_KEYS)}, got {sorted(value)}"
        )
    _validate_verdict(value["positive"], "positive")
    _validate_verdict(value["counterfactual"], "counterfactual")
    return value


def _contains_phrase(text: str, label: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    phrase = re.sub(r"[^a-z0-9]+", " ", humanize_label(label).lower()).strip()
    return bool(re.search(rf"\b{re.escape(phrase)}\b", normalized))


def _has_unsupported_named_entity(text: str) -> bool:
    patterns = (
        r"\b(?:gps|latitude|longitude|address|company named|owned by|operated by)\b",
        r"https?://",
        r"\b-?\d{1,3}\.\d{4,}\s*[,/]\s*-?\d{1,3}\.\d{4,}\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def automatic_audit(
    *,
    perception: dict[str, Any],
    rewrite: dict[str, Any],
    verification: dict[str, Any],
    label: str,
    negative_label: str,
    positive_threshold: float,
    counterfactual_threshold: float,
) -> dict[str, Any]:
    target = rewrite["target"]
    counterfactual = rewrite["counterfactual_target"]
    positive = verification["positive"]
    negative = verification["counterfactual"]
    checks = {
        "event_phrase_present": _contains_phrase(target["evidence"], label),
        "confusion_neighbor_named": _contains_phrase(rewrite["distinction"], negative_label),
        "no_unsupported_named_entity": not _has_unsupported_named_entity(
            target["evidence"]
        ),
        "positive_entailment": (
            positive["supported"]
            and positive["support_score"] >= positive_threshold
            and not positive["unsupported_claims"]
        ),
        "counterfactual_rejected": (
            not negative["supported"]
            and negative["support_score"] < counterfactual_threshold
        ),
        "uncertainty_consistent": (
            not rewrite["missing_required_factors"] or target["uncertain"]
        ),
        "counterfactual_event_phrase_present": _contains_phrase(
            counterfactual["evidence"], negative_label
        ),
    }
    label_blind_text = canonical_json(perception)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_support_score": positive["support_score"],
        "counterfactual_support_score": negative["support_score"],
        "counterfactual_susceptible": not checks["counterfactual_rejected"],
        "label_mentioned_in_label_agnostic_description": _contains_phrase(
            label_blind_text, label
        ),
        "requires_human_review": True,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
