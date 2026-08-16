import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from clear_uav.data import resolve_image_path
from clear_uav.metrics import (
    average_precision,
    classification_metrics,
    group_sets,
    pairwise_metrics,
    ranking_metrics,
)
from clear_uav.modeling import LORA_PATTERNS
from clear_uav.ontology import load_ontology
from clear_uav.teacher_targets import (
    automatic_audit,
    parse_json_object,
    validate_perception,
    validate_rewrite,
    validate_verification,
)
from clear_uav.training import (
    aligned_normalized_log_likelihood,
    forward_answer_logits,
    normalized_log_likelihood,
    token_cross_entropy,
    unlikelihood_loss,
)


ROOT = Path(__file__).parents[1]


def test_ontology_has_paper_taxonomy() -> None:
    ontology = load_ontology(ROOT / "configs/ontology.yaml")
    assert len(ontology.events) == 67
    assert ontology.neighbors("floating_garbage")
    assert all(ontology.neighbors(label) for label in ontology.labels)


def test_resolve_data_prefix(tmp_path: Path) -> None:
    image = tmp_path / "photos" / "sample.jpg"
    image.parent.mkdir()
    image.touch()
    assert resolve_image_path(tmp_path, "data/photos/sample.jpg") == image


def test_metrics_and_group_union() -> None:
    targets, predictions = group_sets(
        ["a", "a", "b"], ["x", "y", "x"], [{"x"}, {"y"}, set()]
    )
    metrics = classification_metrics(targets, predictions, ["x", "y"])
    assert metrics["exact_set_accuracy"] == 0.5
    assert metrics["micro_f1"] == 0.8


def test_ranking_metrics() -> None:
    assert abs(average_precision([True, False, True], [0.9, 0.8, 0.7]) - 5 / 6) < 1e-12
    metrics = ranking_metrics(
        [{"x"}, {"y"}],
        [{"x": 0.9, "y": 0.1}, {"x": 0.2, "y": 0.8}],
        ["x", "y"],
    )
    assert metrics["mean_average_precision"] == 1.0
    assert metrics["micro_average_precision"] == 1.0
    assert metrics["aurc"] == 0.0
    pairwise = pairwise_metrics(
        [{"x"}, {"y"}],
        [{"x": 0.9, "y": 0.1}, {"x": 0.2, "y": 0.8}],
        {"x": ("y",), "y": ("x",)},
    )
    assert pairwise["hard_negative_accuracy"] == 1.0
    assert pairwise["hard_negative_comparisons"] == 2


def test_losses_are_finite() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
    labels = torch.tensor([[-100, 1, 0]])
    assert torch.isfinite(token_cross_entropy(logits, labels))
    assert torch.isfinite(normalized_log_likelihood(logits, labels)).all()
    assert torch.isfinite(unlikelihood_loss(logits, labels))


def test_answer_only_forward_matches_full_sequence_scores() -> None:
    torch.manual_seed(0)
    full_logits = torch.randn(2, 5, 7)
    labels = torch.tensor(
        [
            [-100, -100, 1, 2, -100],
            [-100, -100, -100, 3, 4],
        ]
    )

    class DummyModel:
        def __init__(self) -> None:
            self.received_labels = False

        def __call__(self, *, input_ids, logits_to_keep, **kwargs):
            self.received_labels = "labels" in kwargs
            return SimpleNamespace(logits=full_logits[:, logits_to_keep, :])

    model = DummyModel()
    outputs, selected_labels, positions = forward_answer_logits(
        model,
        {"input_ids": torch.ones_like(labels), "labels": labels},
    )
    optimized = aligned_normalized_log_likelihood(outputs.logits, selected_labels)
    reference = normalized_log_likelihood(full_logits, labels)
    assert positions.tolist() == [1, 2, 3]
    assert not model.received_labels
    assert torch.allclose(optimized, reference)


def test_lora_scope_freezes_visual_blocks() -> None:
    pattern = LORA_PATTERNS["projector_llm"]
    assert re.fullmatch(pattern, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(pattern, "model.visual.merger.linear_fc1")
    assert re.fullmatch(pattern, "model.visual.deepstack_merger_list.2.linear_fc2")
    assert not re.fullmatch(pattern, "model.visual.blocks.0.mlp.linear_fc1")


def test_teacher_target_schema_and_audit() -> None:
    perception = validate_perception(
        parse_json_object(
            """```json
            {
              "description": "A mesh covers exposed soil in the linked crop.",
              "context_visible": ["Open ground is visible."],
              "evidence_visible": ["Mesh lies over exposed soil."],
              "cross_view_relations": ["The crop is part of the open ground."],
              "unclear": []
            }
            ```"""
        )
    )
    rewrite = {
        "target": {
            "events": ["bare_soil_netting"],
            "factors": {},
            "evidence": (
                "The event bare soil netting is supported by visible mesh covering exposed soil."
            ),
            "uncertain": False,
        },
        "counterfactual_target": {
            "events": ["incomplete_soil_cover"],
            "factors": {},
            "evidence": (
                "The event incomplete soil cover is supported by visible uncovered soil gaps."
            ),
            "uncertain": True,
        },
        "distinction": (
            "Bare soil netting has visible mesh coverage, unlike incomplete soil cover."
        ),
        "missing_required_factors": [],
    }
    validate_rewrite(
        rewrite,
        label="bare_soil_netting",
        negative_label="incomplete_soil_cover",
        verified_factors={},
    )
    verification = {
        "positive": {
            "supported": True,
            "support_score": 0.9,
            "unsupported_claims": [],
            "reason": "The mesh and exposed soil are visible.",
        },
        "counterfactual": {
            "supported": False,
            "support_score": 0.1,
            "unsupported_claims": ["Uncovered gaps are not visible."],
            "reason": "The image instead shows mesh coverage.",
        },
    }
    validate_verification(verification)
    audit = automatic_audit(
        perception=perception,
        rewrite=rewrite,
        verification=verification,
        label="bare_soil_netting",
        negative_label="incomplete_soil_cover",
        positive_threshold=0.65,
        counterfactual_threshold=0.5,
    )
    assert audit["passed"]
    assert audit["requires_human_review"]


def test_teacher_review_round_trip(tmp_path: Path) -> None:
    target = {
        "events": ["bare_soil_netting"],
        "factors": {},
        "evidence": "A generic visible description.",
        "uncertain": False,
    }
    counterfactual = {
        "events": ["incomplete_soil_cover"],
        "factors": {},
        "evidence": "The event incomplete soil cover is visibly supported.",
        "uncertain": True,
    }
    common = {
        "record_uid": "rec_x",
        "source_class": "bare_soil_netting",
        "context_path": "/data/context.jpg",
        "evidence_path": "/data/crop.jpg",
        "counterfactual_target": counterfactual,
        "automatic_audit_passed": True,
        "prompt_version": "clear-uav-teacher-v5",
    }
    generic_path = tmp_path / "generic.jsonl"
    grounded_path = tmp_path / "grounded.jsonl"
    generic_path.write_text(
        json.dumps(
            {
                **common,
                "target": target,
                "supervision_tier": "teacher_generic_pending_human_review",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    grounded_target = {
        **target,
        "evidence": "The event bare soil netting is visibly supported.",
    }
    grounded_path.write_text(
        json.dumps(
            {
                **common,
                "target": grounded_target,
                "supervision_tier": "teacher_grounded_auto_pass_pending_human_review",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    review_path = tmp_path / "review.tsv"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/export_teacher_review.py",
            "--generic-targets",
            generic_path,
            "--grounded-targets",
            grounded_path,
            "--output",
            review_path,
        ],
        check=True,
        cwd=ROOT,
    )
    lines = review_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    values = lines[1].split("\t")
    row = dict(zip(header, values))
    row["generic_decision"] = "accept"
    row["grounded_decision"] = "accept"
    row["counterfactual_decision"] = "accept"
    row["reviewer_id"] = "reviewer-1"
    review_path.write_text(
        "\t".join(header) + "\n" + "\t".join(row[field] for field in header) + "\n",
        encoding="utf-8",
    )
    generic_output = tmp_path / "generic_audited.jsonl"
    grounded_output = tmp_path / "grounded_audited.jsonl"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/finalize_teacher_reviews.py",
            "--generic-targets",
            generic_path,
            "--grounded-targets",
            grounded_path,
            "--reviews",
            review_path,
            "--generic-output",
            generic_output,
            "--grounded-output",
            grounded_output,
        ],
        check=True,
        cwd=ROOT,
    )
    audited = json.loads(grounded_output.read_text(encoding="utf-8"))
    assert audited["supervision_tier"] == "human_audited"
    assert audited["human_review"]["reviewer_id"] == "reviewer-1"


def test_set_rescore_thresholds_transfer(tmp_path: Path) -> None:
    source = {
        "labels": ["x", "y"],
        "predictions": [
            {
                "record_uid": "a",
                "group_id": "g1",
                "target": "x",
                "prediction": ["x"],
                "scores": {"x": 0.9, "y": 0.1},
            },
            {
                "record_uid": "b",
                "group_id": "g2",
                "target": "y",
                "prediction": ["y"],
                "scores": {"x": 0.2, "y": 0.8},
            },
        ],
    }
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps(source), encoding="utf-8")
    validation_output = tmp_path / "val_set_max.json"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/rescore_sets.py",
            "--scores",
            scores,
            "--aggregator",
            "max",
            "--fit-thresholds",
            "--output",
            validation_output,
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    thresholds = validation_output.with_suffix(".thresholds.json")
    assert thresholds.is_file()

    test_output = tmp_path / "test_set_max.json"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/rescore_sets.py",
            "--scores",
            scores,
            "--aggregator",
            "max",
            "--thresholds",
            thresholds,
            "--output",
            test_output,
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    result = json.loads(test_output.read_text(encoding="utf-8"))
    assert result["threshold_source"] == str(thresholds)
    assert result["metrics"]["macro_f1"] == 1.0
