from __future__ import annotations

from collections import defaultdict


def average_precision(targets: list[bool], scores: list[float]) -> float:
    if len(targets) != len(scores):
        raise ValueError("Targets and scores have different lengths")
    positives = sum(targets)
    if not positives:
        return 0.0

    ranked = sorted(zip(scores, targets), reverse=True, key=lambda item: item[0])
    true_positives = false_positives = 0
    result = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        group_true_positives = 0
        group_size = 0
        while index < len(ranked) and ranked[index][0] == score:
            group_true_positives += int(ranked[index][1])
            group_size += 1
            index += 1
        true_positives += group_true_positives
        false_positives += group_size - group_true_positives
        precision = true_positives / (true_positives + false_positives)
        result += group_true_positives / positives * precision
    return result


def ranking_metrics(
    targets: list[set[str]], score_rows: list[dict[str, float]], labels: list[str]
) -> dict[str, float]:
    if len(targets) != len(score_rows):
        raise ValueError("Targets and score rows have different lengths")
    missing = [label for row in score_rows for label in labels if label not in row]
    if missing:
        raise ValueError(f"Score row is missing label: {missing[0]}")

    per_label = {
        label: average_precision(
            [label in target for target in targets],
            [row[label] for row in score_rows],
        )
        for label in labels
    }
    micro_targets = [
        label in target for target in targets for label in labels
    ]
    micro_scores = [row[label] for row in score_rows for label in labels]
    order = sorted(
        range(len(targets)),
        key=lambda index: max(score_rows[index][label] for label in labels),
        reverse=True,
    )
    cumulative_errors = 0
    risks = []
    for coverage, index in enumerate(order, 1):
        prediction = max(labels, key=score_rows[index].__getitem__)
        cumulative_errors += prediction not in targets[index]
        risks.append(cumulative_errors / coverage)
    return {
        "mean_average_precision": sum(per_label.values()) / len(labels),
        "micro_average_precision": average_precision(micro_targets, micro_scores),
        "aurc": sum(risks) / len(risks),
    }


def pairwise_metrics(
    targets: list[set[str]],
    score_rows: list[dict[str, float]],
    negative_labels: dict[str, tuple[str, ...]],
) -> dict[str, float | int]:
    outcomes = []
    covered_labels = set()
    for target, row in zip(targets, score_rows):
        for positive in target:
            for negative in negative_labels[positive]:
                if negative not in row:
                    continue
                covered_labels.add(positive)
                outcomes.append(
                    1.0
                    if row[positive] > row[negative]
                    else 0.5
                    if row[positive] == row[negative]
                    else 0.0
                )
    if not outcomes:
        raise ValueError("No scored negative pairs")
    return {
        "hard_negative_accuracy": sum(outcomes) / len(outcomes),
        "hard_negative_comparisons": len(outcomes),
        "hard_negative_covered_labels": len(covered_labels),
    }


def classification_metrics(
    targets: list[set[str]], predictions: list[set[str]], labels: list[str]
) -> dict[str, float]:
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions have different lengths")
    per_label = {}
    total_tp = total_fp = total_fn = 0
    recalls = []
    metric_labels = set(labels)
    count_labels = metric_labels | set().union(*targets, *predictions)
    for label in count_labels:
        pairs = zip(targets, predictions)
        tp = sum(label in target and label in prediction for target, prediction in pairs)
        pairs = zip(targets, predictions)
        fp = sum(label not in target and label in prediction for target, prediction in pairs)
        pairs = zip(targets, predictions)
        fn = sum(label in target and label not in prediction for target, prediction in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if label in metric_labels:
            per_label[label] = f1
        if label in metric_labels and tp + fn:
            recalls.append(recall)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_denom = 2 * total_tp + total_fp + total_fn
    return {
        "macro_f1": sum(per_label.values()) / len(labels),
        "micro_f1": 2 * total_tp / micro_denom if micro_denom else 0.0,
        "exact_set_accuracy": sum(a == b for a, b in zip(targets, predictions)) / len(targets),
        "worst_class_recall": min(recalls, default=0.0),
    }


def group_sets(
    group_ids: list[str], labels: list[str], predictions: list[set[str]]
) -> tuple[list[set[str]], list[set[str]]]:
    targets_by_group: dict[str, set[str]] = defaultdict(set)
    predictions_by_group: dict[str, set[str]] = defaultdict(set)
    for group_id, label, prediction in zip(group_ids, labels, predictions):
        targets_by_group[group_id].add(label)
        predictions_by_group[group_id].update(prediction)
    ordered = sorted(targets_by_group)
    return (
        [targets_by_group[group_id] for group_id in ordered],
        [predictions_by_group[group_id] for group_id in ordered],
    )
