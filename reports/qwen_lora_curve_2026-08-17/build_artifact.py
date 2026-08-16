#!/usr/bin/env python3
"""Build the portable Qwen LoRA training-curve diagnostic artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
PROTOCOLS = ("forward_temporal", "session_disjoint", "unseen_site")
SOURCE_ID = "qwen_lora_seed43_training_logs"


def latest_state(protocol: str) -> tuple[Path, dict]:
    run_dir = REPO_ROOT / "outputs" / "qwen_lora" / protocol / "seed43"
    paths = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"No trainer state found for {protocol}")
    path = paths[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def capped_training_population(protocol: str) -> tuple[int, int, int]:
    labels = {
        line.strip()
        for line in (REPO_ROOT / "configs" / "core18_complete.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    csv_path = REPO_ROOT / "um7" / protocol / "train.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["source_class"] in labels
        ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_class"]].append(row)
    selected = []
    for label in sorted(grouped):
        ordered = sorted(
            grouped[label],
            key=lambda row: hashlib.sha256(
                f"43:{row['record_uid']}".encode()
            ).digest(),
        )
        selected.extend(ordered[:250])
    return len(selected), len(grouped), sum(len(rows) < 250 for rows in grouped.values())


def rounded(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def main() -> None:
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    curve_rows = []
    run_rows = []
    epoch_rows = []
    state_paths = []

    for protocol in PROTOCOLS:
        state_path, state = latest_state(protocol)
        state_paths.append(state_path.relative_to(REPO_ROOT).as_posix())
        logs = [row for row in state["log_history"] if "loss" in row]
        losses = [float(row["loss"]) for row in logs]
        grads = [float(row["grad_norm"]) for row in logs]
        lrs = [float(row["learning_rate"]) for row in logs]
        if not all(math.isfinite(value) for value in losses + grads + lrs):
            raise ValueError(f"Non-finite metric in {state_path}")
        capped_count, class_count, under_cap_classes = capped_training_population(protocol)
        metrics_path = state_path.parents[1] / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        for row in logs:
            loss = float(row["loss"])
            grad = float(row["grad_norm"])
            curve_rows.append(
                {
                    "protocol": protocol,
                    "step": int(row["step"]),
                    "epoch": rounded(row["epoch"], 6),
                    "progress": rounded(row["step"] / state["max_steps"], 6),
                    "loss": rounded(loss, 10),
                    "loss_log10": rounded(math.log10(loss), 6),
                    "grad_norm": rounded(grad, 10),
                    "grad_log10": rounded(math.log10(grad), 6),
                    "learning_rate": rounded(row["learning_rate"], 12),
                    "max_steps": int(state["max_steps"]),
                    "logging_window_steps": 10,
                }
            )

        epoch_means = []
        for epoch in range(1, 6):
            values = [
                float(row["loss"])
                for row in logs
                if epoch - 1 < float(row["epoch"]) <= epoch
            ]
            epoch_means.append(statistics.mean(values))
            epoch_rows.append(
                {
                    "protocol": protocol,
                    "epoch": epoch,
                    "logged_windows": len(values),
                    "mean_loss": rounded(statistics.mean(values), 8),
                    "min_loss": rounded(min(values), 8),
                    "max_loss": rounded(max(values), 8),
                    "last_loss": rounded(values[-1], 8),
                }
            )

        run_rows.append(
            {
                "protocol": protocol,
                "train_samples": capped_count,
                "train_classes": class_count,
                "classes_under_cap": under_cap_classes,
                "steps": int(state["global_step"]),
                "epochs": float(state["epoch"]),
                "first_logged_loss": rounded(losses[0], 6),
                "final_logged_loss": rounded(losses[-1], 8),
                "minimum_logged_loss": rounded(min(losses), 8),
                "drop_percent": rounded((1 - losses[-1] / losses[0]) * 100, 3),
                "reported_train_loss": rounded(metrics["train_loss"], 6),
                "runtime_hours": rounded(metrics["train_runtime"] / 3600, 2),
                "median_grad_norm": rounded(statistics.median(grads), 4),
                "maximum_grad_norm": rounded(max(grads), 4),
                "grad_windows_over_1": sum(value > 1 for value in grads),
                "epoch1_mean_loss": rounded(epoch_means[0], 6),
                "epoch5_mean_loss": rounded(epoch_means[-1], 6),
                "validation_metrics": "未记录",
            }
        )

    analysis_rows = (
        [{"dataset": "training_curves", **row} for row in curve_rows]
        + [{"dataset": "run_summary", **row} for row in run_rows]
        + [{"dataset": "epoch_summary", **row} for row in epoch_rows]
    )
    analysis_data_path = REPORT_DIR / "analysis_data.jsonl"
    analysis_data_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in analysis_rows),
        encoding="utf-8",
    )

    source_query = {
        "engine": "duckdb",
        "language": "sql",
        "sql": (
            "SELECT * FROM read_json_auto("
            "'reports/qwen_lora_curve_2026-08-17/analysis_data.jsonl')"
        ),
        "description": "读取由生成脚本从三个完整 seed43 Trainer state 与 metrics 文件提取的曲线和汇总指标。",
        "executed_at": generated_at,
        "tables_used": state_paths
        + [f"outputs/qwen_lora/{p}/seed43/metrics.json" for p in PROTOCOLS],
        "filters": [
            "seed = 43",
            "latest completed checkpoint per protocol",
            "log_history rows where loss is present",
        ],
        "metric_definitions": [
            "loss：Trainer 每 10 个 optimizer steps 记录的加权 assistant-answer token cross entropy。",
            "loss_log10：log10(loss)，用于同时看清早期快速下降和后期低损失区间。",
            "grad_norm：Trainer 记录的梯度范数；TrainingArguments.max_grad_norm = 1.0。",
            "epoch mean loss：落在对应 epoch 区间内的已记录 10-step loss 窗口的算术平均。",
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Qwen LoRA 训练曲线诊断",
            "description": "seed43 三个协议的训练稳定性、收敛与泛化风险检查。",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "loss_curve",
                    "title": "训练损失曲线（log10）",
                    "subtitle": "seed43；横轴为 epoch，原始 loss 每 10 steps 记录一次",
                    "showDescription": True,
                    "intent": "trend",
                    "question": "三个 protocol 的 loss 是否发散、停滞或异常反弹？",
                    "rationale": "log10 纵轴保留约四个数量级的后期变化；epoch 横轴使不同总步数可比较。",
                    "comparisonContext": {
                        "baseline": "各 protocol 的首个已记录窗口",
                        "grain": "每 10 optimizer steps",
                        "unit": "log10(weighted token cross entropy)",
                    },
                    "type": "line",
                    "dataset": "training_curves",
                    "sourceId": SOURCE_ID,
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        },
                        "y": {
                            "field": "loss_log10",
                            "type": "quantitative",
                            "label": "log10(train loss)",
                        },
                        "color": {
                            "field": "protocol",
                            "type": "nominal",
                            "label": "Protocol",
                        },
                        "tooltip": [
                            {"field": "protocol", "type": "nominal", "label": "Protocol"},
                            {"field": "step", "type": "quantitative", "label": "Step"},
                            {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                            {"field": "loss", "type": "quantitative", "label": "Train loss"},
                            {"field": "learning_rate", "type": "quantitative", "label": "Learning rate"},
                        ],
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "log10(train loss)",
                    "layout": "full",
                    "palette": {"kind": "categorical"},
                    "legend": {"position": "bottom", "sort": "spec"},
                    "surface": {"surface": "card", "viewMode": "both"},
                },
                {
                    "id": "gradient_curve",
                    "title": "梯度范数曲线（log10）",
                    "subtitle": "seed43；配置的 max_grad_norm 为 1.0",
                    "showDescription": True,
                    "intent": "trend",
                    "question": "是否存在持续爆梯度或与 loss 同步恶化的梯度异常？",
                    "rationale": "log10 纵轴显示低梯度区间和孤立尖峰；同一 epoch 归一化便于跨 protocol 比较。",
                    "comparisonContext": {
                        "baseline": "max_grad_norm = 1.0",
                        "grain": "每 10 optimizer steps",
                        "unit": "log10(gradient norm)",
                    },
                    "type": "line",
                    "dataset": "training_curves",
                    "sourceId": SOURCE_ID,
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        },
                        "y": {
                            "field": "grad_log10",
                            "type": "quantitative",
                            "label": "log10(grad norm)",
                        },
                        "color": {
                            "field": "protocol",
                            "type": "nominal",
                            "label": "Protocol",
                        },
                        "tooltip": [
                            {"field": "protocol", "type": "nominal", "label": "Protocol"},
                            {"field": "step", "type": "quantitative", "label": "Step"},
                            {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                            {"field": "grad_norm", "type": "quantitative", "label": "Gradient norm"},
                            {"field": "loss", "type": "quantitative", "label": "Train loss"},
                        ],
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "log10(grad norm)",
                    "layout": "full",
                    "palette": {"kind": "categorical"},
                    "legend": {"position": "bottom", "sort": "spec"},
                    "surface": {"surface": "card", "viewMode": "both"},
                },
            ],
            "tables": [
                {
                    "id": "run_summary_table",
                    "title": "完整训练运行汇总",
                    "subtitle": "seed43；首末 loss 均为 10-step 日志窗口，reported train loss 为全训练聚合值",
                    "showDescription": True,
                    "dataset": "run_summary",
                    "sourceId": SOURCE_ID,
                    "defaultSort": {"field": "protocol", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "protocol", "label": "Protocol", "type": "text"},
                        {"field": "train_samples", "label": "训练样本", "format": "number"},
                        {"field": "train_classes", "label": "训练类别", "format": "number"},
                        {"field": "steps", "label": "Steps", "format": "number"},
                        {"field": "first_logged_loss", "label": "首个 loss", "format": "number"},
                        {"field": "final_logged_loss", "label": "末个 loss", "format": "number"},
                        {"field": "drop_percent", "label": "下降幅度 (%)", "format": "number"},
                        {"field": "reported_train_loss", "label": "全程 train loss", "format": "number"},
                        {"field": "maximum_grad_norm", "label": "最大 grad norm", "format": "number"},
                        {"field": "grad_windows_over_1", "label": "grad>1 窗口", "format": "number"},
                        {"field": "runtime_hours", "label": "耗时 (h)", "format": "number"},
                        {"field": "validation_metrics", "label": "验证指标", "type": "text"},
                    ],
                }
            ],
            "sources": [
                {
                    "id": SOURCE_ID,
                    "label": "Qwen LoRA seed43 Trainer logs",
                    "path": "reports/qwen_lora_curve_2026-08-17/build_artifact.py",
                }
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# Qwen LoRA 训练曲线诊断",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": (
                        "## 技术结论：训练稳定，但泛化风险无法从现有日志排除\n\n"
                        "- **没有发现数值发散。** 三个 seed43 运行都完成 5 个 epoch，loss、梯度范数和学习率全部为有限值；学习率按 linear scheduler 单调下降。\n"
                        "- **最需要警惕的是训练 loss 过低而没有验证曲线。** 末个窗口 loss 为 0.00068–0.00109，相比首个窗口下降 99.94%–99.96%。这说明训练目标几乎被拟合，但不能证明验证集泛化。\n"
                        "- **loss 的含义会放大“看起来很好”的程度。** 当前 loss 是整个固定 JSON assistant answer 的 token 平均交叉熵；大量标点、字段名和固定空值很容易预测。即使 label token 权重为 2，也不能把接近 0 的总 loss 等同于分类准确率接近 100%。\n"
                        "- **建议优先补验证，而不是仅凭曲线改学习率。** 当前配置 `eval_strategy=NO`、无 eval dataset；应至少每个 epoch 记录 validation label accuracy / macro-F1、JSON parse success 和 validation label-token loss，并据此选 checkpoint。"
                    ),
                },
                {
                    "id": "loss_finding",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": (
                        "## 三条 loss 曲线都收敛，但第 5 轮已进入近零训练误差区\n\n"
                        "下图把纵轴换成 log10；每下降 1 表示原始 loss 再缩小 10 倍。三条曲线均先快速下降，随后继续稳定下降，没有持续反弹或平台期。关键含义不是“训练失败”，而是 5 个 epoch 对 2,021–3,569 个 capped 训练样本已经足以把当前 token 级目标拟合得非常彻底。"
                    ),
                },
                {"id": "loss_chart_block", "type": "chart", "chartId": "loss_curve"},
                {
                    "id": "loss_interpretation",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": (
                        "### 如何解读\n\n"
                        "forward_temporal 的末个窗口 loss 为 0.00109，session_disjoint 为 0.00072，unseen_site 为 0.00068。局部小幅回升存在，但都很短暂，并未形成连续恶化。由于没有 validation loss，这些后期下降究竟是有用学习还是训练集记忆，目前无法区分。"
                    ),
                },
                {
                    "id": "gradient_finding",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": (
                        "## 梯度只有孤立尖峰，没有持续爆炸\n\n"
                        "梯度范数总体随 loss 下降。记录值超过 1.0 的窗口数分别为 1、2、5；unseen_site 在约 3.50 和 4.66 epoch 还有两个孤立尖峰，但相邻 loss 很快恢复。结合 `max_grad_norm=1.0`，这更像被裁剪处理的单批难样本，而不是训练失稳。"
                    ),
                },
                {"id": "gradient_chart_block", "type": "chart", "chartId": "gradient_curve"},
                {
                    "id": "gradient_interpretation",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": (
                        "### 如何解读\n\n"
                        "图中尖峰表示某个 10-step 日志窗口附近的梯度更大；判断异常要看是否持续、是否伴随 loss 上升以及是否出现 NaN/Inf。本次三项都没有发生。没有 warmup 的确让首批更新接近 2e-4，但现有证据不支持仅因这些曲线就下调学习率。"
                    ),
                },
                {
                    "id": "run_summary_intro",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": (
                        "## 三个运行均完整结束，训练规模与类别覆盖并不相同\n\n"
                        "forward_temporal 经过每类最多 250 条的 cap 后只有 2,021 条、14 个训练类别；另外两个 protocol 分别有 3,569 和 3,295 条、18 个类别。跨 protocol 的最终 loss 因数据组成和总步数不同，不应直接当成模型优劣排名。"
                    ),
                },
                {"id": "run_summary_block", "type": "table", "tableId": "run_summary_table"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## 范围与指标定义\n\n"
                        "本诊断覆盖 `forward_temporal`、`session_disjoint`、`unseen_site` 的 seed43 完整训练，只分析最新 completed checkpoint 的 Trainer log history。loss 每 10 optimizer steps 记录一次，batch size 为 16、gradient accumulation 为 1；训练使用 bf16、AdamW、2e-4 初始学习率、linear scheduler、0 warmup、最大梯度范数 1.0。\n\n"
                        "训练 loss 是 assistant answer token 上的加权交叉熵。目标 answer 是固定结构 JSON：`events` 中只有类别变化，其余字段和标点高度重复；类别 token 权重为 2.0。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## 诊断方法\n\n"
                        "从三个最新 `trainer_state.json` 提取 loss、grad norm、learning rate、step 和 epoch；检查完成步数、有限值、学习率单调性、局部反弹与 grad>1 窗口；再按 epoch 汇总 loss。TensorBoard 标量的点数和首末值与 Trainer state 一致。跨 protocol 比较使用 epoch 横轴；loss 和 grad norm 用 log10 仅改变显示尺度，不改变趋势结论。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 限制与稳健性\n\n"
                        "- **没有任何 validation 标量。** 无法定位过拟合起点，也无法确认第 4–5 轮是否改善泛化。\n"
                        "- **只有一个完整 seed。** seed42 目录中只有一个 step=10 的旧标量点和一个空事件文件；seed44 无训练日志，因此不能评估随机种子方差。\n"
                        "- **总 token loss 混合了格式和类别学习。** 固定 JSON token 占比未知；缺少独立 label-token loss，近零总 loss 可能主要由格式记忆贡献。\n"
                        "- **日志粒度为 10 steps。** 单批尖峰会被窗口聚合弱化，但这不会改变“无持续失稳”的结论。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 建议的下一步\n\n"
                        "1. 在 Trainer 中加入 eval dataset，并设置 `eval_strategy=\"epoch\"`、`load_best_model_at_end=True`，用 validation macro-F1 或 label accuracy 选 checkpoint。\n"
                        "2. 分开记录 `label_token_loss` 与 `format_token_loss`，同时记录 JSON parse success；这样能判断近零 loss 是否真的来自类别识别。\n"
                        "3. 先评估现有 epoch checkpoints（forward: 508/635，session: 896/1120，unseen: 824/1030）。若倒数第二个 checkpoint 的 validation F1 不差或更好，优先采用早停而非继续训练。\n"
                        "4. 下一轮再做 3 个 seed。学习率 2e-4 当前看起来稳定；可加 3%–5% warmup 作为保守改进，但它不是现有曲线暴露出的首要问题。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 仍待回答的问题\n\n"
                        "- 第 4、5 epoch 的 validation macro-F1 是否继续上升，还是已经回落？\n"
                        "- label token 在加权 loss 中实际占多少权重，单独的 label-token accuracy 是多少？\n"
                        "- forward_temporal 只有 14 个训练类别是否符合协议设计，还是需要对缺失类别单独处理？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "training_curves": curve_rows,
                "run_summary": run_rows,
                "epoch_summary": epoch_rows,
            },
        },
        "sources": [
            {
                "id": SOURCE_ID,
                "label": "Qwen LoRA seed43 training logs and reproducible extraction",
                "path": "reports/qwen_lora_curve_2026-08-17/build_artifact.py",
                "query": source_query,
            }
        ],
    }

    output = REPORT_DIR / "artifact.json"
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
