# 论文表格实验计划

本计划只展开 `paper/sections/experiments.tex` 和现有五张 TBD 表格已经注册的实验，不增加新的模型、数据切分、消融或指标。所有 TBD 在真实运行和汇总完成前保持不变。

## 路径约定

机器可读的相对路径约定见 `configs/paper_experiment_paths.json`。根目录故意留空，运行前由使用者填写：

```bash
DATA_ROOT=""
MODELS_ROOT=""
OUTPUT_ROOT=""
RESULTS_ROOT=""
ANNOTATION_ROOT=""
```

代码假设：协议文件位于 `$DATA_ROOT/<protocol>/`；新 teacher caption 位于 `$DATA_ROOT/description/<原图片相对路径>.json`；模型位于 `$MODELS_ROOT/<模型名>/`；缺失的审核结果、proposal crop 和 challenge slice 位于 `$ANNOTATION_ROOT/` 下配置文件所列相对路径。这里的文件名只是接口约定，不代表文件已经存在。

## 第一阶段：新 crop caption baseline

新 caption 已知商用事件并把事件写入结尾，因此对应论文已有的 **Grounded-caption Qwen3-VL**，不另设“crop-caption”方法。按照论文表格，该学生模型仍输入 context；crop caption 仅作为训练 target 的 evidence 文本。

开发集三 seed 运行命令：

```bash
python scripts/run_experiment_suite.py \
  --profile development \
  --data-root "$DATA_ROOT" \
  --models-root "$MODELS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --cropped-captions-root description \
  --protocols session_disjoint \
  --experiments grounded_caption \
  --seeds 42 43 44 \
  --skip-zero-shot \
  --resume
```

每个 seed 先由 `scripts/build_cropped_caption_targets.py` 把单图 JSON 转成训练 JSONL，再训练 context-view LoRA，并运行 free/closed-set、set max/LSE 和汇总步骤。输出 target 位于：

```text
$OUTPUT_ROOT/session_disjoint/cropped_caption_targets/seed<seed>.jsonl
```

若准备数据的机器没有 GPU，可给同一命令增加 `--prepare-only`；它只执行 split validation 和 target generation，保留完整实验计划供 GPU 机器使用 `--resume` 接续。

这些 caption 当前属于 `teacher_cropped_caption_not_human_audited`，可作为开发 baseline；不能冒充主表要求的 human-audited grounded caption。原始 JSON 中 `check` 未通过的样本不会被静默丢弃，状态会保存在 `caption_check` 中，便于后续固定审核集合。

当前数据准备结果：seed 42/43/44 各生成 3,569 条 target。三份 target 中 `caption_check.passed` 均为 0/3,569（主要是严格禁词检查中的 `a`/`the`），所以本轮数值必须明确标为 non-audited development baseline；正式表仍等待审核后的 targets。

## 主结果表 `main_results.tex`

主表必须在 provenance gate 为 `READY` 后，按 `forward_temporal`、`session_disjoint`、`unseen_site` 三个冻结协议和 seed 42/43/44 运行。统一汇报 macro-F1、mAP、hard-negative accuracy、set F1、evidence assignment、unseen-site macro-F1 和 AURC；不适用单元格继续保留 `--`。

| 表格行 | 代码映射 | 计划 |
|---|---|---|
| OpenCLIP ViT-L/14 | zero-shot `openclip_direct` / `openclip_definition` | 用 context 运行；主表采用冻结 registry 指定的 prompt 结果。（缺：正式模型权重根目录） |
| GeoChat-7B | 论文注册的 GeoChat direct/definition | 接入其原生推理栈后对相同 manifests 评估。（缺：权重、推理实现） |
| Qwen3-VL-8B + definition | zero-shot `qwen_definition` | 已有评估入口；对三个协议运行。（缺：正式模型权重根目录） |
| Grounded-caption Qwen3-VL | `grounded_caption`, context | 先运行新 crop-caption 开发 baseline；主表改用 human-audited targets。（缺：审核后的正式 targets） |
| Two-view concatenation | `label_pair`, pair | 已有训练和评估入口；三个协议、三个 seed。 |
| CLEAR / CLEAR-Set | `clear_full`, pair/set | 已有 pair 训练、set max/LSE、删除和 view-reliance 入口；正式训练需要审核 target 和 counterfactual。（缺：审核后的正式 targets；attention set aggregator） |

统计按论文固定：模型结果报告三 seed 均值和标准差；位置/flight group 做 10,000 次 paired bootstrap；CLEAR 对 grounded-caption 做主 paired permutation test，并对预注册次要比较做 Holm 校正。模型选择只使用 validation macro-F1 和 AURC。

## 消融表 `ablation.tex`

| 表格行 | suite 名称 | 状态/输入 |
|---|---|---|
| Label-only LoRA | `label_context` | 已实现 |
| Grounded-caption LoRA | `grounded_caption` | 开发 baseline 已接入；正式表（缺：审核 targets） |
| Two-view concatenation | `label_pair` | 已实现 |
| Random-negative margin | `random_negative` | 已实现；正式表（缺：审核 targets） |
| Graph-neighbor margin | `graph_neighbor` | 已实现；正式表（缺：审核 targets） |
| CLEAR without view dropout | `clear_no_dropout` | 已实现；正式表（缺：审核 target/counterfactual） |
| CLEAR full | `clear_full` | 已实现；正式表（缺：审核 target/counterfactual） |

同一矩阵还按实验章节执行 paired views、audited captions、label-token weighting、graph neighbors、counterfactual unlikelihood 和 view dropout；其中 `label_pair_unweighted`、`caption_effect` 及 suite 中的 paired comparisons 用于对应差值检验。随机 negatives 必须按论文要求 frequency-matched，不能用当前数据临时定义其他 negative 规则。

## PEFT 效率表 `peft_efficiency.tex`

所有行使用同一 Qwen3-VL backbone、硬件和 visual-token budget，记录 trainable parameters、peak GPU GB、pair/s 和 macro-F1。

| 表格行 | 代码/计划 | 状态 |
|---|---|---|
| Linear probe | 冻结 backbone 后训练注册的线性分类头 | （缺：实现） |
| Projector only | 只训练视觉 projector | （缺：实现） |
| LLM LoRA | `label_pair_llm_lora` | 已实现 |
| Projector + LLM LoRA | `label_pair` | 已实现 |
| QLoRA | 同 backbone 的量化 LoRA | （缺：实现） |
| Full fine-tuning | 同输入预算全量训练 | （缺：实现、对应硬件预算） |

## 鲁棒性表 `robustness.tex`

对 `qwen_definition`、`grounded_caption`、`label_pair` 和 `clear_full` 四行使用完全相同的冻结 slice。`All` 来自主测试集；其余列严格对应论文中的 small evidence、adverse capture、complex background、unseen site 和 detector/retriever proposal crop。

- `unseen_site` 只在正式 site-disjoint split 存在后运行。（缺：已验证 acquisition provenance）
- proposal crop 替换 oracle crop，不改变标签和其他配置。（缺：proposal manifests/images）
- small evidence、adverse capture、complex background 只从审核后的 slice annotation 取样。（缺：slice annotations）
- 论文另外注册的 oracle、spatial jitter、irrelevant same-scene crop、assigned deletion、size-matched random deletion全部保留；现有代码已覆盖后两项，其余（缺：crop 数据/manifest 与评估接线）。

## Caption quality 表 `caption_quality.tex`

在同一 stratified subset 上导出 zero-shot Qwen、generic-caption LoRA、grounded-caption LoRA 和 CLEAR 的 evidence statements，交给三名 blinded raters，填 event correct、evidence supported、contradiction、hallucination。模型输出不得代替人工评分。（缺：冻结 subset、三评审 rating ledger）

## 论文正文其余已注册实验

这些项目不新增表格行，但必须完成正文中的预注册分析：

1. Set aggregation：max、log-sum-exp、parameter-matched attention，以及 shuffled/missing evidence crops。现有 suite 有 max/LSE。（缺：attention、shuffle/missing 实现）
2. Learning curves：每类 1、4、16 和全部 training groups，dominant class 每 epoch cap；验证/测试不做 synthetic duplication。（缺：group-budget sampler）
3. Tail：text-only zero-shot 和 1-/4-shot exemplars，不把 tail test images 混入 core training。（缺：tail manifests/exemplars 与评估入口）
4. Prompt robustness：六个 meaning-preserving paraphrases，报告最大值和标准差。（缺：冻结六组 prompt 与 runner 接线）
5. Ontology：factor exact match、hierarchy-consistent F1 和 confusion-edge pairwise accuracy。现有 suite 有 confusion-edge pairwise。（缺：factor/hierarchy 标注与指标接线）
6. Crop budget：报告 F1 曲线下面积。（缺：预算序列/评估入口）
7. Counterfactual factor swaps：只交换 water location、material state、coverage state、scene relation 中的一个 factor。（缺：审核 case 与评估入口）
8. Error analysis：class-frequency/F1 heat map 和注册的错误 gallery。（缺：最终预测后生成脚本及人工核验）
9. UAVIT-1M-adapted model：仅在兼容时、不在 benchmark test sources 上重训地评估。（缺：兼容权重与推理接口）
10. Calibration/selective prediction：ECE、AURC 和 risk-coverage。已有 AURC；（缺：ECE 汇总）。
11. Statistics/secondary metrics：per-domain macro-F1、evidence-assignment accuracy、CLEAR 对 grounded-caption 的 paired permutation test 和次要比较的 Holm correction。（缺：domain/assignment 标注与统计接线）

## 填表顺序

当前可执行部分的一键入口为：

```bash
bash runs/30_run_paper_tables.sh
```

该入口明确采用“现有 crop caption 即 grounded caption”的开发假设，运行三个协议、三个
seed 和当前已经接线的全部方法。结束后由 `scripts/export_paper_tables.py` 生成五张同结构
LaTeX 表到 `results/paper_tables/paper_tables/`，同时生成逐格缺失原因清单
`table_manifest.json`。它不会覆盖 `paper/table/` 中注册的正式表，也不会把缺少人工评分、
slice 标注或模型推理接口的格子伪造为结果。

1. 完成 crop caption baseline，检查三个 seed 的 target 覆盖率、训练元数据和 validation 结果。
2. 冻结并审核 generic/grounded targets、provenance gate、协议 manifests、prompt 和 seeds。
3. 完成主表与 ablation 的所有可执行行，汇总三 seed 和分组统计。
4. 补齐 PEFT 六行并在同一硬件重跑吞吐/显存记录。
5. 冻结 robustness slices/proposal crops，逐方法统一评估。
6. 完成三评审 caption-quality ledger，再填 caption 表。
7. 运行正文剩余的 set、learning-curve、tail、prompt、ontology、crop-budget、factor-swap 和 error analyses。
8. 只把自动汇总且通过完整性检查的数值写入 TBD；任何缺行或缺 seed 都继续保留 TBD。

## 缺少的实验

- （缺：权重/接口）GeoChat direct 与 definition；UAVIT-1M compatible evaluation。
- （缺：正式权重根目录）OpenCLIP ViT-L/14、Qwen3-VL-8B。
- （缺：审核数据）official generic-caption、grounded-caption、counterfactual targets，以及三名 blinded raters 的 caption-quality ledger。
- （缺：provenance/数据）正式 unseen-site、location/flight group bootstrap、small-evidence、adverse-capture、complex-background slices。
- （缺：crop 数据/接线）proposal、spatial jitter、irrelevant same-scene crop、crop-budget 曲线。
- （缺：训练实现）linear probe、projector-only、QLoRA、full fine-tuning、parameter-matched attention aggregation。
- （缺：消融实现/输入）shuffled/missing evidence crops、1/4/16/all group learning curves、tail zero/1/4-shot、六 prompt paraphrases。
- （缺：标注/指标接线）factor exact match、hierarchy-consistent F1、单 factor swaps、per-domain macro-F1、evidence-assignment accuracy、ECE。
- （缺：统计实现）CLEAR 对 grounded-caption 的 paired permutation test，以及预注册次要比较的 Holm correction。
- （缺：最终预测和核验）class-frequency/F1 heat map 与错误 gallery。
