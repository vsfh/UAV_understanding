# Perception 四个扩展实验

本目录为现有 `Perception-Qwen = Qwen3-VL + LoRA + <vis> 连续框头` 增加四个独立实验。它们保留原模型和原输出目录。新增脚本需要用户显式执行；交付代码本身不会启动完整训练。

这四个版本是面向当前 UAV 数据和代码的可运行技术改造，不宣称完整复现 MVP-LM、ZoomEarth、GETok 或 Perceval，也没有预先声称获得精度提升。

## 1. 环境与共同起点

在服务器上先进入并核对项目位置：

```bash
cd /media/data2/feihong/UAV_understanding
pwd
export PYTHON=/home/feihong/miniconda3/bin/python
export CUDA_VISIBLE_DEVICES=0
```

`runs/perception_runtime.sh` 统一选择解释器：显式 `PYTHON`，兼容 `PYTHON_BIN`，然后查找 PATH 中的 `python`，最后尝试 `$HOME/miniconda3/bin/python`。不安装或更新环境。远端已检查到 `/home/feihong/miniconda3/bin/python` 包含 torch 2.6、transformers 5.8 和 peft 0.19.1；真实 8B 小样本验证结果见第 7 节。

远端 `hf_cache` 位于 SSHFS 挂载，默认 safetensors mmap 的碎片化页读取可能使权重加载和后续 `.to(device)` 很慢。新增 `scripts/perception_extension_runtime.py` 使用本机 transformers 5.8 已支持的 `disable_mmap=True`，以 eager 方式读取现有权重，并保留 `require_local_model` / `enable_offline_mode` 的离线检查。它不复制约 17 GB 模型文件、不修改模型缓存，也不改变训练架构；模型仍需读取全部权重并占用相应主机内存，首次加载可能持续数分钟。`build_baseline` 用相同 loader 恢复冻结 baseline，防止 proposal 准备阶段重新走旧 mmap 加载路径。加载耗时不计为训练算法的速度结论；真实 8B 验证结果见第 7 节。

默认协议和种子为 `session_disjoint / 43`，共同的已训练起点为：

```text
outputs/perception_qwen/session_disjoint/seed43/best/
outputs/perception_qwen/session_disjoint/seed43/config.yaml
```

四份 YAML 都依赖现有 `src/clear_uav`、baseline 训练/测试脚本、模型缓存和 UM7 数据。Warm-start 必须包含 adapter、processor/tokenizer 和 `box_head.pt`；不能用只有模型目录或只有 adapter 的路径代替。若改变 `seed` 或 `protocol`，默认路径模板也会改变，须先生成相应 baseline，或明确将初始化路径固定到共同的 seed43 checkpoint。多种子实验应区分“初始化 checkpoint 的种子”和“新增训练的随机种子”，且保持相同的数据划分。

Zoom 还要求 proposal checkpoint 上级保存的 `config.yaml` 与本次协议、数据配置一致。该检查用于阻止误用来自另一划分的 proposal 缓存。

## 2. 四个版本实际训练什么

| 版本 | 训练路径 | 推理路径 | 实现范围 |
|---|---|---|---|
| `spatial` 空间特征交互 | `<vis>` query 与最终 LLM 层的图像 tokens 做带网格位置的 cross-attention，残差融合后回归；联合语言、L1、GIoU 监督 | 类别生成，回放 `<vis>`，执行空间交互和原图框回归 | 复用现有 Qwen 空间 tokens；没有新增多尺度视觉 backbone。新增残差门初值为零，使初始预测与原框头一致 |
| `zoom` 主动局部放大 | 冻结 baseline 预测 proposal；共享 VLM 学习 global-only 与 global+crop 两分支；监督 gate 判断 crop 是否降低逐样本损失且值得复看成本 | gate 只读取图像产生的冻结状态、proposal 几何、presence 和预算成本；决定是否加入高分辨率 crop | **监督效用门控**，不是 GRPO；不是仅固定裁剪。缺失或无效 proposal 强制跳过 crop |
| `actions` 可学习定位动作 | 先 SFT 学习有限步平移、缩放和 stop 策略；再对可采样的离散动作执行 GRPO | 类别生成与 `<vis>` 回放后，执行贪心框动作 | GRPO 冻结 VLM 与原框头，只优化动作策略；只有类别正确的正样本提供定位奖励，因此这一阶段不会改善分类或拒识 |
| `reward` 视觉奖励模型 | 从冻结 baseline 图像 tokens 提取独立视觉特征；训练类别条件 ROI 质量模型，再冻结它，与真值规则奖励混合优化相同的动作策略 | 最终推理仍运行 learned actions；训练用 RM 不作为测试真值来源 | 监督目标为 contextual ROI IoU，可加入仅来自训练集的人工框偏好；不是通用大模型 PRM，也不是已证明可靠的语义证据判官 |

Zoom 的两图分支明确以**完整原图**为坐标参考，crop 的原图边界写入提示。训练 target 与最终 `bbox_1000` 始终为原图坐标；不会把 crop 内坐标误当作原图坐标。crop 使用预测 ROI 扩展、边界裁剪与最小尺寸处理，训练 GT 仅用于语言/框损失和 gate 效用标签。

Actions 的 GRPO 具有真实动作采样概率、旧策略概率、参考策略约束和组内优势；它优化的是离散定位决策，不是假定确定性 MLP 框拥有 token log-prob。由于类别固定，应把它作为定位后训练实验解释。

## 3. 统一入口与分阶段运行

```bash
bash runs/perception_extensions.sh help
bash runs/perception_extensions.sh spatial all
bash runs/perception_extensions.sh zoom all
bash runs/perception_extensions.sh actions all
bash runs/perception_extensions.sh reward all
```

完整顺序执行四组默认实验：

```bash
bash runs/perception_extensions.sh all all
```

这个命令实际执行完整训练和测试，可能耗时较长；运行前检查 GPU、数据和初始化 checkpoint。`all` 变体只接收 `train / val / test / all`，不接收共享 YAML、输出目录或额外参数，避免四组实验写入同一目录。需要限制样本或改变输出时，分别运行各版本。

单个版本的入口为：

```text
bash runs/perception_extensions.sh VARIANT MODE [config.yaml] [options]
```

不提供 YAML 时使用 `configs/yaml/perception_VARIANT.yaml`。`train` 与 `all` 在动作和奖励模型版本包含多个阶段；希望只运行 SFT、RM 或 GRPO 时须使用相应阶段模式。

### Spatial

```bash
bash runs/perception_spatial.sh train
bash runs/perception_spatial.sh val
bash runs/perception_spatial.sh test
```

支持 `--initial-checkpoint PATH`、`--epochs`、样本限制和 `--output PATH`。Checkpoint 保存 adapter、原框头与空间交互头，测试读取训练时保存的配置。

### Zoom

```bash
# 显式建立 train/val 的冻结 proposal 缓存，不准备 test。
bash runs/perception_zoom.sh prepare
bash runs/perception_zoom.sh train
bash runs/perception_zoom.sh val

# test 的 proposal 只在测试前显式准备；其预测特征不使用 GT。
bash runs/perception_zoom.sh prepare-test
bash runs/perception_zoom.sh test
```

`all` 自动按以上顺序执行。单独 `train` 不会生成缓存，也不会访问 test 来造缓存。每个 split 的 `.pt` 缓存和 `.manifest.json` 记录 source checkpoint 的内容哈希、source config 哈希、protocol、split 成员/图像元信息哈希。缓存不保存 target 标签或 target 框；初始化或 split 改变会被拒绝，须显式重建。

默认 `zoom.review_cost = 0.1`，训练从 `zoom.train_review_cost_range` 采样成本，gate 监督为：

```text
review = (global_loss - crop_loss - review_cost > utility_margin)
```

请记录 gate 复看率与成本曲线。验证选择的是实际 gate 决策下的损失加成本，测试不根据 GT 改 gate 阈值。Checkpoint 保存 `zoom_gate.pt`、`box_head.pt`、adapter、processor 和 `zoom_manifest.json`。

### Actions

```bash
bash runs/perception_actions.sh sft
bash runs/perception_actions.sh val --stage sft
bash runs/perception_actions.sh test --stage sft
bash runs/perception_actions.sh grpo
bash runs/perception_actions.sh val --stage grpo
bash runs/perception_actions.sh test --stage grpo
```

SFT 与 GRPO 输出分别位于运行根目录下的 `sft/` 与 `grpo/`。`--checkpoint PATH` 表示明确的初始化 checkpoint，不恢复优化器。`train` 顺序运行 SFT 和 GRPO，仅使用各阶段内部 validation；`all` 额外对两个阶段分别进行 val/test 评估。

### Visual reward

```bash
bash runs/perception_reward.sh prepare
bash runs/perception_reward.sh rm
bash runs/perception_reward.sh quality
bash runs/perception_reward.sh sft
bash runs/perception_reward.sh grpo
bash runs/perception_reward.sh val
bash runs/perception_reward.sh test
```

`prepare` 仅提取 train/val 特征，`rm` 训练并根据 validation 选择质量模型，`quality` 报告验证集上的质量评分误差和成对排序准确率，`sft` 生成本版本的动作 SFT 起点，`grpo` 冻结 RM 后训练动作策略。也可省略本版本 `sft`，改为在 `grpo` 使用 `--checkpoint` 指定共同的 actions SFT checkpoint。`train` 顺序执行 prepare、rm、sft、grpo；`all` 再进行质量模型和感知任务评估。测试感知任务不读取 target 作为 policy 输入，也不会用 test 训练 RM。

## 4. 公平比较规则 GRPO 和 RM-GRPO

**两组 GRPO 必须从同一个 actions SFT checkpoint 开始。** 不能将“baseline 原框头 + 新随机动作头”和“已做 actions SFT 的动作头”作为不同 reward 的对照。

以下使用 Python 入口明确指定相同起点，并分别写入两个新目录：

```bash
shared_sft=./outputs/perception_actions/session_disjoint/seed43/sft/best

"$PYTHON" scripts/train_perception_actions.py \
  --config configs/yaml/perception_actions.yaml --stage grpo \
  --checkpoint "$shared_sft" --output ./outputs/perception_rule_grpo_comparison

# 先用 perception_reward.yaml 完成 prepare 和 rm。
"$PYTHON" scripts/train_perception_actions.py \
  --config configs/yaml/perception_reward.yaml --stage grpo \
  --checkpoint "$shared_sft" --output ./outputs/perception_rm_grpo_comparison
```

除 `reward_model.enabled`、RM 路径和混合权重外，对齐：训练样本、seed、总 optimizer updates、batch/accumulation、group size、动作步数、GRPO clip/KL、imitation anchor、规则奖励和输入分辨率。两组分别用自己的 validation 输出选 presence 阈值，再固定阈值评估 test。RM 开销须计入训练总成本。

建议至少报告：原 Perception-SFT、同预算 continued SFT、spatial、zoom、actions SFT、actions rule-GRPO、actions RM-GRPO。新结构与后训练收益分开比较。RM 的额外价值应通过“同 SFT 起点、同训练预算”的对照说明；只有 IoU 拟合标签时，不能将它宣称为超越 IoU 的人工偏好奖励。

## 5. Tiny smoke：只验证管线

Spatial 与 Zoom 可完整运行小样本链路：

```bash
bash runs/perception_spatial.sh all --epochs 1 \
  --max-train-samples 2 --max-val-samples 2 --max-test-samples 2 \
  --output ./outputs/perception_smoke/spatial

bash runs/perception_zoom.sh all --epochs 1 \
  --max-train-samples 2 --max-val-samples 2 --max-test-samples 2 \
  --output ./outputs/perception_smoke/zoom
```

Zoom 的 prepare/train/evaluate 样本限制必须一致，split 成员哈希不同会被拒绝。评估的样本限制以本次显式参数为准，不会从训练保存的 tiny 限制静默继承。小样本缓存与正式训练放在不同输出目录。仅一两个样本仍须加载完整 8B 模型，也仍需要足够 GPU 显存。

Actions 使用阶段模式限制 optimizer updates：

```bash
bash runs/perception_actions.sh sft --steps 1 --epochs 1 \
  --max-train-samples 2 --max-val-samples 2 --output ./outputs/perception_smoke/actions

bash runs/perception_actions.sh grpo --steps 1 --epochs 1 \
  --max-train-samples 2 --max-val-samples 2 \
  --checkpoint ./outputs/perception_smoke/actions/sft/best \
  --output ./outputs/perception_smoke/actions

bash runs/perception_actions.sh val --stage grpo --max-val-samples 2 \
  --output ./outputs/perception_smoke/actions
```

Reward 的 smoke 分为 feature prepare、RM 拟合和小步 GRPO。下面复用前面 actions smoke 的 SFT 起点，`--output` 同时设置专用 RM cache/checkpoint 路径：

```bash
bash runs/perception_reward.sh prepare --max-train-samples 2 --max-val-samples 2 \
  --output ./outputs/perception_smoke/reward
bash runs/perception_reward.sh rm --rm-epochs 1 --output ./outputs/perception_smoke/reward
bash runs/perception_reward.sh quality --max-val-samples 2 --output ./outputs/perception_smoke/reward
bash runs/perception_reward.sh grpo --steps 1 --epochs 1 \
  --max-train-samples 2 --max-val-samples 2 \
  --checkpoint ./outputs/perception_smoke/actions/sft/best \
  --output ./outputs/perception_smoke/reward
bash runs/perception_reward.sh val --max-val-samples 2 --output ./outputs/perception_smoke/reward
```

不要把 tiny RM 缓存与全量 policy 训练混用。

Tiny subset 可能不包含负样本或完整类别，会使用 fallback calibration；这些数值不可进入 paper 主表，也不能说明 gate/RM/GRPO 已有效。

## 6. 结果、指标与费用

默认输出根分别为：

```text
outputs/perception_spatial/{protocol}/seed{seed}
outputs/perception_zoom/{protocol}/seed{seed}
outputs/perception_actions/{protocol}/seed{seed}
outputs/perception_reward/{protocol}/seed{seed}
```

保留每次运行的 YAML、history、完整 learned heads、adapter 与验证/测试逐记录 JSON。验证选择 checkpoint 和 presence 阈值，test 不参与选择。主任务指标沿用现有 `table4_metrics`：`g_map50`、`ap50`、`c_f1`、`p_ap`、`p_precision/p_recall/p_f1`、`n_fpr`、`valid_rate`、`median_ms`、`mean_calls/max_calls`。

现有 `n_fpr` 是负样本的 **presence_score 过阈值比例**，不要求同时输出有效类别和框。新增 `final_output_n_fpr` 单独记录实际事件输出误报率，要求 presence 过阈值、有效类别和有效框，保留原 `n_fpr`；论文必须注明二者口径。

Zoom 额外输出 review rate、image views、proposal/refinement visual tokens、crop 边界和 gate probability。其总延迟是“已测 proposal 时间 + 在线 refinement 时间”，字段中明确说明 gate 与 cache IO 不含在内；不能把缓存 proposal 当作零成本。需要严格在线延迟时，应补包含 proposal、gate、crop、refinement 的端到端测量。

同时记录 GPU 型号、峰值显存、训练 wall time、GPU-hours、输入像素/视觉 token 预算、GRPO 采样数量和最大动作步数。Zoom 最多输入原图加 crop，RM 需要额外特征准备和质量模型训练；这些成本影响公平性。不同方法的 `num_calls` 与 `timing_scope` 不同，应连同表格报告。

## 7. 验证记录（2026-09-10 远端实测）

验收环境为 `jizheng_fei`、`/home/feihong/miniconda3/bin/python`、Qwen3-VL-8B-Instruct、NVIDIA RTX 4090。实际结果如下：

| 检查 | 结果 |
|---|---|
| CPU 数值与契约测试 | **51/51 通过**：Spatial 11、Zoom 10、Actions 18、Reward 12 |
| YAML / Python / shell | 四份 YAML 无重复键；Python AST、六个新增 shell 的 `bash -n` 和统一入口分发通过 |
| Spatial | 真实前反向与一个优化步、残差门更新、checkpoint 保存与重载、val/test 通过 |
| Zoom | 无 GT proposal 准备、global 与 global+crop 两分支训练、gate 更新、保存重载、val/test 通过 |
| Actions | 一个 SFT 更新、两次 GRPO 更新、保存重载、val/test 通过；第二次更新出现非零 clipping |
| Reward | 冻结视觉特征准备、RM 拟合与质量评分、同一 SFT 起点的 RM-GRPO 两次更新、保存重载、val/test 通过 |
| RM CLI 超参覆盖 | checkpoint 以 1 epoch 训练，在运行配置写 9 epochs 时仍可冻结评分；评分无梯度，实际训练设置仍从 checkpoint 读取 |
| 原 baseline | 原训练、测试、YAML、modeling.py、table4.py 的 SHA-256 与实现前一致 |

验收输出在 `reports/perception_extensions_2026-09-10/smoke_v1/`，日志在其上级目录。训练和评估均限定为各 split 的前 **2 个样本**；新增分支使用 `min_pixels=65536, max_pixels=262144`，Zoom proposal 仍使用 baseline 保存的输入配置。规则 GRPO 与 RM-GRPO 从完全相同的 actions SFT checkpoint 出发。完整数据训练没有启动，验收结束时两个 GPU 均已空闲。

这些 tiny 样本没有负图像，不能验证完整类别覆盖、负样本误报率或统计显著性；相关负样本和非法框分支由 CPU 测试覆盖。Tiny Zoom 推理中 gate 选择了 global 分支，而原图+crop 的训练分支已实际执行。以上是实现连通性验收，**不是精度提升结论，也不能用于 paper 主表**。

冻结 RM 后，未使用的 RM 训练超参数可以与 YAML 默认值不同；质量评估采用 checkpoint 保存的实际训练设置。数据 UID、缓存、GT 来源、偏好与目标定义变更仍会被拒绝。

重跑 CPU 检查：

```bash
PYTHON=/home/feihong/miniconda3/bin/python
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" -m unittest discover -s tests -p 'test_perception_*.py' -v
```
