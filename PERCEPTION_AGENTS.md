# What / Where / Verify 三角色方案

已实现代码，尚未完成 8B 模型的 GPU 训练或完整推理验证；没有新实验效果数值。

## 模型与分工

一份现有 Qwen Perception + Spatial 权重、一个共享 LoRA，按角色串行调用。不是三份独立 8B 模型，也不是三个独立训练出的专家。初始化使用 `outputs/perception_spatial/session_disjoint/seed43/best`，同时恢复 LoRA、bbox MLP 和空间交互头。

1. **What** 看完整原图，输出 `类别<vis>` 或 `no_event`。
2. **Where** 看完整原图及 What 给出的类别，回放该类别与 `<vis>`，由现有空间交互头和 bbox MLP 输出全图绝对位置。重新定位会重新预测四个坐标；没有沿用旧动作策略的小步移动。
3. **Verify** 看完整原图，候选框以紫色边框标在原图上，同时提供候选类别与坐标。输出 A（通过）、B（重新定位）、C（重新分类）、D（无事件）。只传一张完整图片，兼容现有 Spatial 的单图 token packing。

What / Verify 使用共享 VLM 的语言输出；新增 Spatial 分支仍在 Where 的 bbox 路径中使用，未声称它也直接修改语言输出。

What 初次输出无事件时仍调用 Verify，因此可以纠正漏检。B 返回 Where，C 返回 What 后重新执行 Where；每次修正后再次 Verify，最多修正两轮。D 直接输出无事件。修正次数用尽或出现非法判断时明确弃答，保留原因。共享权重可能共同犯错，Verify 的“通过”不是正确性的保证。

## 训练

`prepare` 冻结现有 Spatial 模型，用原来的提示词生成 train / val 的真实候选框和类别。只缓存候选，不改变现有图片。校验 train / val 的记录、内容组和图片路径不相交；缓存绑定来源检查点、类别定义、图片元数据与监督目标。

`train` 在这份 Spatial 权重上进行多任务 SFT：

- What：事件类别 / 无事件语言交叉熵。
- Where：GT 类别条件下的语言交叉熵 + bbox L1 + GIoU；包括普通定位和反馈后的全图重定位。
- Verify：A/B/C/D 语言交叉熵。候选混合真实 Spatial 预测、正确框、明显偏移/缩放框、随机框、错类别和空候选。标签规则为：负图 D；正图错类/漏类 C；类别正确但 IoU<0.5 或无框 B；类别和框均正确 A。

每张正图一次采样训练三个角色，负图训练 What 和 Verify。各角色在一张图中的总权重相同；验证阶段使用多个固定候选、平均 Verify 损失。训练时按图进行类别平衡，Verify 候选随记录种子、epoch 和采样位置轮换。GT 只用于监督构造和评估；正式推理接口仅接收图片路径。

保留现有像素预算 65,536–995,328；单图 batch=1，梯度累积 8。各角色依次前向和反向，避免同时保留三个角色的激活。这里增加训练与推理调用次数，不能宣称成本与原单次流程相同。

## 运行

在仓库根目录运行。使用现有 Python 环境，不下载模型、不安装依赖。`PYTHON` 可指定解释器，`GPU` 或 `--gpu` 选择卡。

```bash
bash runs/perception_agents.sh plan
GPU=0 bash runs/perception_agents.sh smoke
GPU=0 bash runs/perception_agents.sh all
```

默认不带参数只打印计划。`smoke` 使用独立目录、4 个 train / 4 个 val / 4 个 test 样本和一个优化步骤，是运行检查，不是实验结果。已存在的训练目录不会覆盖，重复实验用 `--output` 指定新目录。

分阶段运行：

```bash
GPU=0 bash runs/perception_agents.sh prepare
GPU=0 bash runs/perception_agents.sh train
GPU=0 bash runs/perception_agents.sh val --mode all
GPU=0 bash runs/perception_agents.sh test --mode all
```

prepare / train 的样本限制必须一致。中断在 prepare 写完 manifest 之前，应使用新输出目录重新开始；此版本不提供逐条缓存续跑或 optimizer 断点续训。

## 评估与消融

同一训练后检查点支持三个模式：

| 模式 | 流程 | 作用 |
|---|---|---|
| `no_verify` | What → Where | 看角色 SFT 后的初始预测 |
| `verify_only` | What → Where → Verify | 看检查 / 拒绝本身的影响，不反馈 |
| `full` | What → Where → Verify → 有限反馈 | 看反馈是否真的纠正了错误 |

```bash
GPU=0 bash runs/perception_agents.sh all --mode all
```

每个模式在 val 上独立校准，再用于 test。校准绑定检查点、角色源码、空间前向和评测依赖、提示词、像素预算、标签定义和修正次数；不允许用很小的 val 冒充完整 test 的校准。

通过候选的最终分数为 `What 存在分数 × Verify 的 A 分数`，是组合排序分数，不是经过概率校准的事件存在概率。What 原始分数和 Verify 分数分别保存。D / 弃答输出不包含事件框；`no_verify` 保留 What 原始分数。

默认输出位于 `outputs/perception_agents/session_disjoint/seed43/`：

- `proposals/`：真实 Spatial 候选和来源 manifest。
- `best/`：训练后 adapter、processor、bbox/spatial 头、agent_schema。
- `config.yaml`、`history.json`：实际设置及损失。
- `evaluation/{mode}/{val,test}_results.json`：原有论文指标、最终误报率、调用次数、修正率、弃答率。
- `evaluation/{mode}/{val,test}_traces.jsonl`：每个记录的初始预测、所有反馈、最终决定。

额外统计 `agent_joint50_recovered` / `agent_joint50_lost`（相对该图第一次 What/Where，类别正确且 IoU≥0.5，置信度过滤前），便于区分改对与改错。与旧 Spatial 直接比较仍包含额外 SFT 和计算预算；上述三个模式隔离检查与反馈效果，但不能独立证明“多 agent”结构本身优于所有同预算方案。

## 已做检查

22 项本地单元测试通过，覆盖反馈恢复、修正上限、GT 隔离、候选轮换、角色权重、数据监督、校准绑定和 trace。

远端只读 CPU 检查使用真实 Spatial processor 和一张训练图片，三种角色成功编码，单图网格均为 `[1,54,72]`；What / Where / Verify 序列长度分别为 1440 / 1515 / 1567。A/B/C/D 是四个独立单 token；真实 PyYAML 的 plan CLI 检查通过。没有加载 VLM 权重或运行 GPU 训练。

本地运行测试：

```bash
python -m unittest discover -s tests -p 'test_perception_agents_*.py' -v
```
