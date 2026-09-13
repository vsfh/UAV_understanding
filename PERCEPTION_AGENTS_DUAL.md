# 双卡 Perception + Spatial / What–Where–Verify

此入口从已完成的 train/val proposals 直接联合训练，然后并行验证、校准和测试。
没有 prepare、下载、description 生成或重新挑选训练样本的阶段。
旧的单卡脚本和旧输出不覆盖；默认输出 `outputs/perception_agents_dual/session_disjoint/seed43`。

## 新机器推荐调用

在仓库根目录、包含 PyTorch/Transformers/PEFT 的环境中执行：

```bash
bash runs/perception_agents_dual.sh check --gpus 0,1
bash runs/perception_agents_dual.sh fast --gpus 0,1
```

`fast` 先用真实全分辨率样本短跑四组配置，每组 8 个 optimizer step，排除前 2 步 warmup；
所有配置的有效 batch 都是 8，候选为每卡 batch 4/2 × 梯度检查点关闭/开启。
每组复用相同训练样本和候选，不生成新候选。按峰值 reserved 显存不超过每卡 76 GiB 的条件，
选短跑实测训练吞吐最高的一组，然后**从原 Spatial 权重重新开始完整 3 epoch**，最后完整验证和测试。
短跑权重不用于正式训练。8 步只是硬件配置筛选，不能保证它是所有实现或整套流程的理论最优。
单项 OOM 会排除该项；其他错误立即停止并指向日志，不吞掉程序错误。

默认只评估 `full`：包含三个角色与最多两次修正。它没有删减 agent。
若还要两个消融模式，使用 `fast --mode all`，会增加评估量。

不测速，直接用推荐起始配置：

```bash
bash runs/perception_agents_dual.sh all --gpus 0,1
```

| 参数 | 默认 |
|---|---|
| 并行 | 每卡完整一份 8B + LoRA + Spatial/box head，跨卡同步梯度 |
| 精度 / attention | BF16 / PyTorch SDPA |
| 每卡训练 batch / 累积 | 2 / 2；全局 batch 8 |
| 梯度检查点 | 关闭；用显存换重计算时间，`fast` 会实测开关 |
| 每卡验证 loss / 推理 batch | 4 / 4 |
| 图像尺寸 | 原 min_pixels 65536 / max_pixels 995328，不缩小 |
| epoch / 学习率 | 3；LoRA 2e-5，heads 1e-4 |
| optimizer | fused AdamW |

该默认配置尚未在目标 PRO 6000 上实测占用与加速比；以 `fast` 的显存和吞吐结果为准。
两卡显存各自装一份模型，不能相加视为单卡 160 GB。无需 FSDP/CPU offload。

## 必须同步的已有资产

代码之外，同时保留仓库现有 `src`、其他 perception 脚本、配置和数据，以及：

1. `hf_cache/qwen3-vl/`：同一个基础模型，优先放新机器本地 SSD。
2. `outputs/perception_spatial/session_disjoint/seed43/best/`：完整初始 Spatial adapter、处理器和 heads。
3. `outputs/perception_agents/session_disjoint/seed43/proposals/`：
   `train.json`、`val.json`、`manifest.json`、**`portable_samples.json`** 四个文件。
4. `um7/`：相同图片、definition、bbox 和原 session_disjoint CSV；本次训练不纳入新 synthetic_weak。

`portable_samples.json` 已在旧机器基于原 predict 的 manifest 核验后导出。
它固定 18,227 个 train 和 2,728 个 val 的 record UID、类别、bbox、group、相对图片路径和文件大小；
新机直接读取这些固定样本，不再运行 discovery 选择逻辑。mtime 和仓库绝对路径可以随复制改变。
程序仍核验原 proposals 内容、Spatial checkpoint、类别文件、固定图片大小和配置语义；不重新计算图片像素哈希。
每个 epoch 原有 class-balanced sampler 会从固定 train 索引抽样 16,597 次，这是原训练采样，不是重跑 prepare。
最终评估读取原固定 CSV / no_event 划分，须同步同一数据集。

若目录不同，编辑 YAML 的路径；候选目录也可用 `--proposal-cache /absolute/path/proposals`。
不要重新执行旧的 `perception_agents.sh all`，它会再次调用 prepare。

## 单独测试与短跑

```bash
# 2 个 optimizer step，独立临时输出，再做 4 张 val / 4 张 test
bash runs/perception_agents_dual.sh smoke --gpus 0,1

# 只测速，不跑正式训练；结果给出 selected.yaml
bash runs/perception_agents_dual.sh tune --gpus 0,1

# 已训好后，双卡完整验证校准 + 测试由 all 的末尾自动执行。
# 单独 val/test：test 必须已有同模式、同权重的完整 val calibration。
bash runs/perception_agents_dual.sh val --gpus 0,1 --mode full
bash runs/perception_agents_dual.sh test --gpus 0,1 --mode full
```

`--output` 指向新输出可做下一组实验。此版本从初始 Spatial 重新做 SFT，
不会接续另一个机器正在运行但尚未保存 optimizer 的训练进度。

## 正确性与记录

正例 What/Where/Verify 各占 1/3，负例 What/Verify 各占 1/2；验证的 4 个 Verify 候选共同占一个角色权重。
只对 assistant answer 计算逐样本平均语言 CE，保留答案前一位置用于因果 shift；
不再物化每个图像/提示 token 的全词表 logits。Where 的完整 hidden 和 Spatial memory 不裁剪。
各卡按真实 global group 大小归一，一次 optimizer step 才对可训练参数梯度做 FP32 SUM。
动态不用的 Where head、末尾空 rank、最后不足 8 图的 group 均正确处理。
由于 dropout 和数值运算顺序变化，不承诺与单卡逐步 bitwise 相同。

测试按原索引无重复切片；每卡角色队列真实批处理，最多两次纠正；
合并完整 val 后统一校准阈值，再测试。保留原指标、traces、误修正/恢复统计和校准签名。
新推理入口同时修正旧 Where 把整数 `logits_to_keep` 当 tensor 调用 `.to()` 的问题。

日志：`outputs/perception_agents_dual_logs/`；输出中的 `progress.json` 每 10 个 update 更新，
含 loss、images/s、roles/s、训练 ETA（不含验证）、各卡最大显存。
`history.json` 记录各 epoch 的训练/验证耗时；`evaluation/<mode>/` 保留校准、结果和 traces。
批量推理单图 latency 是批次摊销值，另有双卡 walltime 吞吐；不要与旧串行响应延迟直接当同一指标。

## 环境依据

旧远端为 PyTorch 2.6.0+cu124 / Transformers 5.8.0，不能原样复制该 CUDA build 到 Blackwell。
使用新机已验证可运行 Qwen3-VL 的环境，至少要支持 Blackwell 的 PyTorch CUDA 12.8+ 构建。
脚本不安装或升级环境，也不要求额外编译 flash-attn。SDPA 由当前 PyTorch 选择可用内核。

- [PyTorch 2.7 首次支持 Blackwell 与 CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch distributed 通信与同步训练](https://docs.pytorch.org/docs/stable/distributed)
- [Transformers Qwen3-VL 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

## 本次验证范围

部署时记录在同批交付的 validation_receipt.json：纯 CPU 的控制流、分片/合并、
短尾批次、真实两进程 Gloo 梯度一致性、裁剪 logits 损失和梯度等价检查。
原机器第二张卡有正在进行的图片合成作业，不在它上面抢占执行双卡 8B 测试。
目标 PRO 6000 的全模型显存与吞吐由 `smoke` / `fast` 在新机器测量。
