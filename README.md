# UAV Understanding

实验代码采用一项实验一个 Python 文件、一个 YAML 配置的结构。训练脚本直接读取 YAML，
不再经过 experiment suite，也不再拼接子进程命令。

## 目录

```text
configs/yaml/                  实验配置
scripts/openclip_linear_probe.py
scripts/openclip_finetune.py
scripts/qwen_lora.py
scripts/test_openclip.py
scripts/test_qwen.py
scripts/test_geochat.py
runs/                          YAML 对应的一键入口
```

默认数据和模型路径分别是 `./um7` 与 `./hf_cache`。当前训练和测试配置统一使用 seed 43；
主实验覆盖 `forward_temporal`、`session_disjoint`、`unseen_site`，label-crop 和
label-context 消融只跑 `session_disjoint`。

## 安装

```bash
pip install -e .
```

OpenCLIP 独立环境可以使用：

```bash
bash runs/setup_openclip_env.sh
```

模型下载与数据检查仍然独立：

```bash
bash runs/00_download_paper_models.sh
bash runs/01_validate_data.sh ./um7
```

## 训练

OpenCLIP 仅训练分类头：

```bash
bash runs/openclip_linear_probe.sh
```

OpenCLIP 完整视觉编码器微调：

```bash
bash runs/openclip_full_finetune.sh
```

Qwen3-VL LoRA：

```bash
bash runs/qwen_lora.sh
```

脚本的第一个参数可以换成另一个 YAML：

```bash
bash runs/qwen_lora.sh configs/yaml/qwen_lora.yaml
```

## 测试

```bash
bash runs/test_openclip.sh
bash runs/test_qwen.sh
bash runs/test_geochat.sh
```

GeoChat 测试需要官方源码及其独立环境。源码默认放在 `third_party/GeoChat`；也可以分别通过
`GEOCHAT_ROOT` 和 `GEOCHAT_PYTHON` 指定源码目录与 Python。该测试以 seed 43 对 context 图像
分别运行 YAML 中的 direct prompt 和 definition prompt。

OpenCLIP 测试配置中的 `checkpoints` 是列表；Qwen 测试配置中的 `adapters` 也是列表。
增加同一模型族的新 checkpoint 只需增加一项，无需修改测试循环。增加新的模型族时，可以复用
`src/clear_uav/evaluation.py` 中的数据、指标和结果写入函数。

## TensorBoard

每个 run 都把曲线写入自己的 `tensorboard/` 子目录：

```bash
tensorboard --logdir outputs --port 6006
```

OpenCLIP 记录 step loss、learning rate、epoch loss 和 validation macro-F1；Qwen 使用
Transformers TensorBoard callback 记录 loss、learning rate、epoch 和训练吞吐。

## 配置原则

- 路径、protocol、seed、epoch、batch size 和 learning rate 全部放在 YAML。
- bash 只负责选择 CUDA 设备、Python 环境并把 YAML 交给对应 Python 文件。
- 已存在最终 checkpoint 时默认跳过该 run；修改 `skip_existing: false` 可重新训练。
- teacher caption 生成、模型下载和数据准备工具与实验训练解耦，仍保留在 `scripts/` 和 `runs/`。
