# 弱势类图片生成

使用已经训练完成的 crop LoRA、inpainting LoRA 和现有 description，直接生成弱势类的 crop、original 与 bbox。入口复用原有 `scripts/uav_synthesis_pipeline.py`；不重新生成 description，也不下载模型或训练 LoRA。

## 直接调用

在远端仓库执行：

```bash
cd /media/data2/feihong/UAV_understanding
bash runs/uav_synthesis_weak.sh plan
bash runs/uav_synthesis_weak.sh smoke --gpu 1
bash runs/uav_synthesis_weak.sh all --gpu 1
```

`plan` 查看现有 description、选定类别及生成计划。`smoke` 每类生成 2 组，保存到独立的 smoke 目录。`all` 默认每类最多使用 3,000 条已经生成的 description；如果某类数量不足，就使用该类已有的数量。

默认类别在 `configs/yaml/uav_synthesis_weak.yaml` 中配置：`building_demolition`、`glass_greenhouse`、`floating_garbage`、`collapsed_house_wall`、`dangerous_building`。它们是当前优先补充的类别，可按实验目标修改，例如：

```bash
bash runs/uav_synthesis_weak.sh all --gpu 1 \
  --classes collapsed_house_wall glass_greenhouse \
  --per-class 1000 \
  --selection-dir outputs/uav_synthesis_weak_two_classes/seed43 \
  --synthetic-root um7/synthetic_weak_two_classes
```

已有一次生成记录后，改变类别或数量时请同时使用新的 `--selection-dir` 与 `--synthetic-root`，以便不同实验的样本选择保持可复现。

## 实际流程

1. 从已有 description 中选择指定类别，冻结本次选择清单。
2. 按 `chunk_size` 分块；每块先生成 crop，再生成匹配的 original。
3. original 生成沿用原 pipeline：按训练输入尺寸创建白底画布，按已有样本的框比例放置 crop，记录 bbox，再补全背景并保留放置区域。
4. 导出本次生成数据的训练记录。生成图片保留原数据的层级格式，新的数据根目录默认为 `um7/synthetic_weak`。

分块默认 100 组，因此不必等所有类别的 crop 生成完，便能得到成套的 crop、original 和 bbox。两个模型按阶段依次加载，同一时刻只执行一个生成子进程。

本次记录保存在 `outputs/uav_synthesis_weak/seed43`，图片保存在 `um7/synthetic_weak`；smoke 对应目录为 `outputs/uav_synthesis_weak_smoke/seed43` 与 `um7/synthetic_weak_smoke`。原训练图片和已有 synthesis 产物保持独立。

每组 bbox 与生成信息保存在新图片根目录的 `metadata/<synthetic_id>.json`；导出汇总为 `synthetic_manifest.jsonl` 和 COCO 格式的 `crop_bboxes_all_photos/crop_bboxes.json`。bbox 是 crop 在 original 中的放置坐标，生成区域在扩图后会重新贴回，以保持 crop 对应的像素区域。

## 续跑与检查

```bash
# 查看当前生成进度。
bash runs/uav_synthesis_weak.sh status

# 只准备并冻结选择清单。
bash runs/uav_synthesis_weak.sh prepare

# 中断后使用原命令续跑，复用已冻结清单，跳过已完成产物。
bash runs/uav_synthesis_weak.sh all --gpu 1

# 重新导出已有图片的训练记录。
bash runs/uav_synthesis_weak.sh export
```

`--chunk-size`、`--crop-inference-steps`、`--inpaint-inference-steps` 可以覆盖 YAML；默认分别为 100、50、30。环境变量 `PYTHON` 可覆盖默认 `.venvs/uav_synthesis/bin/python` 解释器。

检查远端状态时，description 生成进程已经不在运行，因此此脚本不执行停进程操作。运行前用 `nvidia-smi` 确认所选 GPU；其他 Perception agent 准备或训练进程不受本脚本控制。
