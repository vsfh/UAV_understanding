# Experiment entrypoints

| Experiment | Python | YAML | Bash |
|---|---|---|---|
| OpenCLIP linear probe | `scripts/openclip_linear_probe.py` | `configs/yaml/openclip_linear_probe.yaml` | `runs/openclip_linear_probe.sh` |
| OpenCLIP full fine-tune | `scripts/openclip_finetune.py` | `configs/yaml/openclip_full_finetune.yaml` | `runs/openclip_full_finetune.sh` |
| Qwen LoRA | `scripts/qwen_lora.py` | `configs/yaml/qwen_lora.yaml` | `runs/qwen_lora.sh` |
| OpenCLIP test | `scripts/test_openclip.py` | `configs/yaml/openclip_test.yaml` | `runs/test_openclip.sh` |
| Qwen test | `scripts/test_qwen.py` | `configs/yaml/qwen_test.yaml` | `runs/test_qwen.sh` |

每个训练 YAML 当前定义 3 protocols × 3 seeds，共 9 个 run。不同算法由不同 Python 文件运行，
没有总 suite、shard scheduler 或动态命令拼接层。

配置修改示例：复制一个 YAML，修改参数，然后把它作为 bash 的第一个参数：

```bash
cp configs/yaml/openclip_full_finetune.yaml configs/yaml/openclip_full_finetune_ablation.yaml
bash runs/openclip_full_finetune.sh configs/yaml/openclip_full_finetune_ablation.yaml
```

训练输出：

```text
outputs/openclip_linear_probe/<protocol>/seed<seed>/
outputs/openclip_full_finetune/<protocol>/seed<seed>/
outputs/qwen_lora/<protocol>/seed<seed>/
```

测试结果：

```text
results/openclip/<checkpoint-name>/<protocol>/seed<seed>_val.json
results/qwen/<adapter-name>/<protocol>/seed<seed>_val.json
```
