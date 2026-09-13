"""CPU integration: real Qwen3-VL processor/architecture, tiny random weights, two full-size images."""
import copy
import csv
import json
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p) for p in ('scripts', 'src')]

import torch
import yaml
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from clear_uav.table4 import DiscoverySample, definitions_from_config, bbox_targets
from train_perception_spatial import SpatialPerceptionQwen
from perception_agents_dual_common import role_batches, RoleBatchLoss, cache_path
from perception_agents_batched_runtime import BatchedAgentRuntime


def main():
    torch.set_num_threads(4)
    torch.manual_seed(43)
    config = yaml.safe_load(Path('configs/yaml/perception_agents_dual_pro6000.yaml').read_text())
    config['device'] = 'cpu'
    config['model']['head_dim'] = 16
    config['spatial'].update(interaction_dim=16, num_heads=2, dropout=0.0)
    config['agents']['max_revisions'] = 0
    folder = cache_path(config)
    proposals = json.loads((folder / 'train.json').read_text())
    labels, _ = definitions_from_config(config)
    boxes = bbox_targets(Path(config['data']['bbox_annotations']))
    with (Path(config['data']['root']) / config['protocol'] / 'train.csv').open(encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if r['record_uid'] in proposals and r['source_class'] in labels][:2]
    samples = [DiscoverySample(record_uid=r['record_uid'], label=r['source_class'], presence=True,
        image_path=Path(config['data']['root']) / r['context_path'].removeprefix('data/'), evidence_path=None,
        bbox_1000=boxes[r['context_path'].removeprefix('data/')], group_id=r['content_group_id']) for r in rows]
    source = Path(config['initial_checkpoint'].format(protocol=config['protocol'], seed=config['seed']))
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    tiny = Qwen3VLConfig.from_pretrained(config['model']['path'], local_files_only=True)
    text = tiny.text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 64, 128, 2
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 4, 16
    text.vocab_size = len(processor.tokenizer)
    if hasattr(text, 'rope_parameters'):
        text.rope_parameters['mrope_section'] = [2, 3, 3]
    if getattr(text, 'rope_scaling', None) is not None:
        text.rope_scaling['mrope_section'] = [2, 3, 3]
    vision = tiny.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size, vision.num_heads = 2, 32, 64, 4
    vision.out_hidden_size, vision.deepstack_visual_indexes = 64, [0, 1]
    base = Qwen3VLForConditionalGeneration(tiny)
    vlm = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'],
                                        task_type='CAUSAL_LM', lora_dropout=0))
    model = SpatialPerceptionQwen(vlm, processor.tokenizer.convert_tokens_to_ids('<vis>'), config).float()
    criterion = RoleBatchLoss(model, processor, config)
    model.train()
    losses = {}
    for batch in role_batches(samples, proposals, labels, config):
        loss = criterion(batch)
        assert torch.isfinite(loss), batch['role']
        loss.backward()
        losses[batch['role']] = loss.item()
        model.zero_grad(set_to_none=True)
    runtime = BatchedAgentRuntime(model, processor, config, labels)
    model.eval()
    with torch.inference_mode():
        paired_boxes = runtime.where_batch([{'image_path': s.image_path, 'category': s.label} for s in samples])
    assert len(paired_boxes) == 2 and all(len(box) == 4 for box in paired_boxes)
    results = runtime.predict_batch([s.image_path for s in samples], mode='full')
    assert len(results) == 2 and all(r['agent_calls']['verify'] == 1 for r in results)
    print(json.dumps({'cpu_tiny_qwen_two_full_size_images': 'passed', 'batched_where_boxes': len(paired_boxes), 'losses': losses,
                      'calls': [r['agent_calls'] for r in results]}, indent=2))


if __name__ == '__main__':
    main()
