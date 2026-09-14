"""CPU integration checks using a tiny random Qwen; no data, 8B weights or training."""
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from train_perception_spatial import SpatialPerceptionQwen, LORA_PATTERNS
from perception_spatial_head import save_spatial_heads, load_spatial_heads


def main():
    torch.set_num_threads(2)
    torch.manual_seed(43)
    qwen_config = Qwen3VLConfig(
        text_config=dict(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=256,
                         rope_scaling=dict(rope_type="default", mrope_section=[2, 3, 3], mrope_interleaved=True)),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=64, patch_size=2, temporal_patch_size=2,
                           spatial_merge_size=2, num_position_embeddings=16, deepstack_visual_indexes=[0]),
        image_token_id=110, video_token_id=114, vision_start_token_id=111, vision_end_token_id=112,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
    )
    base = Qwen3VLForConditionalGeneration(qwen_config)
    base_state = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
    vlm = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0., target_modules=LORA_PATTERNS["projector_llm"],
        task_type="CAUSAL_LM", trainable_token_indices={
            "model.language_model.embed_tokens": [113], "lm_head": [113]}))
    ids = torch.tensor([[1, 111, 110, 110, 110, 110, 110, 110, 112, 3, 113, 2]] * 2)
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  pixel_values=torch.randn(48, 24), image_grid_thw=torch.tensor([[1, 4, 6], [1, 6, 4]]),
                  mm_token_type_ids=(ids == 110).long())
    vlm.eval()
    with torch.no_grad():
        before = vlm(**inputs, use_cache=False).logits
    config = dict(model=dict(head_dim=16), spatial=dict(interaction_dim=16, num_heads=4, dropout=0.),
                  spatial_lora=dict(enabled=True, rank=4, alpha=4, dropout=0.))
    model = SpatialPerceptionQwen(vlm, 113, config).eval()
    with torch.no_grad():
        after = model.vlm(**inputs, use_cache=False).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    print("PASS tiny Qwen + PEFT: zero-init logits are identical")

    # Synthetic backward only; no optimizer or parameter update is performed.
    model.train()
    model.vlm.enable_input_require_grads()
    model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    labels = ids.clone()
    labels[:, :9] = -100
    language, boxes = model({**inputs, "labels": labels})
    target = torch.tensor([[.1, .2, .8, .9], [.2, .1, .7, .8]])
    (language + (boxes - target).square().mean()).backward()
    for branch in model.spatial_adapter.branches:
        assert branch.up.weight.grad is not None and branch.up.weight.grad.abs().sum() > 0
    assert model.box_head[-1].weight.grad.abs().sum() > 0
    assert model.spatial_head.residual_gate.grad.abs().sum() > 0
    assert torch.isfinite(boxes).all() and ((boxes >= 0) & (boxes <= 1)).all()
    print("PASS tiny Qwen: checkpointed language/ROI backward reaches both visual streams")

    # Perturb new weights to make checkpoint/generation checks nontrivial.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for branch in model.spatial_adapter.branches:
            branch.up.weight.normal_(std=.02)
    model.eval()
    calls = []
    handle = model.vlm.get_base_model().model.visual.register_forward_hook(lambda *args: calls.append(1))
    prompt = {**inputs, "input_ids": ids[:, :9], "attention_mask": inputs["attention_mask"][:, :9],
              "mm_token_type_ids": inputs["mm_token_type_ids"][:, :9]}
    with torch.no_grad():
        generated = model.vlm.generate(**prompt, max_new_tokens=2, do_sample=False,
                                       use_cache=True, pad_token_id=0, eos_token_id=2)
        reference_loss, reference_boxes = model({**inputs, "labels": labels})
        reference_logits = model.vlm(**inputs, use_cache=False).logits
    handle.remove()
    assert generated.shape[0] == 2 and calls
    print("PASS tiny Qwen: autoregressive generation and full ROI replay")

    with tempfile.TemporaryDirectory(prefix="spatial-lora-check-") as folder:
        checkpoint = Path(folder)
        model.vlm.save_pretrained(checkpoint, save_embedding_layers=False)
        save_spatial_heads(model, checkpoint)
        restored_base = Qwen3VLForConditionalGeneration(qwen_config)
        restored_base.load_state_dict(base_state)
        restored = SpatialPerceptionQwen(PeftModel.from_pretrained(restored_base, checkpoint), 113, config)
        load_spatial_heads(restored, checkpoint, require_spatial=True)
        restored.eval()
        with torch.no_grad():
            restored_loss, restored_boxes = restored({**inputs, "labels": labels})
            restored_logits = restored.vlm(**inputs, use_cache=False).logits
        torch.testing.assert_close(reference_logits, restored_logits, rtol=0, atol=0)
        torch.testing.assert_close(reference_boxes, restored_boxes, rtol=0, atol=0)
        torch.testing.assert_close(reference_loss, restored_loss, rtol=0, atol=0)
    print("PASS tiny Qwen: PEFT + ROI + spatial head + spatial adapter round trip")


if __name__ == "__main__":
    main()
