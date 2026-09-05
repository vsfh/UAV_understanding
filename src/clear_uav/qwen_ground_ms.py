from __future__ import annotations

import json
import math
import random
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionModel,
)

from clear_uav.data import Sample, iter_csv_rows
from clear_uav.run_progress import phase
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_snapshot(model_root: Path) -> Path:
    if (model_root / "config.json").is_file():
        return model_root
    return next((model_root / "snapshots").glob("*/config.json")).parent


def load_qwen_vision(model_root: Path, device: torch.device) -> tuple[nn.Module, object]:
    with phase("Locate vision model and read configuration"):
        snapshot = model_snapshot(model_root)
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        config.vision_config._attn_implementation = "sdpa"
    with phase("Create vision model structure (meta device)"):
        with torch.device("meta"):
            vision = Qwen3_5MoeVisionModel(config.vision_config)
        # Non-persistent buffer must be materialized outside the meta device.
        rotary = vision.rotary_pos_emb
        rotary.inv_freq = 1.0 / (
            rotary.theta
            ** (torch.arange(0, rotary.dim, 2, dtype=torch.float32) / rotary.dim)
        )

    weight_path = snapshot / "outside.safetensors"
    state = {}
    with ExitStack() as stack:
        with phase(f"Open weight file: {weight_path} (first access may be slow)"):
            weights = stack.enter_context(
                safe_open(weight_path, framework="pt", device="cpu")
            )
        keys = [key for key in weights.keys() if key.startswith("model.visual.")]
        with phase(f"Extract {len(keys)} vision tensors (mapped views, not disk-byte progress)"):
            for key in tqdm(keys, desc="Vision tensors", unit="tensor", dynamic_ncols=True):
                state[key.removeprefix("model.visual.")] = weights.get_tensor(key)
    with phase("Assign vision weights"):
        vision.load_state_dict(state, strict=True, assign=True)
    with phase(f"Materialize vision weights on {device} (may read mapped storage)"):
        vision.to(device=device, dtype=torch.bfloat16)
    with phase("Load image processor"):
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True).image_processor
    return vision, processor


def load_bbox_targets(path: Path) -> dict[str, torch.Tensor]:
    coco = json.loads(path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in coco["images"]}
    targets = {}
    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        targets[image["file_name"]] = torch.tensor(
            [
                (x + width / 2) / image["width"],
                (y + height / 2) / image["height"],
                width / image["width"],
                height / image["height"],
            ],
            dtype=torch.float32,
        )
    return targets


def read_ground_samples(
    inputs_csv: Path,
    data_root: Path,
    labels_csv: Path | None = None,
    limit: int | None = None,
) -> list[Sample]:
    private_labels = None
    if labels_csv is not None:
        private_labels = {
            row["record_uid"]: row["source_class"] for row in iter_csv_rows(labels_csv)
        }
    samples = []
    for row in iter_csv_rows(inputs_csv):
        relative = Path(row["context_path"])
        if relative.parts[0] == "data":
            relative = Path(*relative.parts[1:])
        context_path = data_root / relative
        if not context_path.is_file():
            continue
        samples.append(
            Sample(
                record_uid=row["record_uid"],
                label=(private_labels or {}).get(row["record_uid"], row.get("source_class", "")),
                context_path=context_path,
                evidence_path=context_path,
                content_group_id=row["content_group_id"],
                session_id=row["session_id"],
                site_id=row["site_id"],
            )
        )
        if limit is not None and len(samples) == limit:
            break
    return samples


class GroundDataset(Dataset):
    def __init__(self, samples, data_root: Path, targets: dict[str, torch.Tensor]) -> None:
        self.rows = []
        self.labels = []
        for sample in samples:
            image_file = sample.context_path.relative_to(data_root).as_posix()
            if image_file in targets:
                self.rows.append(
                    (sample.record_uid, image_file, sample.context_path, targets[image_file])
                )
                self.labels.append(sample.label)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        uid, image_file, image_path, target = self.rows[index]
        return {
            "record_uid": uid,
            "image_file": image_file,
            "image_path": image_path,
            "target": target,
        }


class GroundCollator:
    def __init__(self, image_processor, input_config: dict) -> None:
        self.image_processor = image_processor
        self.size = {
            "longest_edge": input_config["max_pixels"],
            "shortest_edge": input_config["min_pixels"],
        }

    def __call__(self, rows: list[dict]) -> dict:
        images = []
        for row in rows:
            with Image.open(row["image_path"]) as source:
                images.append(source.convert("RGB"))

        inputs = self.image_processor(
            images=images,
            size=self.size,
            return_tensors="pt",
        )
        return {
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "targets": torch.stack([row["target"] for row in rows]),
            "record_uids": [row["record_uid"] for row in rows],
            "image_files": [row["image_file"] for row in rows],
        }


class QwenGroundMS(nn.Module):
    def __init__(self, vision_encoder: nn.Module, model_config: dict) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.trainable_vision_blocks = model_config["trainable_vision_blocks"]
        components = model_config["components"]
        self.pyramid_scales = model_config["pyramid_scales"]
        self.center_mode = components["center_mode"]
        self.size_uses_heatmap = components["size_uses_heatmap"]
        vision_dim = vision_encoder.config.out_hidden_size
        fusion_dim = model_config["hidden_dim"]
        self.feature_projection = nn.Linear(vision_dim, fusion_dim)
        self.position_embedding = nn.Sequential(
            nn.Linear(2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.level_embedding = nn.Embedding(len(self.pyramid_scales), fusion_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=model_config["num_heads"],
            dim_feedforward=fusion_dim * model_config["mlp_ratio"],
            dropout=model_config["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=model_config["decoder_layers"],
            norm=nn.LayerNorm(fusion_dim),
        )
        self.event_query = nn.Parameter(torch.randn(1, fusion_dim))
        self.query_center_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 2),
        )
        size_input_dim = fusion_dim * (2 if self.size_uses_heatmap else 1)
        self.size_head = nn.Sequential(
            nn.Linear(size_input_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 2),
        )
        self.heatmap_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        self.vision_encoder.requires_grad_(False)
        for block in self.vision_encoder.blocks[-self.trainable_vision_blocks :]:
            block.float().requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        for block in self.vision_encoder.blocks[-self.trainable_vision_blocks :]:
            block.train(mode)
        return self

    @staticmethod
    def spatial_coordinates(
        device: torch.device,
        height: int,
        width: int,
    ) -> torch.Tensor:
        y = (torch.arange(height, device=device) + 0.5) / height
        x = (torch.arange(width, device=device) + 0.5) / width
        y, x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((x.flatten(), y.flatten()), dim=-1)

    def grounding_queries(self) -> torch.Tensor:
        return self.event_query

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        vision_output = self.vision_encoder(
            pixel_values.to(self.vision_encoder.dtype),
            grid_thw=image_grid_thw,
            return_dict=True,
        ).pooler_output
        merge = self.vision_encoder.spatial_merge_size
        split_sizes = (image_grid_thw.prod(-1) // merge**2).tolist()
        image_features = torch.split(vision_output, split_sizes)

        memories, fine_sequences, fine_shapes = [], [], []
        for features, grid in zip(image_features, image_grid_thw.tolist()):
            time, source_height, source_width = grid
            source_height //= merge
            source_width //= merge
            spatial = features.view(time, source_height, source_width, -1).mean(0)
            projected = self.feature_projection(spatial.flatten(0, 1)).view(
                source_height, source_width, -1
            )
            levels = []
            for level, scale in enumerate(self.pyramid_scales):
                height = max(1, round(source_height / scale))
                width = max(1, round(source_width / scale))
                level_grid = F.adaptive_avg_pool2d(
                    projected.permute(2, 0, 1).unsqueeze(0), (height, width)
                ).squeeze(0).permute(1, 2, 0)
                coordinates = self.spatial_coordinates(features.device, height, width)
                level_tokens = (
                    level_grid.flatten(0, 1)
                    + self.position_embedding(coordinates)
                    + self.level_embedding.weight[level]
                )
                levels.append(level_tokens)
            fine_sequences.append(levels[0])
            fine_shapes.append((source_height, source_width))
            memories.append(torch.cat(levels))

        batch_size = len(memories)
        sequence_lengths = [len(memory) for memory in memories]
        maximum = max(sequence_lengths)
        memory = vision_output.new_zeros(
            batch_size,
            maximum,
            self.feature_projection.out_features,
            dtype=self.feature_projection.weight.dtype,
        )
        padding_mask = torch.ones(
            batch_size, maximum, dtype=torch.bool, device=vision_output.device
        )
        for batch_index, sequence in enumerate(memories):
            memory[batch_index, : len(sequence)] = sequence
            padding_mask[batch_index, : len(sequence)] = False

        query = self.grounding_queries().unsqueeze(0).expand(batch_size, -1, -1)
        query_features = self.event_decoder(
            query,
            memory,
            memory_key_padding_mask=padding_mask,
        )
        event_feature = query_features[:, 0]
        heatmap_logits, heatmap_features, heatmap_centers, fine_features = [], [], [], []
        scale = self.heatmap_logit_scale.exp().clamp(max=100)
        for fine, shape, event in zip(fine_sequences, fine_shapes, event_feature):
            height, width = shape
            logits = scale * (F.normalize(fine, dim=-1) @ F.normalize(event, dim=-1))
            probabilities = logits.softmax(-1)
            coordinates = self.spatial_coordinates(fine.device, height, width)
            heatmap_logits.append(logits.view(1, 1, height, width))
            heatmap_features.append(probabilities @ fine)
            heatmap_centers.append(probabilities.float() @ coordinates)
            fine_features.append(fine.transpose(0, 1).reshape(-1, height, width))

        heatmap_feature = torch.stack(heatmap_features)
        heatmap_center = torch.stack(heatmap_centers)
        center = (
            heatmap_center
            if self.center_mode == "softargmax"
            else self.query_center_head(event_feature).sigmoid()
        )
        size_input = (
            torch.cat((event_feature, heatmap_feature), dim=-1)
            if self.size_uses_heatmap
            else event_feature
        )
        size = self.size_head(size_input).sigmoid()
        bbox_cxcywh = torch.cat((center, size), dim=-1)
        return {
            "bbox_cxcywh": bbox_cxcywh,
            "heatmap_logits": heatmap_logits,
            "heatmap_center": heatmap_center,
            "event_feature": event_feature,
            "query_features": query_features,
            "heatmap_feature": heatmap_feature,
            "fine_features": fine_features,
            "memory": memory,
            "padding_mask": padding_mask,
            "fine_shapes": fine_shapes,
        }


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def generalized_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(first[:, :2], second[:, :2])
    bottom_right = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(-1)
    union = first_area + second_area - intersection
    iou = intersection / union.clamp(min=1e-7)
    enclosing_top_left = torch.minimum(first[:, :2], second[:, :2])
    enclosing_bottom_right = torch.maximum(first[:, 2:], second[:, 2:])
    enclosing = (enclosing_bottom_right - enclosing_top_left).clamp(min=0).prod(-1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-7)


def localization_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    l1_weight: float,
    giou_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1 = F.l1_loss(predictions, targets)
    giou = 1 - generalized_iou(cxcywh_to_xyxy(predictions), cxcywh_to_xyxy(targets)).mean()
    return l1_weight * l1 + giou_weight * giou, l1, giou


def gaussian_heatmap(
    targets: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    y = (torch.arange(height, device=targets.device, dtype=targets.dtype) + 0.5) / height
    x = (torch.arange(width, device=targets.device, dtype=targets.dtype) + 0.5) / width
    y, x = torch.meshgrid(y, x, indexing="ij")
    center_x = targets[:, 0, None, None]
    center_y = targets[:, 1, None, None]
    sigma_x = torch.maximum(targets[:, 2] / 4, targets.new_tensor(1 / width))
    sigma_y = torch.maximum(targets[:, 3] / 4, targets.new_tensor(1 / height))
    heatmap = torch.exp(
        -0.5
        * (
            ((x - center_x) / sigma_x[:, None, None]).square()
            + ((y - center_y) / sigma_y[:, None, None]).square()
        )
    )
    heatmap = heatmap / heatmap.sum((-2, -1), keepdim=True).clamp_min(1e-7)
    return heatmap.unsqueeze(1)


def spatial_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.flatten(1)
    return (
        target
        * (target.clamp_min(1e-8).log() - logits.float().flatten(1).log_softmax(-1))
    ).sum(-1).mean()


def per_box_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_xyxy = cxcywh_to_xyxy(predictions).clamp(0, 1)
    target_xyxy = cxcywh_to_xyxy(targets).clamp(0, 1)
    top_left = torch.maximum(predicted_xyxy[:, :2], target_xyxy[:, :2])
    bottom_right = torch.minimum(predicted_xyxy[:, 2:], target_xyxy[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    predicted_area = (predicted_xyxy[:, 2:] - predicted_xyxy[:, :2]).prod(-1)
    target_area = (target_xyxy[:, 2:] - target_xyxy[:, :2]).prod(-1)
    ious = intersection / (predicted_area + target_area - intersection).clamp(min=1e-7)
    center_errors = torch.linalg.vector_norm(
        predictions[:, :2] - targets[:, :2], dim=-1
    ) / math.sqrt(2)
    return ious, center_errors


def localization_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    ious, center_errors = per_box_metrics(predictions, targets)
    return {
        "median_iou": float(ious.quantile(0.5)),
        "recall_at_iou_0.25": float((ious >= 0.25).float().mean()),
        "recall_at_iou_0.50": float((ious >= 0.50).float().mean()),
        "recall_at_iou_0.75": float((ious >= 0.75).float().mean()),
        "normalized_center_error": float(center_errors.mean()),
    }


def cosine_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def checkpoint_payload(model: QwenGroundMS, epoch: int, metrics: dict, config: dict) -> dict:
    trainable_parameters = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("vision_encoder.") or key in trainable_parameters
    }
    return {
        "format_version": 6,
        "epoch": epoch,
        "metrics": metrics,
        "model_config": config["model"],
        "model_state_dict": state,
    }


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_ground_checkpoint(model: QwenGroundMS, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return checkpoint
