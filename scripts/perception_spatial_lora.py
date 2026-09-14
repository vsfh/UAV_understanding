"""Zero-init spatial low-rank adapters on Qwen3-VL's merged visual streams.

This is a nonlinear spatial adapter alongside PEFT LoRA, not a mergeable BA
weight update. The full Qwen language model and its native DeepStack stay intact.
"""
from copy import copy

import torch
from torch import nn
import torch.nn.functional as F


class SpatialLowRankResidual(nn.Module):
    def __init__(self, hidden_dim, rank=32, alpha=32, dropout=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.position = nn.Linear(2, rank, bias=False)
        self.local = nn.Conv2d(rank, rank, 3, padding=1, groups=rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        # Only the up projection is zero: its gradient is live on step one.
        nn.init.zeros_(self.up.weight)

    def forward(self, features, grids):
        """features: concatenated merged tokens; grids: [(t, merged_h, merged_w)]."""
        low = self.down(self.norm(features))
        pieces = low.split([t * h * w for t, h, w in grids])
        residuals = []
        for tokens, (t, h, w) in zip(pieces, grids):
            y = (torch.arange(h, device=features.device, dtype=low.dtype) + 0.5) * (2 / h) - 1
            x = (torch.arange(w, device=features.device, dtype=low.dtype) + 0.5) * (2 / w) - 1
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            xy = torch.stack((xx, yy), dim=-1)
            z = F.silu(tokens.reshape(t, h, w, -1) + self.position(xy))
            # Process each image/frame independently; never convolve across images.
            local = self.local(z.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            residuals.append((z + local).reshape(t * h * w, -1))
        delta = self.up(self.dropout(torch.cat(residuals))) * self.scale
        return features + delta.to(features.dtype)


class VisualSpatialAdapter(nn.Module):
    def __init__(self, hidden_dim, num_deepstack, merge_size, rank=32, alpha=32, dropout=0.05):
        super().__init__()
        self.merge_size = merge_size
        self.branches = nn.ModuleList([
            SpatialLowRankResidual(hidden_dim, rank, alpha, dropout)
            for _ in range(1 + num_deepstack)
        ])

    def inject(self, module, args, kwargs, output):
        """Visual forward hook shared by SFT, autoregressive generation and ROI replay.

        Qwen3-VL orders merged tokens as [t, h/merge, w/merge], x fastest.
        Get grids from this invocation, not mutable state used by another batch.
        """
        grid_thw = kwargs["grid_thw"] if "grid_thw" in kwargs else args[1]
        grids = [(t, h // self.merge_size, w // self.merge_size)
                 for t, h, w in grid_thw.tolist()]
        streams = [output.pooler_output, *output.deepstack_features]
        if len(streams) != len(self.branches):
            raise ValueError("Spatial adapter branches disagree with Qwen DeepStack streams")
        adapted = [branch(features, grids) for branch, features in zip(self.branches, streams)]
        result = copy(output)
        result["pooler_output"] = adapted[0]
        result["deepstack_features"] = adapted[1:]
        return result
