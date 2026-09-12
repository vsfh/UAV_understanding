"""Bounded discrete box refinement; tensor-only helpers also work on CPU.

This is an action-policy adaptation, not a reproduction of GETok. The 9
categorical actions have explicit likelihoods. No ground truth is read by
rollout() or by the inference policy.
"""
import torch
from torch import nn
import torch.nn.functional as F

ACTION_NAMES = ("stop", "left", "right", "up", "down", "narrow", "widen", "shorten", "heighten")


def valid_boxes(boxes, minimum=1e-4):
    lo = torch.minimum(boxes[..., :2], boxes[..., 2:]).clamp(0, 1 - minimum)
    hi = torch.maximum(boxes[..., :2], boxes[..., 2:]).clamp(minimum, 1)
    return torch.cat((lo, torch.maximum(hi, lo + minimum)), -1)


def aligned_iou(boxes, target):
    intersection = (torch.minimum(boxes[..., 2:], target[..., 2:]) -
                    torch.maximum(boxes[..., :2], target[..., :2])).clamp_min(0).prod(-1)
    area = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp_min(0).prod(-1)
    return intersection / (area + target_area - intersection).clamp_min(1e-8)


def apply_action(boxes, actions, step, settings):
    """Shift by a box-relative distance or scale one extent; stop is identity."""
    if ((actions < 0) | (actions >= len(ACTION_NAMES))).any():
        raise ValueError("Action must be in [0, 8]")
    boxes = valid_boxes(boxes)
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(settings.get("minimum_size", 1e-4))
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    decay = settings["decay"] ** step
    shift = settings["center_step"] * decay
    scale = settings["log_scale_step"] * decay
    dx = (actions == 2).float() - (actions == 1).float()
    dy = (actions == 4).float() - (actions == 3).float()
    sx = (actions == 6).float() - (actions == 5).float()
    sy = (actions == 8).float() - (actions == 7).float()
    center = center + torch.stack((dx, dy), -1) * sizes * shift
    sizes = (sizes * torch.exp(torch.stack((sx, sy), -1) * scale)).clamp(
        settings.get("minimum_size", 1e-4), 1)
    center = torch.maximum(torch.minimum(center, 1 - sizes / 2), sizes / 2)
    updated = valid_boxes(torch.cat((center - sizes / 2, center + sizes / 2), -1))
    return torch.where((actions == 0)[..., None], boxes, updated)


class ActionPolicy(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden_dim + 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 9))

    def forward(self, features, boxes, step, max_steps):
        progress = torch.full_like(boxes[..., :1], float(step) / max(1, max_steps - 1))
        return self.head(torch.cat((self.feature(features.float()), boxes.float(), progress), -1))


def rollout(policy, features, boxes, settings, greedy=False, enabled=None):
    """Return a finite trajectory including stop likelihood, masking after stop."""
    boxes = valid_boxes(boxes.detach())
    active = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device) if enabled is None else enabled.clone()
    states, actions, log_probs, masks = [], [], [], []
    for step in range(settings["max_steps"]):
        logits = policy(features, boxes, step, settings["max_steps"])
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        action = torch.where(active, action, torch.zeros_like(action))
        states.append(boxes)
        actions.append(action)
        log_probs.append(torch.where(active, dist.log_prob(action), torch.zeros_like(logits[..., 0])))
        masks.append(active)
        boxes = torch.where(active[:, None], apply_action(boxes, action, step, settings), boxes)
        active = active & (action != 0)
    return {"states": torch.stack(states, 1), "actions": torch.stack(actions, 1),
            "log_probs": torch.stack(log_probs, 1), "mask": torch.stack(masks, 1), "boxes": boxes}


def trajectory_logits(policy, features, trajectory, settings):
    return torch.stack([policy(features, trajectory["states"][:, t], t, settings["max_steps"])
                        for t in range(settings["max_steps"])], 1)


def imitation_loss(policy, features, initial_boxes, targets, positive, settings):
    """Greedy GT oracle builds train-only action labels; stopped rows stay terminal."""
    boxes = valid_boxes(initial_boxes.detach())
    active = positive.clone()
    total, count = features.sum() * 0, torch.zeros((), device=features.device)
    for step in range(settings["max_steps"]):
        with torch.no_grad():
            candidates = torch.stack([apply_action(boxes, torch.full_like(positive, a, dtype=torch.long), step, settings)
                                      for a in range(9)], 1)
            quality = aligned_iou(candidates, targets[:, None, :])
            # A small preference for STOP prevents pointless zero-gain paths.
            quality[:, 0] += settings.get("oracle_stop_margin", 1e-4)
            oracle = quality.argmax(-1)
        logits = policy(features, boxes, step, settings["max_steps"])
        total = total + (F.cross_entropy(logits, oracle, reduction="none") * active).sum()
        count = count + active.sum()
        boxes = torch.where(active[:, None], candidates[torch.arange(len(boxes), device=boxes.device), oracle], boxes)
        active = active & (oracle != 0)
    return total / count.clamp_min(1)


def trajectory_reward(initial, final, targets, eligible, trajectory, settings):
    initial_iou, final_iou = aligned_iou(initial, targets), aligned_iou(final, targets)
    moves = ((trajectory["actions"] != 0) & trajectory["mask"]).sum(-1).float()
    reward = (settings["final_iou"] * final_iou + settings["iou_gain"] * (final_iou - initial_iou)
              - settings["step_cost"] * moves)
    # This policy cannot change the category. Invalid/no-event and misclassified
    # examples are excluded, not spuriously credited as classification RL.
    return torch.where(eligible, reward, torch.zeros_like(reward))


def group_advantages(rewards, group_size, epsilon=1e-4):
    grouped = rewards.reshape(-1, group_size)
    advantages = (grouped - grouped.mean(-1, keepdim=True)) / (grouped.std(-1, unbiased=False, keepdim=True) + epsilon)
    return advantages.reshape(-1)


def grpo_loss(policy, reference, features, trajectory, advantages, settings, grpo):
    logits = trajectory_logits(policy, features, trajectory, settings)
    log_probs = F.log_softmax(logits, -1)
    selected = log_probs.gather(-1, trajectory["actions"][..., None]).squeeze(-1)
    ratio = (selected - trajectory["log_probs"].detach()).exp()
    unclipped = ratio * advantages.detach()[:, None]
    clipped = ratio.clamp(1 - grpo["clip"], 1 + grpo["clip"]) * advantages.detach()[:, None]
    with torch.no_grad():
        reference_log_probs = F.log_softmax(trajectory_logits(reference, features, trajectory, settings), -1)
    # Exact categorical KL at the visited states, including terminal STOP.
    kl = (log_probs.exp() * (log_probs - reference_log_probs)).sum(-1)
    mask = trajectory["mask"].float()
    denom = mask.sum().clamp_min(1)
    loss = ((-torch.minimum(unclipped, clipped) + grpo["beta"] * kl) * mask).sum() / denom
    clipped_fraction = (((unclipped > clipped).detach().float()) * mask).sum() / denom
    return loss, {"kl": float((kl.detach() * mask).sum() / denom),
                  "ratio": float((ratio.detach() * mask).sum() / denom),
                  "clip_fraction": float(clipped_fraction)}


def update_rollout_buffer(policy, reference, buffer, settings, grpo, optimizer, scheduler=None, max_updates=None):
    """Reuse one frozen behavior buffer for several optimizer updates.

    Each buffer entry is one microbatch with immutable states, actions, rewards,
    advantages and old log-probabilities. Gradients accumulate over the actual
    number of microbatches, including a short final buffer. Only current policy
    probabilities are recomputed. The caller counts these optimizer updates,
    not the number of rollout buffers, against its global step budget.
    """
    if not buffer:
        raise ValueError("A rollout buffer must contain at least one microbatch")
    repetitions = grpo.get("updates_per_rollout", 2)
    if max_updates is not None:
        repetitions = min(repetitions, max_updates)
    history = []
    for repeat in range(repetitions):
        optimizer.zero_grad(set_to_none=True)
        values = {"loss": 0.0, "kl": 0.0, "ratio": 0.0, "clip_fraction": 0.0, "anchor": 0.0, "reward": 0.0}
        for item in buffer:
            loss, parts = grpo_loss(policy, reference, item["grouped_features"], item["trajectory"],
                                    item["advantages"], settings, grpo)
            anchor = imitation_loss(policy, item["features"], item["initial"], item["target"], item["eligible"], settings)
            loss = loss + grpo["imitation_anchor"] * anchor
            (loss / len(buffer)).backward()
            parts.update(loss=float(loss.detach()), anchor=float(anchor.detach()), reward=float(item["rewards"].mean()))
            for key in values:
                values[key] += parts[key] / len(buffer)
        nn.utils.clip_grad_norm_(policy.parameters(), grpo["max_grad_norm"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        history.append({"rollout_update": repeat + 1, "rollout_microbatches": len(buffer), **values})
    optimizer.zero_grad(set_to_none=True)
    return history
