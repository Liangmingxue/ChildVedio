"""Soft-target construction and JEV-style multi-head distillation losses."""

from __future__ import annotations

import ast
import json
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


OUTCOME_TO_ID = {"harm": 0, "unchanged": 1, "fix": 2}


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return value


def _softmax_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def build_teacher_targets(
    rows: Sequence[dict],
    gt_index: np.ndarray,
    num_candidates: int = 16,
    temperature: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build path-wise VLM distributions and soft/hard light-to-deep outcomes.

    Teacher rank distributions are softmax-normalised over the candidates actually
    verified by each path. The route target is derived from the teacher probability
    mass assigned to the ground-truth candidate:
      p_fix  = max(p_deep(gt) - p_light(gt), 0)
      p_harm = max(p_light(gt) - p_deep(gt), 0)
      p_same = 1 - abs(p_deep(gt) - p_light(gt))

    This is a probability simplex by construction. The hard outcome label is kept
    as an auxiliary target from the cached VLM reranking ranks.
    """
    if temperature <= 0:
        raise ValueError("teacher temperature must be positive")
    if len(rows) != len(gt_index):
        raise ValueError("rows and gt_index must have the same length")

    teacher = np.zeros((len(rows), 2, num_candidates), dtype=np.float32)
    teacher_mask = np.zeros((len(rows), 2, num_candidates), dtype=bool)
    route_soft = np.zeros((len(rows), 3), dtype=np.float32)
    route_hard = np.zeros(len(rows), dtype=np.int64)

    for i, row in enumerate(rows):
        for path, key in enumerate(("light_scores", "deep_scores")):
            values = np.asarray(_parse(row[key]), dtype=np.float64).reshape(-1)
            if values.size == 0 or values.size > num_candidates:
                raise ValueError(f"{row.get('qid', i)} has invalid {key} length {values.size}")
            if not np.isfinite(values).all():
                raise ValueError(f"{row.get('qid', i)} has non-finite values in {key}")
            probs = _softmax_np(values / temperature)
            teacher[i, path, : values.size] = probs.astype(np.float32)
            teacher_mask[i, path, : values.size] = True

        gt = int(gt_index[i])
        p_light_gt = float(teacher[i, 0, gt]) if 0 <= gt < num_candidates and teacher_mask[i, 0, gt] else 0.0
        p_deep_gt = float(teacher[i, 1, gt]) if 0 <= gt < num_candidates and teacher_mask[i, 1, gt] else 0.0
        delta = float(np.clip(p_deep_gt - p_light_gt, -1.0, 1.0))
        route_soft[i] = (
            max(-delta, 0.0),
            1.0 - abs(delta),
            max(delta, 0.0),
        )

        ranks = _parse(row["ranks"])
        light_hit = int(ranks["light"]) == 1
        deep_hit = int(ranks["deep"]) == 1
        if light_hit and not deep_hit:
            route_hard[i] = OUTCOME_TO_ID["harm"]
        elif deep_hit and not light_hit:
            route_hard[i] = OUTCOME_TO_ID["fix"]
        else:
            route_hard[i] = OUTCOME_TO_ID["unchanged"]

    if not np.allclose(route_soft.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Route soft targets do not sum to one")
    return teacher, teacher_mask, route_soft, route_hard


def masked_rank_kl(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    teacher_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student), evaluated only on each path's verified candidates."""
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    if student_logits.shape != teacher_probs.shape or student_logits.shape != teacher_mask.shape:
        raise ValueError("student logits, teacher probabilities and mask must have identical shapes")
    if student_logits.ndim != 3:
        raise ValueError("rank tensors must have shape [B, paths, K]")

    logits = student_logits.float().reshape(-1, student_logits.shape[-1])
    target = teacher_probs.float().reshape_as(logits)
    mask = teacher_mask.bool().reshape_as(logits)
    valid = mask.any(dim=-1)
    if not valid.any():
        return student_logits.sum() * 0.0

    masked_logits = (logits / temperature).masked_fill(~mask, -1.0e4)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    per_row = F.kl_div(log_probs, target, reduction="none").sum(dim=-1)
    return per_row[valid].mean() * (temperature ** 2)


def listwise_hard_ce(student_logits: torch.Tensor, gt_index: torch.Tensor) -> torch.Tensor:
    """Hard listwise anchor for both light and deep ranking branches."""
    valid = gt_index >= 0
    if not valid.any():
        return student_logits.sum() * 0.0
    logits = student_logits[valid].float().reshape(-1, student_logits.shape[-1])
    labels = gt_index[valid].long().repeat_interleave(student_logits.shape[1])
    return F.cross_entropy(logits, labels)


def compute_loss(
    student_logits: torch.Tensor,
    outcome_logits: torch.Tensor,
    gt_index: torch.Tensor,
    teacher_probs: torch.Tensor,
    teacher_mask: torch.Tensor,
    route_soft: torch.Tensor,
    route_hard: torch.Tensor,
    config: dict,
    route_class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Composite listwise hard supervision + VLM KL + soft/hard outcome supervision."""
    rank_kl = masked_rank_kl(
        student_logits,
        teacher_probs,
        teacher_mask,
        float(config.get("rank_temperature", 1.0)),
    )
    rank_ce = listwise_hard_ce(student_logits, gt_index)

    route_target = route_soft.float()
    route_log_probs = F.log_softmax(outcome_logits.float(), dim=-1)
    route_kl = F.kl_div(route_log_probs, route_target, reduction="batchmean")
    route_ce = F.cross_entropy(
        outcome_logits.float(),
        route_hard.long(),
        weight=route_class_weights,
        label_smoothing=float(config.get("route_label_smoothing", 0.02)),
    )

    total = (
        float(config.get("rank_kd_weight", 1.0)) * rank_kl
        + float(config.get("rank_hard_weight", 0.25)) * rank_ce
        + float(config.get("route_kd_weight", 0.5)) * route_kl
        + float(config.get("route_hard_weight", 0.25)) * route_ce
    )
    return {
        "total": total,
        "rank_kl": rank_kl,
        "rank_ce": rank_ce,
        "route_kl": route_kl,
        "route_ce": route_ce,
    }
