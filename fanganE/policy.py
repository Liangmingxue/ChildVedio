"""Budget-matched light-to-deep allocation and rank reconstruction."""

from __future__ import annotations

import numpy as np

from fanganD.lib import BUDGET, fill_budget


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(logits, dtype=np.float64) / temperature
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def allocate_budget(
    priority: np.ndarray,
    costs: np.ndarray,
    gallery: np.ndarray,
    budget: int = BUDGET,
) -> np.ndarray:
    """Select deep upgrades independently inside each 1,000-query gallery.

    Priority is expected utility P(fix) - P(harm); larger is better.
    Every query first pays its light cost. Deep costs are incremental.
    """
    priority = np.asarray(priority, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.int64)
    gallery = np.asarray(gallery, dtype=np.int64)
    if priority.shape != gallery.shape or costs.shape != (len(gallery), 3):
        raise ValueError("priority, gallery and costs have incompatible shapes")

    chosen = np.zeros(len(gallery), dtype=bool)
    for group in np.unique(gallery):
        ids = np.flatnonzero(gallery == group)
        group_budget = int(round(budget * len(ids) / 1000))
        light_calls = int(costs[ids, 1].sum())
        if light_calls > group_budget:
            raise ValueError(
                f"Gallery {group} cannot afford light for every query: "
                f"{light_calls} > {group_budget}"
            )
        selected, _ = fill_budget(
            -priority[ids],
            costs[ids, 2],
            light_calls,
            group_budget,
        )
        chosen[ids] = selected
    return chosen


def rank_from_path(
    path_scores: np.ndarray,
    use_deep: np.ndarray,
    gt_index: np.ndarray,
    original_rank: np.ndarray,
) -> np.ndarray:
    """Reconstruct final ranks; preserve original rank when GT is outside Top-K."""
    scores = np.asarray(path_scores)
    use_deep = np.asarray(use_deep, dtype=bool)
    gt_index = np.asarray(gt_index, dtype=np.int64)
    original_rank = np.asarray(original_rank, dtype=np.int64)
    if scores.ndim != 3 or scores.shape[1] != 2:
        raise ValueError("path_scores must have shape [N, 2, K]")
    if len(use_deep) != len(scores) or len(gt_index) != len(scores) or len(original_rank) != len(scores):
        raise ValueError("rank inputs have inconsistent lengths")

    selected = scores[np.arange(len(scores)), use_deep.astype(np.int64)]
    ranks = original_rank.copy()
    valid = (gt_index >= 0) & (gt_index < scores.shape[-1])
    if valid.any():
        order = np.argsort(-selected[valid], axis=1, kind="stable")
        ranks[valid] = (order == gt_index[valid, None]).argmax(axis=1) + 1
    return ranks
