"""Re-ranking helpers shared by the light (yes/no) and deep (counterfactual) paths."""
from __future__ import annotations

import asyncio
from typing import List, Sequence

import numpy as np


def _minmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def rerank_topk(cands: Sequence[str], base_scores: Sequence[float], vlm_scores: Sequence[float],
                beta: float) -> List[str]:
    """Re-order the first len(vlm_scores) candidates by beta*vlm + (1-beta)*minmax(base)."""
    k = len(vlm_scores)
    if k == 0:
        return list(cands)
    comb = beta * np.asarray(vlm_scores, dtype=np.float64) + (1.0 - beta) * _minmax(base_scores[:k])
    order = np.argsort(-comb, kind="stable")
    return [cands[i] for i in order] + list(cands[k:])


def new_gt_rank(row: dict, new_cands: Sequence[str]) -> int:
    gt = row["gt"]
    if gt in new_cands:
        return list(new_cands).index(gt) + 1
    return int(row["gt_rank"])


async def light_scores(verifier, row: dict, k: int) -> List[float]:
    return list(await asyncio.gather(*[verifier.yes_prob(row["text"], v) for v in row["cands"][:k]]))


async def deep_scores(verifier, row: dict, hyps: Sequence[str], k: int, n_perm: int, seed: int = 0) -> List[float]:
    return list(await asyncio.gather(*[
        verifier.forced_choice(row["text"], hyps, v, n_perm=n_perm, seed=seed) for v in row["cands"][:k]
    ]))
