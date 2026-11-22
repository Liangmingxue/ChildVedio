"""Retrieval, flip (intervention) and cost metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Sequence

import numpy as np

from .utils import tag_query


def retrieval_metrics(ranks: Sequence[int]) -> Dict[str, float]:
    r = np.asarray(ranks, dtype=np.float64)
    if r.size == 0:
        return {"n": 0}
    return {
        "n": int(r.size),
        "R@1": float(100 * (r <= 1).mean()),
        "R@5": float(100 * (r <= 5).mean()),
        "R@10": float(100 * (r <= 10).mean()),
        "MdR": float(np.median(r)),
        "MnR": float(r.mean()),
    }


def flip_metrics(base_ranks: Sequence[int], new_ranks: Sequence[int]) -> Dict[str, float]:
    """HFR = harmful flip rate: share of originally-correct top-1 results the agent broke."""
    b = np.asarray(base_ranks) == 1
    n = np.asarray(new_ranks) == 1
    harmful = int((b & ~n).sum())
    helpful = int((~b & n).sum())
    return {
        "harmful_flips": harmful,
        "helpful_flips": helpful,
        "HFR": float(100 * harmful / max(int(b.sum()), 1)),
        "HelpR": float(100 * helpful / max(int((~b).sum()), 1)),
        "net_R@1": float(100 * (n.mean() - b.mean())) if b.size else 0.0,
    }


def summarize_ranks(base_ranks: Sequence[int], new_ranks: Sequence[int]) -> Dict[str, dict]:
    return {
        "base": retrieval_metrics(base_ranks),
        "new": retrieval_metrics(new_ranks),
        "flips": flip_metrics(base_ranks, new_ranks),
    }


def breakdown_by_tag(texts: Sequence[str], base_ranks: Sequence[int], new_ranks: Sequence[int]) -> Dict[str, dict]:
    base = np.asarray(base_ranks)
    new = np.asarray(new_ranks)
    groups = defaultdict(list)
    for i, t in enumerate(texts):
        for tag in tag_query(t):
            groups[tag].append(i)
    out = {}
    for tag, idx in sorted(groups.items()):
        idx = np.asarray(idx)
        out[tag] = {
            "n": int(idx.size),
            "base_R@1": float(100 * (base[idx] == 1).mean()),
            "new_R@1": float(100 * (new[idx] == 1).mean()),
            **flip_metrics(base[idx], new[idx]),
        }
    return out
