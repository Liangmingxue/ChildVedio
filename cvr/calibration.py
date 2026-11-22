"""Temperature scaling, calibration metrics and conformal risk control (CRC)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .utils import log_softmax_np, softmax_np


def nll_score(logits, labels, T: float = 1.0) -> float:
    lp = log_softmax_np(np.asarray(logits, dtype=np.float64) / T)
    labels = np.asarray(labels, dtype=np.int64)
    return float(-lp[np.arange(len(labels)), labels].mean())


def fit_temperature(logits, labels) -> Tuple[float, float]:
    """Grid search over log-temperature minimising NLL (robust, no optimiser needed)."""
    best_T, best = 1.0, nll_score(logits, labels)
    for T in np.exp(np.linspace(np.log(0.05), np.log(20.0), 400)):
        v = nll_score(logits, labels, T)
        if v < best:
            best_T, best = float(T), v
    return best_T, best


def apply_temperature(logits, T: float) -> np.ndarray:
    return softmax_np(np.asarray(logits, dtype=np.float64) / T)


def ece_score(probs, labels, n_bins: int = 15) -> Tuple[float, List[Dict[str, float]]]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        c, a, w = conf[mask].mean(), correct[mask].mean(), mask.mean()
        ece += w * abs(c - a)
        bins.append({"lo": float(lo), "hi": float(hi), "conf": float(c), "acc": float(a), "count": int(mask.sum())})
    return float(ece), bins


def brier_score(probs, labels) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.eye(probs.shape[1])[np.asarray(labels, dtype=np.int64)]
    return float(((probs - onehot) ** 2).sum(1).mean())


def crc_threshold(p_accept, loss_if_accept, alpha: float) -> float:
    """Smallest tau s.t. the CRC bound (sum_i L_i(tau) + 1) / (n + 1) <= alpha holds.

    L_i(tau) = loss_if_accept_i * 1[p_accept_i >= tau] is non-increasing in tau and bounded
    by B = 1, so accepting the first-stage ranking whenever p_accept >= tau controls the
    expected loss at level alpha on exchangeable test queries (Angelopoulos et al., CRC).
    """
    p = np.asarray(p_accept, dtype=np.float64)
    loss = np.asarray(loss_if_accept, dtype=np.float64)
    n = len(p)
    candidates = np.unique(np.concatenate([p, [1.0 + 1e-9]]))  # ascending
    for tau in candidates:
        risk = float((loss * (p >= tau)).sum())
        if (risk + 1.0) / (n + 1.0) <= alpha:
            return float(tau)
    return float(candidates[-1])
