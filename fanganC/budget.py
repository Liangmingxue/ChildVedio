"""Loss-augmented training of the same exact-budget allocator used at inference."""
from __future__ import annotations

import numpy as np
import torch

from .common import legacy


def structured_budget_loss(probabilities, correct, calls, deep_fraction, harm_weights):
    """Penalize the realized utility regret of assigning a training batch's budget."""
    costs, counts = np.unique(calls[:, 2], return_counts=True)
    budget = sum(int(cost) * round(int(count) * deep_fraction) for cost, count in zip(costs, counts, strict=True))
    coarse = correct[:, :1]
    fixes = (~coarse & correct[:, 1:]).astype(np.float32)
    harms = (coarse & ~correct[:, 1:]).astype(np.float32)
    zero = probabilities.new_zeros((len(calls), 1))
    rows = torch.arange(len(calls), device=probabilities.device)
    losses = []
    for weight in harm_weights:
        predicted = torch.cat((zero, probabilities[:, :, 1] - weight * probabilities[:, :, 2]), dim=1)
        actual = np.column_stack((np.zeros(len(calls)), fixes - weight * harms))
        oracle = legacy().allocate(actual, calls, budget)
        # Add the true utility gap to the predicted score. The oracle term of
        # that gap is constant, so inference uses predicted minus actual utility.
        rival = legacy().allocate(predicted.detach().cpu().numpy() - actual, calls, budget)
        regret = (actual[np.arange(len(calls)), oracle] - actual[np.arange(len(calls)), rival]).mean()
        best = torch.as_tensor(oracle, dtype=torch.long, device=rows.device)
        alternative = torch.as_tensor(rival, dtype=torch.long, device=rows.device)
        losses.append(((predicted[rows, alternative] - predicted[rows, best]).mean() + regret).clamp_min(0.))
    return torch.stack(losses).mean()
