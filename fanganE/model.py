"""Candidate-conditioned JEV-style ranking and intervention-outcome heads."""

from __future__ import annotations

import torch
from torch import nn

from fanganC.vendor.pointer import PointerHead


OUTCOME_NAMES = ("harm", "unchanged", "fix")


class JevCandidateRanker(nn.Module):
    """Two listwise rankers plus a three-way expected-intervention-value head.

    The text representation is available before deep verification. The two ranking
    branches consume different cached evidence:
      branch 0: coarse + light evidence;
      branch 1: coarse + light + deep evidence.

    The outcome head sees only the text/evidence hidden state. It therefore cannot
    inspect deep scores when deciding whether a query should be upgraded.
    """

    def __init__(
        self,
        hidden_dim: int,
        numeric_dim: int,
        clip_dim: int,
        pointer_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min(hidden_dim, numeric_dim, clip_dim, pointer_dim) <= 0:
            raise ValueError("All feature dimensions must be positive")

        self.context = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
        )
        self.offset = nn.Linear(32, 2 * numeric_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

        self.query = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, clip_dim),
        )
        self.pointer = PointerHead(
            clip_dim,
            pointer_dim,
            similarity="cosine",
            logit_scale=10.0,
        )
        self.semantic_scale = nn.Parameter(torch.zeros(2))

        self.outcome = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, len(OUTCOME_NAMES)),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        numeric: torch.Tensor,
        initial_scores: torch.Tensor,
        candidate_videos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [B, H]")
        if numeric.ndim != 4 or numeric.shape[1] != 2:
            raise ValueError("numeric must have shape [B, 2, K, D]")
        if initial_scores.shape != numeric.shape[:3]:
            raise ValueError("initial_scores must have shape [B, 2, K]")
        if candidate_videos.ndim != 3 or candidate_videos.shape[:2] != (hidden.shape[0], numeric.shape[2]):
            raise ValueError("candidate_videos must have shape [B, K, Dv]")

        context = self.context(hidden.float())
        offsets = self.offset(context).reshape(hidden.shape[0], 2, numeric.shape[-1])
        rank_logits = initial_scores.float() + (numeric.float() * offsets[:, :, None, :]).sum(-1)

        query = self.query(hidden.float())
        semantic_logits = self.pointer(candidate_videos.float(), query)
        rank_logits = rank_logits + self.semantic_scale[None, :, None] * semantic_logits[:, None, :]

        outcome_logits = self.outcome(hidden.float())
        return rank_logits, outcome_logits
