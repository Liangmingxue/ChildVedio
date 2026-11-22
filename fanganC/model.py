"""Candidate evidence plus the existing B probability parameterization."""
import numpy as np
import torch
from torch import nn

from .common import legacy
from .vendor.pointer import PointerHead


class CandidateRouter(nn.Module):
    def __init__(self, text_dim, clip_dim, numeric_train, margins, coarse_correct,
                 variant="candidate", hidden_dim=128, pointer_dim=128, dropout=.1, *, k_light=5, k_deep=None,
                 temporal_candidates=False):
        super().__init__()
        if variant not in ("candidate", "text_only"):
            raise ValueError("Unknown routing variant")
        if k_light < 1 or (k_deep is not None and k_deep < 1):
            raise ValueError("Light and deep verification each require at least one candidate")
        self.variant = variant
        self.k_light = k_light
        self.k_deep = k_deep
        numeric = torch.as_tensor(numeric_train, dtype=torch.float32)
        self.register_buffer("numeric_mean", numeric.mean(0))
        self.register_buffer("numeric_scale", numeric.std(0, unbiased=False).clamp_min(1e-6))
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.numeric = nn.Sequential(nn.Linear(numeric.shape[1], 32), nn.GELU())
        self.pointer = PointerHead(clip_dim, pointer_dim, similarity="cosine", logit_scale=10.)
        self.temporal = None
        evidence_dim = clip_dim * 4
        if temporal_candidates and variant == "candidate":
            self.temporal = nn.GRU(clip_dim + 1, pointer_dim, batch_first=True)
            evidence_dim += pointer_dim * 3
        self.evidence = nn.Sequential(nn.Linear(evidence_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2 + 32, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.outcome = legacy().FactorizedGainHead(hidden_dim, np.asarray(margins))
        self.outcome.fit_margin_prior(margins, coarse_correct)

    def temporal_context(self, query, frames, frame_mask, mask):
        valid = frame_mask & mask[..., None]
        frames = torch.where(valid[..., None], frames.float(), 0.)
        relevance = torch.einsum("bd,bktd->bkt", query, frames)
        ordered, _ = self.temporal(torch.cat((frames, relevance[..., None]), dim=-1).flatten(0, 1))
        last = (valid.sum(-1).flatten() - 1).clamp_min(0)
        outputs = ordered[torch.arange(len(last), device=last.device), last].reshape(*mask.shape, -1)
        return torch.where(mask[..., None], outputs, 0.)

    def forward(self, states, batch):
        text = self.text(states.float())
        numeric = self.numeric((batch["numeric"] - self.numeric_mean) / self.numeric_scale)
        if self.variant == "text_only":
            evidence = torch.zeros_like(text)
        else:
            query, videos, mask = batch["query"], batch["videos"], batch["mask"]
            if not mask.any(1).all() or not mask[:, 0].all():
                raise ValueError("Every query requires a valid first candidate")
            # Explicitly eliminate padding before projections, including NaN padding.
            videos = torch.where(mask[..., None], videos, torch.zeros_like(videos))
            scores = self.pointer(videos, query)
            positions = torch.arange(mask.shape[1], device=mask.device)[None]
            deep_mask = mask if self.k_deep is None else mask & (positions < self.k_deep)
            light_mask = mask & (positions < self.k_light)
            full = scores.masked_fill(~deep_mask, -torch.inf).softmax(-1)
            light = scores.masked_fill(~light_mask, -torch.inf).softmax(-1)
            full_context = torch.einsum("bk,bkd->bd", full, videos)
            light_context = torch.einsum("bk,bkd->bd", light, videos)
            first = videos[:, 0]
            comparisons = [query * first, (query - first).abs(), first - full_context, light_context - full_context]
            if self.temporal is not None:
                ordered = self.temporal_context(query, batch["frames"], batch["frame_mask"], mask)
                full_ordered = torch.einsum("bk,bkd->bd", full, ordered)
                light_ordered = torch.einsum("bk,bkd->bd", light, ordered)
                comparisons.extend((ordered[:, 0], ordered[:, 0] - full_ordered, light_ordered - full_ordered))
            evidence = self.evidence(torch.cat(comparisons, dim=-1))
        state = self.fusion(torch.cat((text, numeric, evidence), -1))
        return self.outcome(state, batch["margin"])

    def loss(self, logits, correct):
        return self.outcome.loss(logits, correct)

    def probabilities(self, logits):
        return self.outcome.probabilities(logits)
