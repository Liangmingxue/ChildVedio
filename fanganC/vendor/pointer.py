"""Extracted from VideoJev timejev/model.py PointerHead (Apache-2.0).

Source: /media/omnisky/11D1C4300244251B1/xlm/VideoJev/timejev/model.py
Adaptation: standalone module; removed unrelated multimodal backbone imports.
The head's computation and initialization are unchanged.
"""
import math
import torch
from torch import nn


class PointerHead(nn.Module):
    def __init__(self, hidden, dimension=256, head_norm="none", *,
                 similarity="dot_product", logit_scale=10.0):
        super().__init__()
        if head_norm not in ("none", "layernorm"):
            raise ValueError("Unknown pointer head normalization")
        if similarity not in ("dot_product", "cosine"):
            raise ValueError("Unknown pointer head similarity")
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError("Pointer head logit scale must be finite and positive")
        self.similarity = similarity
        self.head_norm = head_norm
        self.q = nn.Linear(hidden, dimension)
        self.k = nn.Linear(hidden, dimension)
        self.bias = nn.Parameter(torch.zeros(()))
        self.dimension = dimension
        self.scale = dimension ** -.5 if similarity == "dot_product" else float(logit_scale)
        nn.init.normal_(self.q.weight, std=.005)
        nn.init.normal_(self.k.weight, std=.005)
        nn.init.zeros_(self.q.bias)
        nn.init.zeros_(self.k.bias)
        if head_norm == "layernorm":
            self.q_norm = nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=True)
            self.k_norm = nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=True)

    def forward(self, pairs, query):
        if self.head_norm == "layernorm":
            keys = self.k(self.k_norm(pairs.float()))
            q = self.q(self.q_norm(query.float()))
        else:
            keys = self.k(pairs.float())
            q = self.q(query.float())
        if self.similarity == "cosine":
            keys = nn.functional.normalize(keys, dim=-1, eps=1e-6)
            q = nn.functional.normalize(q, dim=-1, eps=1e-6)
        return torch.einsum("bgd,bd->bg", keys, q) * self.scale + self.bias
