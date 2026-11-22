"""Build clients from the config."""
from __future__ import annotations

from .counterfactual import NLIFilter, TextLLMClient
from .vlm import VLMVerifier


def make_verifier(cfg) -> VLMVerifier:
    v = cfg.vlm
    return VLMVerifier(
        base_url=v.base_url,
        model=v.model,
        frames_dir=cfg.paths.frames_dir,
        n_frames=cfg.frames.n_frames,
        max_side=cfg.frames.max_side,
        concurrency=v.concurrency,
        option_temp=v.option_temp,
        yes_no_token_ids=v.get("yes_no_token_ids"),
        option_token_ids=v.get("option_token_ids"),
    )


def make_llm(cfg, section: str = "llm") -> TextLLMClient:
    s = cfg[section]
    return TextLLMClient(
        base_url=s.base_url,
        model=s.model,
        concurrency=s.concurrency,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
    )


def make_nli(cfg, device: str | None = None) -> NLIFilter:
    c = cfg.cf
    return NLIFilter(c.nli_model, device=device or c.nli_device, thr=c.nli_thr)
