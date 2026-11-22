"""Real CUDA retrieval compatibility, warmup, LoRA and reload smoke check."""
import gc
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import atomic_json, legacy, module_from_file, read_jsonl, resolve
from .features import encode, numeric_features
from .model import CandidateRouter
from .training import optimizer_groups
from cvr.router import JevFormatter, field_states, load_router
from cvr.utils import load_config, set_seed


def smoke(run_dir, device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requires an available CUDA runtime")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    cfg = load_config(str(resolve("outputs/stage2_clip4clip_qwen3vl8b/config.yaml")))
    source = read_jsonl(resolve(cfg.paths.work_dir) / "router_data/router_data.jsonl")
    rows = [next(r for r in source if r["split"] == "train" and r["ranks"]["accept"] == 1),
            next(r for r in source if r["split"] == "train" and r["ranks"]["accept"] != 1)]
    started = time.perf_counter()
    upstream = module_from_file("temp/stage0_clip4clip.py")
    clip = upstream.load_model(device)
    tokenizer = upstream.SimpleTokenizer()
    query_tokens = torch.zeros(2, upstream.MAX_WORDS, dtype=torch.long)
    for i, row in enumerate(rows):
        ids = [tokenizer.encoder["<|startoftext|>"]] + tokenizer.encode(row["text"])[:30] + [tokenizer.encoder["<|endoftext|>"]]
        query_tokens[i, :len(ids)] = torch.tensor(ids)
    videos = {r["video_id"]: r for r in read_jsonl(resolve(cfg.paths.data_dir) / "videos.jsonl")}
    # Four real candidates are enough to verify decoding and original score units.
    chosen = [row["cands"][:2] for row in rows]
    gallery = [videos[v] for group in chosen for v in group]
    with torch.inference_mode():
        query = F.normalize(clip.clip.encode_text(query_tokens.to(device)).float(), dim=-1)
        frames, mask = next(iter(DataLoader(upstream.Gallery(gallery), batch_size=4, num_workers=0)))
        mask = mask.to(device)
        visual = clip.get_visual_output(frames.to(device).flatten(0, 1), mask, shaped=True, video_frame=12).float()
        visual = F.normalize(visual, dim=-1)
        video = F.normalize((visual * mask[..., None]).sum(1) / mask.sum(1, keepdim=True), dim=-1).reshape(2, 2, -1)
        reconstructed = clip.clip.logit_scale.exp() * torch.einsum("bd,bkd->bk", query, video)
        expected = torch.tensor([r["scores"][:2] for r in rows], device=device)
        error = float((reconstructed - expected).abs().max())
        if error > .02:
            raise ValueError(f"Retriever smoke score mismatch: {error}")
    query, video = query.clone(), video.clone()
    del clip, visual, frames
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer, backbone = load_router(str(resolve(cfg.router.base_model)), device=device, train=True,
                                      lora_r=cfg.router.lora_r, lora_alpha=cfg.router.lora_alpha)
    backbone.config.use_cache = False
    formatter = JevFormatter(tokenizer, max_len=cfg.router.max_len)
    encoded = encode(rows, cfg, formatter)
    flags = {name: p.requires_grad for name, p in backbone.named_parameters()}
    backbone.requires_grad_(False).eval()
    with torch.no_grad():
        _, cached = field_states(backbone, formatter, encoded, device)
        cached = cached[:, 0].detach()
    numeric = numeric_features(rows, cfg)
    margins = numeric[:, 5]
    correct = torch.tensor([[r["ranks"][a] == 1 for a in legacy().ACTIONS] for r in rows], device=device)
    head = CandidateRouter(cached.shape[1], query.shape[1], numeric, margins, correct[:, 0].cpu(), dropout=0.,
                           k_light=cfg.rerank.k_light, k_deep=cfg.rerank.k_deep).to(device)
    batch = {"query": query, "videos": video, "mask": torch.ones(2, 2, dtype=torch.bool, device=device),
             "numeric": torch.tensor(numeric, device=device), "margin": torch.tensor(margins, device=device)}
    warmup = torch.optim.AdamW(head.parameters(), lr=3e-4)
    warmup.zero_grad(set_to_none=True)
    warmup_loss = head.loss(head(cached, batch), correct)
    warmup_loss.backward()
    if any(p.grad is not None for p in backbone.parameters()):
        raise AssertionError("Frozen backbone accumulated gradients")
    warmup.step()
    for name, p in backbone.named_parameters():
        p.requires_grad_(flags[name])
    optimizer = torch.optim.AdamW([p for p in backbone.parameters() if p.requires_grad] + list(head.parameters()), lr=1e-4)
    gradient_report = []
    for step in range(2):
        backbone.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        _, states = field_states(backbone, formatter, encoded, device)
        loss = head.loss(head(states[:, 0], batch), correct)
        loss.backward()
        nonzero = {name: float(p.grad.float().norm()) for name, p in backbone.named_parameters()
                   if "lora_" in name and p.grad is not None}
        if not any(v > 0 for n, v in nonzero.items() if "lora_B" in n):
            raise AssertionError("LoRA B has no gradient")
        if step and not any(v > 0 for n, v in nonzero.items() if "lora_A" in n):
            raise AssertionError("LoRA A has no gradient after update")
        if not torch.isfinite(loss) or not all(np.isfinite(v) for v in nonzero.values()):
            raise FloatingPointError("Smoke gradients are nonfinite")
        torch.nn.utils.clip_grad_norm_([p for p in backbone.parameters() if p.requires_grad] + list(head.parameters()), 1.)
        optimizer.step()
        gradient_report.append({"step": step, "loss": float(loss.detach()), "nonzero_lora": sum(v > 0 for v in nonzero.values())})
    backbone.eval()
    head.eval()
    with torch.inference_mode():
        _, states = field_states(backbone, formatter, encoded, device)
        expected_probability = head.probabilities(head(states[:, 0], batch)).cpu()
    backbone.save_pretrained(run_dir / "adapter")
    torch.save(head.state_dict(), run_dir / "head.pt")
    del optimizer, warmup, backbone, states, cached
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _, restored = load_router(str(resolve(cfg.router.base_model)), adapter=str(run_dir / "adapter"), device=device)
    restored.config.use_cache = False
    head.load_state_dict(torch.load(run_dir / "head.pt", map_location=device, weights_only=True))
    with torch.inference_mode():
        _, states = field_states(restored, formatter, encoded, device)
        actual = head.probabilities(head(states[:, 0], batch)).cpu()
    torch.testing.assert_close(actual, expected_probability, atol=1e-5, rtol=1e-5)
    report = {"device": str(device), "real_pretrained_4b": True, "queries": [r["qid"] for r in rows],
              "retriever_max_abs_error": error, "warmup_loss": float(warmup_loss.detach()),
              "warmup_backbone_gradients": False, "joint_gradients": gradient_report,
              "reload_max_abs_probability_error": float((actual - expected_probability).abs().max()),
              "seconds": time.perf_counter() - started}
    atomic_json(run_dir / "smoke.json", report)
    print(report, flush=True)
