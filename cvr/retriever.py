"""First-stage retriever (CLIP frame mean-pooling by default) and score statistics.

A stronger retriever (CLIP4Clip, InternVideo2, ...) can be plugged in by exporting a
query x gallery similarity matrix and passing it to ``scripts/02_retrieve.py --sim_npy``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from .data import load_frame_paths
from .utils import softmax_np


class ClipRetriever:
    def __init__(self, model_name: str, device: str = "cuda:0", batch_size: int = 64):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                inp = self.proc.tokenizer(
                    list(texts[i:i + self.batch_size]), padding=True, truncation=True,
                    max_length=77, return_tensors="pt",
                ).to(self.device)
                emb = self.model.get_text_features(**inp).float()
                out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, self.dim), np.float32)

    def encode_videos(self, video_ids: Sequence[str], frames_dir: str, n_frames: int) -> np.ndarray:
        from PIL import Image
        from tqdm import tqdm

        torch = self.torch
        sums = np.zeros((len(video_ids), self.dim), np.float64)
        counts = np.zeros(len(video_ids), np.int64)
        buf: List[tuple] = []

        def flush():
            if not buf:
                return
            pix = self.proc.image_processor([im for _, im in buf], return_tensors="pt")["pixel_values"]
            pix = pix.to(self.device, torch.float16)
            with torch.no_grad():
                emb = self.model.get_image_features(pixel_values=pix).float()
                emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
            for (vi, _), e in zip(buf, emb):
                sums[vi] += e
                counts[vi] += 1
            buf.clear()

        for vi, vid in enumerate(tqdm(video_ids, desc="encode videos")):
            for p in load_frame_paths(frames_dir, vid, n_frames):
                buf.append((vi, Image.open(p).convert("RGB")))
                if len(buf) >= self.batch_size:
                    flush()
        flush()
        missing = int((counts == 0).sum())
        if missing:
            print(f"[warn] {missing} videos have no frames; their embeddings are zero.")
        feats = sums / np.maximum(counts, 1)[:, None]
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return (feats / np.maximum(norms, 1e-12)).astype(np.float32)


def rank_topk(sim: np.ndarray, queries: Sequence[dict], gallery_ids: Sequence[str], topk: int) -> List[dict]:
    """Turn a (n_query, n_gallery) similarity matrix into retrieval rows."""
    gid_to_idx = {g: i for i, g in enumerate(gallery_ids)}
    rows = []
    for qi, q in enumerate(queries):
        s = sim[qi]
        order = np.argsort(-s, kind="stable")
        gi = gid_to_idx.get(q["video_id"])
        gt_rank = len(gallery_ids) + 1 if gi is None else int(np.flatnonzero(order == gi)[0]) + 1
        candidates = order[:topk]
        rows.append({
            "qid": q["qid"],
            "text": q["text"],
            "gt": q["video_id"],
            "gt_rank": gt_rank,
            "cands": [gallery_ids[i] for i in candidates],
            "scores": [round(float(s[i]), 6) for i in candidates],
        })
    return rows


def score_stats(scores: Sequence[float], temp: float = 0.01, k: int = 10) -> Dict[str, float]:
    s = np.asarray(scores[:k], dtype=np.float64)
    p = softmax_np(s / temp)
    ent = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p))) if len(p) > 1 else 0.0
    return {
        "top1": float(s[0]),
        "margin": float(s[0] - s[1]) if len(s) > 1 else 0.0,
        "entropy": ent,
        "p_top1": float(p[0]),
        "topk": [float(x) for x in s],
    }
