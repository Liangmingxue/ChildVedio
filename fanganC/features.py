"""Frozen feature extraction using the repository's actual retriever and formatter."""
from __future__ import annotations

import gc
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import (atomic_json, digest, fingerprint, frozen_rows, module_from_file,
                     read_json, read_jsonl, resolve, snapshot_config)
from cvr.retriever import score_stats
from cvr.router import JevFormatter, build_state, field_states, load_router


CLIP_QUERY_BATCH_SIZE = 64
CLIP_VIDEO_BATCH_SIZE = 16


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def numeric_features(rows, cfg):
    values = []
    for row in rows:
        stats = score_stats(row["scores"], cfg.retriever.stat_temp, cfg.retriever.stat_k)
        top = np.asarray(row["scores"][:5], dtype=np.float32)
        if len(top) != 5:
            raise ValueError("The experiment requires at least five candidates")
        values.append([*top, *(top[0] - top[1:]), stats["entropy"], stats["p_top1"], len(row["text"].split())])
    return np.asarray(values, dtype=np.float32)


def backbone_signature(base):
    directory = resolve(base)
    paths = sorted(directory.glob("*.safetensors")) + [directory / "config.json", directory / "tokenizer.json"]
    if not paths or not all(p.is_file() for p in paths):
        raise ValueError("A complete local Qwen3 checkpoint is required")
    return {p.name: digest(p) for p in paths}


def clip_source_identity():
    upstream = resolve("data/msrvtt/source/clip4clip")
    return {"weight": digest(resolve("data/msrvtt/source/qbnorm_clip4clip_msrvtt_7k.bin")),
            "source": {str(p.relative_to(upstream)): digest(p) for p in sorted(upstream.rglob("*.py"))},
            "libraries": {name: version(name) for name in ("torch", "torchvision", "numpy", "pillow", "ftfy")}}


@torch.inference_mode()
def clip_features(run_dir, device):
    config, cfg = snapshot_config(run_dir)
    train, test, manifest = frozen_rows(run_dir)
    rows = train + test
    temporal = getattr(config, "temporal_candidates", False)
    folder = Path(run_dir) / "features"
    folder.mkdir(parents=True, exist_ok=True)
    upstream = module_from_file("temp/stage0_clip4clip.py")
    splits = ("train", "test")
    data_dir = resolve(cfg.paths.data_dir)
    source_queries = {split: read_jsonl(data_dir / f"queries_{split}.jsonl") for split in splits}
    source_text = {r["qid"]: r["text"] for queries in source_queries.values() for r in queries}
    if any(source_text.get(r["qid"]) != r["text"] for r in rows):
        raise ValueError("A frozen query differs from the original retrieval input")
    source_videos = read_jsonl(data_dir / "videos.jsonl")
    galleries = {split: [r for r in source_videos if r["split"] == split] for split in splits}
    gallery = [r for videos in galleries.values() for r in videos]
    gallery_index = {r["video_id"]: i for i, r in enumerate(gallery)}
    needed = {v for row in rows for v in row["cands"]}
    if not needed.issubset(gallery_index):
        raise ValueError("A candidate video is absent from videos.jsonl")
    identity = {"snapshot": manifest["frozen_sha256"], "encoder": clip_source_identity(),
                "export_code": digest(Path(__file__)), "upstream": digest(resolve("temp/stage0_clip4clip.py")),
                "preprocess": "original cv2 1fps; uniform max12; normalize frames, masked mean, normalize video",
                "temporal_frames": temporal,
                "query_order": {split: fingerprint([(r["qid"], r["text"]) for r in queries])
                                for split, queries in source_queries.items()},
                "gallery_sizes": {split: len(videos) for split, videos in galleries.items()},
                "batch_sizes": {"query": CLIP_QUERY_BATCH_SIZE, "video": CLIP_VIDEO_BATCH_SIZE},
                "gallery": [(r["video_id"], r["path"], Path(r["path"]).stat().st_size,
                             Path(r["path"]).stat().st_mtime_ns) for r in gallery]}
    signature = fingerprint(identity)
    meta_path = folder / "clip_manifest.json"
    clip_path = folder / "clip.npz"
    if meta_path.exists():
        previous = read_json(meta_path)
        if previous.get("complete"):
            if previous["signature"] != signature:
                raise ValueError("Feature extraction inputs changed; use a new run directory")
            if clip_path.is_file() and digest(clip_path) == previous["sha256"]:
                return
    cache = folder / "clip_batches" / signature
    atomic_json(meta_path, {"signature": signature, "identity": identity, "complete": False})
    started = time.perf_counter()
    upstream.cv2.setNumThreads(1)
    model = upstream.load_model(device)
    qids = [r["qid"] for r in rows]
    query_file = cache / "queries.npz"
    # FP16 encoding depends on batch shape. Keep each source split's original
    # order and batch boundaries, including rows outside the frozen snapshot.
    if not query_file.exists():
        tokenizer = upstream.SimpleTokenizer()
        query_vectors = {}
        for queries in source_queries.values():
            tokens = torch.zeros((len(queries), upstream.MAX_WORDS), dtype=torch.long)
            for i, row in enumerate(queries):
                token_ids = [tokenizer.encoder["<|startoftext|>"]]
                token_ids += tokenizer.encode(row["text"])[:upstream.MAX_WORDS - 2]
                token_ids += [tokenizer.encoder["<|endoftext|>"]]
                tokens[i, :len(token_ids)] = torch.tensor(token_ids)
            embeddings = [F.normalize(model.clip.encode_text(batch.to(device)).float(), dim=-1).cpu().numpy()
                          for batch in tokens.split(CLIP_QUERY_BATCH_SIZE)]
            query_vectors.update(zip((r["qid"] for r in queries), np.concatenate(embeddings)))
        save_arrays(query_file, query=np.stack([query_vectors[qid] for qid in qids]), qids=np.array(qids))
    chunks = []
    frame_chunks, mask_chunks = [], []
    processed = 0
    chunk_size = 8 * CLIP_VIDEO_BATCH_SIZE
    for split, videos in galleries.items():
        for start in range(0, len(videos), chunk_size):
            chunk = videos[start:start + chunk_size]
            ids = [r["video_id"] for r in chunk]
            file = cache / f"{split}-{start:06d}.npz"
            if not file.exists():
                loader = DataLoader(upstream.Gallery(chunk), batch_size=CLIP_VIDEO_BATCH_SIZE,
                                    num_workers=config.workers, pin_memory=True)
                parts = []
                frame_parts, mask_parts = [], []
                for frames, mask in loader:
                    frames, mask = frames.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                    visual = model.get_visual_output(frames.flatten(0, 1), mask, shaped=True,
                                                     video_frame=upstream.MAX_FRAMES)
                    frames_normalized = F.normalize(visual, dim=-1)
                    pooled = model._mean_pooling_for_similarity_visual(frames_normalized, mask)
                    parts.append(F.normalize(pooled, dim=-1).cpu().numpy())
                    if temporal:
                        frame_parts.append(frames_normalized.to(torch.float16).cpu().numpy())
                        mask_parts.append(mask.cpu().numpy())
                arrays = {"video": np.concatenate(parts), "ids": np.array(ids)}
                if temporal:
                    arrays.update(frames=np.concatenate(frame_parts), frame_mask=np.concatenate(mask_parts))
                save_arrays(file, **arrays)
            with np.load(file, allow_pickle=False) as saved:
                video_chunk = saved["video"]
                if (saved["ids"].tolist() != ids or video_chunk.ndim != 2 or len(video_chunk) != len(ids)
                        or not np.isfinite(video_chunk).all()):
                    raise ValueError("Video feature chunk is invalid")
                chunks.append(video_chunk)
                if temporal:
                    frames, frame_mask = saved["frames"], saved["frame_mask"]
                    validate_frame_features(frames, frame_mask, video_chunk.shape)
                    frame_chunks.append(frames)
                    mask_chunks.append(frame_mask)
            processed += len(chunk)
            print(f"CLIP gallery {processed}/{len(gallery)}", flush=True)
    video_ids = sorted(needed)
    selected = [gallery_index[v] for v in video_ids]
    video = np.concatenate(chunks)[selected]
    with np.load(query_file, allow_pickle=False) as saved:
        if saved["qids"].tolist() != qids:
            raise ValueError("Query feature order differs")
        query = saved["query"].copy()
    if query.shape != (len(rows), video.shape[1]) or not np.isfinite(query).all() or not np.isfinite(video).all():
        raise ValueError("CLIP feature shapes or finite values are invalid")
    index = {v: i for i, v in enumerate(video_ids)}
    scale = float(model.clip.logit_scale.exp())
    errors = [float(np.max(np.abs(scale * (video[[index[v] for v in row["cands"]]] @ q) - np.array(row["scores"]))))
              for row, q in zip(rows, query)]
    if not np.isfinite(errors).all() or max(errors) > .02:
        raise ValueError(f"CLIP features do not reproduce saved retrieval scores: max error={max(errors):.6f}")
    arrays = {"query": query, "video": video, "qids": np.array(qids), "video_ids": np.array(video_ids)}
    if temporal:
        arrays.update(frames=np.concatenate(frame_chunks)[selected], frame_mask=np.concatenate(mask_chunks)[selected])
    save_arrays(clip_path, **arrays)
    atomic_json(meta_path, {"signature": signature, "identity": identity, "complete": True,
                           "sha256": digest(clip_path), "score_max_abs_error": max(errors),
                           "query_count": len(rows), "video_count": len(video_ids),
                           "seconds": time.perf_counter() - started})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode(rows, cfg, formatter):
    return [formatter.encode(build_state(r, cfg.retriever.stat_temp, cfg.retriever.stat_k)) for r in rows]


def validate_frame_features(frames, mask, pooled_shape):
    if (frames.ndim != 3 or mask.dtype != np.bool_ or frames.shape[:2] != mask.shape
            or (frames.shape[0], frames.shape[-1]) != pooled_shape or not np.isfinite(frames).all()
            or not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("Invalid ordered CLIP4Clip frame features")


@torch.inference_mode()
def text_features(run_dir, device):
    config, cfg = snapshot_config(run_dir)
    train, test, manifest = frozen_rows(run_dir)
    rows = train + test
    folder = Path(run_dir) / "features"
    identity = {"snapshot": manifest["frozen_sha256"], "backbone": backbone_signature(cfg.router.base_model),
                "libraries": {name: version(name) for name in ("torch", "transformers", "peft")},
                "router_code": digest(resolve("cvr/router.py")), "export_code": digest(Path(__file__))}
    signature = fingerprint(identity)
    meta = folder / "text_manifest.json"
    text_path = folder / "text.npy"
    if meta.exists():
        previous = read_json(meta)
        if previous["signature"] != signature:
            raise ValueError("Text cache inputs changed")
        if previous.get("complete") and text_path.is_file() and digest(text_path) == previous["sha256"]:
            return
    atomic_json(meta, {"signature": signature, "identity": identity, "complete": False})
    started = time.perf_counter()
    tokenizer, model = load_router(str(resolve(cfg.router.base_model)), device=device)
    model.config.use_cache = False
    formatter = JevFormatter(tokenizer, max_len=cfg.router.max_len)
    encoded = encode(rows, cfg, formatter)
    chunks = []
    for start in range(0, len(rows), 128):
        file = folder / "text_chunks" / f"{start:06d}.npz"
        if not file.exists():
            parts = []
            for offset in range(start, min(start + 128, len(rows)), config.eval_batch_size):
                end = min(offset + config.eval_batch_size, start + 128, len(rows))
                _, states = field_states(model, formatter, encoded[offset:end], device)
                parts.append(states[:, 0].float().cpu().numpy())
            save_arrays(file, states=np.concatenate(parts), qids=np.array([r["qid"] for r in rows[start:start + 128]]))
        with np.load(file, allow_pickle=False) as saved:
            if saved["qids"].tolist() != [r["qid"] for r in rows[start:start + 128]]:
                raise ValueError("Text cache order differs")
            chunks.append(saved["states"].copy())
        print(f"Qwen states {min(start + 128, len(rows))}/{len(rows)}", flush=True)
    states = np.concatenate(chunks)
    if states.shape != (len(rows), model.config.hidden_size) or not np.isfinite(states).all():
        raise ValueError("Text feature shapes or finite values are invalid")
    temporary = text_path.with_name(text_path.name + ".part")
    with temporary.open("wb") as stream:
        np.save(stream, states)
    temporary.replace(text_path)
    atomic_json(meta, {"signature": signature, "identity": identity, "complete": True,
                       "sha256": digest(text_path), "seconds": time.perf_counter() - started})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FeatureStore:
    def __init__(self, run_dir):
        self.config, self.cfg = snapshot_config(run_dir)
        train, test, self.manifest = frozen_rows(run_dir)
        self.rows = train + test
        self.temporal = getattr(self.config, "temporal_candidates", False)
        self.train_indices = [i for i, r in enumerate(train) if r["split"] == "train"]
        self.dev_indices = [i for i, r in enumerate(train) if r["split"] == "dev"]
        self.calib_indices = [i for i, r in enumerate(train) if r["split"] == "calib"]
        self.test_indices = list(range(len(train), len(self.rows)))
        folder = Path(run_dir) / "features"
        self.feature_manifests = {}
        for name, data in (("clip", "clip.npz"), ("text", "text.npy")):
            meta = read_json(folder / f"{name}_manifest.json")
            if not meta["complete"] or digest(folder / data) != meta["sha256"]:
                raise ValueError(f"Incomplete or changed {name} features")
            if meta["identity"]["snapshot"] != self.manifest["frozen_sha256"]:
                raise ValueError("Feature cache belongs to another snapshot")
            if meta["identity"]["export_code"] != digest(Path(__file__)):
                raise ValueError("Feature export implementation changed")
            self.feature_manifests[name] = meta
        clip_identity = self.feature_manifests["clip"]["identity"]
        if clip_identity["encoder"] != clip_source_identity():
            raise ValueError("CLIP weights, preprocessing implementation or libraries changed")
        for _, path, size, modified in clip_identity["gallery"]:
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != (size, modified):
                raise ValueError(f"Retrieval gallery video changed: {path}")
        text_identity = self.feature_manifests["text"]["identity"]
        if text_identity["backbone"] != backbone_signature(self.cfg.router.base_model):
            raise ValueError("Qwen3 backbone weights or tokenizer changed")
        if text_identity["libraries"] != {name: version(name) for name in ("torch", "transformers", "peft")}:
            raise ValueError("Text encoder libraries changed")
        with np.load(folder / "clip.npz", allow_pickle=False) as saved:
            if saved["qids"].tolist() != [r["qid"] for r in self.rows]:
                raise ValueError("Feature row IDs differ")
            self.query, self.video = saved["query"].copy(), saved["video"].copy()
            mapping = {key: i for i, key in enumerate(saved["video_ids"].tolist())}
            if self.temporal:
                self.frames, self.frame_mask = saved["frames"], saved["frame_mask"]
                validate_frame_features(self.frames, self.frame_mask, self.video.shape)
        if (self.query.shape[0] != len(self.rows) or self.query.shape[1] != self.video.shape[1]
                or not np.isfinite(self.query).all() or not np.isfinite(self.video).all()):
            raise ValueError("Invalid CLIP feature arrays")
        candidate_count = max(self.cfg.rerank.k_light, self.cfg.rerank.k_deep)
        self.indices = np.zeros((len(self.rows), candidate_count), dtype=np.int64)
        self.mask = np.zeros_like(self.indices, dtype=bool)
        for i, row in enumerate(self.rows):
            ids = [mapping[v] for v in row["cands"][:candidate_count]]
            self.indices[i, :len(ids)], self.mask[i, :len(ids)] = ids, True
        self.numeric = numeric_features(self.rows, self.cfg)
        self.margins = np.array([r["scores"][0] - r["scores"][1] for r in self.rows], dtype=np.float32)
        self.hidden = np.load(folder / "text.npy", mmap_mode="r")
        self.correct = np.array([[r["ranks"][a] == 1 for a in ("accept", "light", "deep")] for r in self.rows])
        if len(self.hidden) != len(self.rows) or not np.isfinite(self.hidden).all():
            raise ValueError("Invalid text states")

    def batch(self, indices, device, *, include_candidates=True):
        arrays = {"numeric": self.numeric[indices], "margin": self.margins[indices]}
        if include_candidates:
            candidates = self.indices[indices]
            arrays.update(query=self.query[indices], videos=self.video[candidates], mask=self.mask[indices])
            if self.temporal:
                arrays.update(frames=self.frames[candidates], frame_mask=self.frame_mask[candidates])
        return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}
