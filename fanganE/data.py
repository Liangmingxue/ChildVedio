"""Load the frozen fanganC aligned snapshot and construct fanganE inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fanganC.common import digest, legacy, read_json, read_jsonl, resolve, snapshot_config
from fanganD.lib import K, ListwiseFusion, build

from .losses import build_teacher_targets
from .policy import allocate_budget, softmax_np


def load_bundle(source: str | Path, teacher_temperature: float = 0.25) -> dict:
    """Load the nine-gallery protocol without touching the original fanganC run."""
    source = resolve(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Source run directory does not exist: {source}")

    snapshot = source / "snapshot"
    manifest = read_json(snapshot / "manifest.json")
    for name, checksum in manifest.get("frozen_sha256", {}).items():
        path = snapshot / name
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Frozen source snapshot checksum mismatch: {path}")

    alignment_hashes = manifest.get("alignment", {}).get("sha256", {})
    gallery_file = source / "alignment" / "galleries.json"
    expected_gallery_hash = alignment_hashes.get("galleries.json")
    if expected_gallery_hash and digest(gallery_file) != expected_gallery_hash:
        raise ValueError("Aligned gallery split checksum mismatch")

    rows = read_jsonl(snapshot / "router_data.jsonl")
    groups = read_json(gallery_file)
    if len(groups) != 9 or len(rows) != 9000 or any(len(group) != 1000 for group in groups):
        raise ValueError("Expected nine disjoint galleries of 1,000 videos and 9,000 rows")

    video_to_gallery = {
        str(video_id): gallery_id
        for gallery_id, group in enumerate(groups)
        for video_id in group
    }
    if len(video_to_gallery) != 9000:
        raise ValueError("Gallery target-video IDs are not unique")

    gallery = np.asarray([video_to_gallery[str(row["gt"])] for row in rows], dtype=np.int64)
    counts = np.bincount(gallery, minlength=9)
    if not np.all(counts == 1000):
        raise ValueError(f"Expected 1,000 queries per gallery, got {counts.tolist()}")

    cfg, stage = snapshot_config(source)
    costs = np.asarray(legacy().Replay.from_rows(rows, stage).calls, dtype=np.int64)
    if costs.shape != (len(rows), 3):
        raise ValueError(f"Unexpected VLM call-cost shape: {costs.shape}")

    archive = source / "features" / "clip.npz"
    clip_manifest_path = source / "features" / "clip_manifest.json"
    if not archive.is_file() or not clip_manifest_path.is_file():
        raise FileNotFoundError(
            "Missing frozen CLIP candidate features. Expected "
            f"{archive} and {clip_manifest_path}; run the fanganC feature stage first."
        )
    clip_manifest = read_json(clip_manifest_path)
    if digest(archive) != clip_manifest.get("sha256"):
        raise ValueError("CLIP feature archive checksum mismatch")

    with np.load(archive, allow_pickle=False) as saved:
        video_ids = saved["video_ids"].tolist()
        video_features = np.asarray(saved["video"], dtype=np.float32)
    video_to_index = {
        (value.decode("utf-8") if isinstance(value, bytes) else str(value)): index
        for index, value in enumerate(video_ids)
    }
    if video_features.ndim != 2 or not np.isfinite(video_features).all():
        raise ValueError("CLIP video features must be a finite [N, D] array")

    candidate_ids = []
    for row_index, row in enumerate(rows):
        candidates = row["cands"]
        if isinstance(candidates, str):
            from fanganD.lib import parse
            candidates = parse(candidates)
        if len(candidates) != K:
            raise ValueError(f"{row['qid']} has {len(candidates)} candidates; expected {K}")
        if any(video_to_gallery.get(str(video_id)) != int(gallery[row_index]) for video_id in candidates):
            raise ValueError(f"{row['qid']} contains a candidate from a different gallery")
        try:
            candidate_ids.append([video_to_index[str(video_id)] for video_id in candidates])
        except KeyError as error:
            raise ValueError(f"Candidate video missing from CLIP archive: {error.args[0]}") from error
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

    train_ids = np.flatnonzero(gallery < 4)
    dev_ids = np.flatnonzero(gallery == 6)
    calib_ids = np.flatnonzero(gallery == 7)
    holdout_ids = np.flatnonzero(gallery == 8)
    if any(len(index) != 1000 for index in (dev_ids, calib_ids, holdout_ids)):
        raise ValueError("Development, calibration and holdout galleries must each contain 1,000 rows")

    normalized_views = []
    initial_views = []
    d0_models = {}
    gt_index = None
    for path_index, use_deep in enumerate((False, True)):
        path_data = build(rows, use_light=True, use_deep=use_deep)
        current_gt = np.asarray([target for _, target in path_data], dtype=np.int64)
        if gt_index is None:
            gt_index = current_gt
        elif not np.array_equal(gt_index, current_gt):
            raise ValueError("Light and deep feature builders produced different GT indices")

        fusion = ListwiseFusion(seed=1).fit([path_data[int(i)] for i in train_ids])
        raw_scores = np.stack([fusion.score(features) for features, _ in path_data]).astype(np.float32)
        normalized = np.stack([
            (features - fusion.mu) / fusion.sd for features, _ in path_data
        ]).astype(np.float32)
        normalized_views.append(normalized)
        initial_views.append(raw_scores)
        prefix = "deep" if use_deep else "light"
        d0_models[f"{prefix}_mu"] = fusion.mu
        d0_models[f"{prefix}_sd"] = fusion.sd
        d0_models[f"{prefix}_w1"] = fusion.W1
        d0_models[f"{prefix}_w"] = fusion.w

    numeric = np.stack(normalized_views, axis=1)
    initial_scores = np.stack(initial_views, axis=1)
    teacher, teacher_mask, route_soft, route_hard = build_teacher_targets(
        rows,
        gt_index,
        num_candidates=K,
        temperature=teacher_temperature,
    )

    light_probs = softmax_np(initial_scores[:, 0])
    sorted_probs = np.sort(light_probs, axis=1)
    light_gap = sorted_probs[:, -1] - sorted_probs[:, -2]
    fixed_route = allocate_budget(-light_gap, costs, gallery)

    return {
        "source": source,
        "rows": rows,
        "gallery": gallery,
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "calib_ids": calib_ids,
        "holdout_ids": holdout_ids,
        "gt_index": gt_index,
        "original_rank": np.asarray([int(row["gt_rank"]) for row in rows], dtype=np.int64),
        "costs": costs,
        "numeric": numeric,
        "initial_scores": initial_scores,
        "teacher": teacher,
        "teacher_mask": teacher_mask,
        "route_soft": route_soft,
        "route_hard": route_hard,
        "video_features": video_features,
        "candidate_ids": candidate_ids,
        "fixed_route": fixed_route,
        "d0_models": d0_models,
        "stage_config": stage,
        "base_model": str(resolve(stage.router.base_model)),
        "source_manifest": manifest,
        "clip_manifest": clip_manifest,
    }
