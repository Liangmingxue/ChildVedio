"""Freeze published labels without touching the active annotation workflow."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .common import (atomic_json, atomic_rows, digest, frozen_rows, legacy,
                     module_from_file, read_json, read_jsonl, require_local_output,
                     resolve, settings)
from cvr.utils import load_config


def published_rows(path):
    # The producer publishes via os.replace. One open/read sees one complete inode.
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return rows, hashlib.sha256(raw).hexdigest()


def merge_completed(base, extra):
    merged = {row["qid"]: row for row in base}
    if len(merged) != len(base):
        raise ValueError("Duplicate query in base annotations")
    for row in extra:
        if row["split"] != "train":
            raise ValueError("Additional annotations must belong to train")
        qid = row["qid"]
        if qid in merged:
            if merged[qid] != row:
                raise ValueError(f"Conflicting duplicate: {qid}")
            raise ValueError(f"Repeated query in additional annotations: {qid}")
        merged[qid] = row
    videos = {}
    for row in merged.values():
        split = row["split"]
        if split not in ("train", "dev", "calib"):
            raise ValueError(f"Unknown split {split}")
        if row["gt"] in videos and videos[row["gt"]] != split:
            raise ValueError("A supervision video crosses split boundaries")
        videos[row["gt"]] = split
    return list(merged.values())


def source_rows(config, stage_cfg):
    stage = resolve(stage_cfg.paths.work_dir)
    base_path, extra_path = resolve(config.base_data), resolve(config.extra_data)
    base, base_hash = published_rows(base_path)
    base_run = read_json(base_path.with_name("run.json"))
    builder = module_from_file("scripts/05_build_router_data.py")
    keys = builder.SCORING_CONFIG_KEYS
    if (any(stage_cfg[k] != base_run["config"][k] for k in keys)
            or any(stage_cfg.paths[k] != base_run["config"]["paths"][k] for k in ("data_dir", "frames_dir"))):
        raise ValueError("Stage scoring configuration differs from original labels")
    if [r["qid"] for r in base] != base_run["qids"]:
        raise ValueError("Original label order differs from its manifest")
    extra, extra_hash = published_rows(extra_path) if extra_path.exists() else ([], None)
    if extra:
        extension = read_json(extra_path.with_name("run.json"))
        if extension["source_sha256"] != base_hash:
            raise ValueError("Extension was created from different original annotations")
        if (any(stage_cfg[k] != extension["config"][k] for k in keys)
                or any(stage_cfg.paths[k] != extension["config"]["paths"][k] for k in ("data_dir", "frames_dir"))):
            raise ValueError("Extension scoring protocol differs")
        if [r["qid"] for r in extra] != extension["qids"][:len(extra)]:
            raise ValueError("Additional scored rows are not the published prefix")
    rows = merge_completed(base, extra)
    retrieval = {r["qid"]: r for r in read_jsonl(stage / "retrieval/train.jsonl")}
    builder.validate_text_labels(rows)
    for row in rows:
        builder.validate_ranks(stage_cfg, row, retrieval[row["qid"]])
    test = legacy().load_test(stage_cfg, stage).rows
    for row in test:
        for action, key, count in (("light", "light_scores", stage_cfg.rerank.k_light),
                                   ("deep", "deep_scores", stage_cfg.rerank.k_deep)):
            if len(row[key]) != min(count, len(row["cands"])) or not all(math.isfinite(v) for v in row[key]):
                raise ValueError(f"Invalid cached test scores for {row['qid']} {action}")
        if builder.verification_ranks(row, row["light_scores"], row["deep_scores"], stage_cfg.rerank.beta) != row["ranks"]:
            raise ValueError(f"Cached test ranks do not match scores for {row['qid']}")
    train_videos = {r["gt"] for r in rows}
    if train_videos & {r["gt"] for r in test} or {r["qid"] for r in rows} & {r["qid"] for r in test}:
        raise ValueError("Train and test overlap")
    return rows, test, {"sources": {str(base_path): base_hash, str(extra_path): extra_hash},
                        "base_rows": len(base), "extra_rows": len(extra)}


def create_snapshot(config_path, run_dir):
    run_dir = require_local_output(run_dir)
    destination = run_dir / "snapshot"
    if (destination / "manifest.json").exists():
        _, _, manifest = frozen_rows(run_dir)
        return manifest
    config = settings(config_path)
    stage_cfg = load_config(config.stage_config)
    rows, test, provenance = source_rows(config, stage_cfg)
    reused = ["scripts/13_stage2_router_experiment.py", "scripts/05_build_router_data.py",
              "cvr/router.py", "temp/stage0_clip4clip.py"]
    protocol = "existing labels; original gallery sizes and fragile-guided train / auto test unchanged"
    if config.get("align_verification", False):
        from .alignment import alignment_inputs, load_aligned_rows

        folder, rankings, receipt = alignment_inputs(run_dir, config, stage_cfg, rows, test, provenance)
        rows = load_aligned_rows(folder, rankings, receipt, stage_cfg)
        provenance["alignment"] = receipt
        reused.extend(receipt["identity"]["code"])
        protocol = "train/dev/calib in disjoint test-sized training galleries; auto deep on both train and test"
    budget = read_json(resolve(config.budget_report))["shared_budget"]
    if budget["queries"] != len(test):
        raise ValueError("Test budget query count differs")
    # Freeze all candidate paths, config and budget metadata before any training.
    atomic_rows(destination / "router_data.jsonl", rows)
    atomic_rows(destination / "test.jsonl", test)
    atomic_json(destination / "experiment.yaml", config)
    atomic_json(destination / "stage.yaml", stage_cfg)
    atomic_json(destination / "budget.json", budget)
    files = ["router_data.jsonl", "test.jsonl", "experiment.yaml", "stage.yaml", "budget.json"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                **provenance,
                "splits": dict(Counter(row["split"] for row in rows)), "test_rows": len(test),
                "frozen_sha256": {name: digest(destination / name) for name in files},
                "reused_code_sha256": {name: digest(resolve(name)) for name in reused},
                "protocol": protocol}
    atomic_json(destination / "manifest.json", manifest)
    print("Snapshot", manifest["splits"], "new completed", provenance["extra_rows"], flush=True)
    return manifest
