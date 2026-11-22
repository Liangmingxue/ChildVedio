"""Create test-sized training galleries and score their paths with the test protocol."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import random

import numpy as np

from cvr.factory import make_llm, make_nli, make_verifier
from cvr.retriever import rank_topk
from cvr.utils import gather_progress, load_config
from .common import (atomic_json, atomic_rows, digest, fingerprint, module_from_file,
                     read_json, read_jsonl, require_local_output, resolve, settings)
from .data import source_rows


def shard_rankings(similarities, queries, gallery, gallery_size, topk, seed):
    if gallery_size < topk or len(gallery) % gallery_size:
        raise ValueError("Training videos cannot form complete test-sized galleries")
    if similarities.shape != (len(queries), len(gallery)) or len(set(gallery)) != len(gallery):
        raise ValueError("Invalid source similarity matrix or gallery IDs")
    positions = list(range(len(gallery)))
    random.Random(seed).shuffle(positions)
    by_video = {query["video_id"]: index for index, query in enumerate(queries)}
    if len(by_video) != len(queries) or set(by_video) != set(gallery):
        raise ValueError("Aligned MSR-VTT requires one source query per training video")
    ranks, shards = {}, []
    for start in range(0, len(gallery), gallery_size):
        columns = sorted(positions[start:start + gallery_size])
        videos = [gallery[index] for index in columns]
        indexes = [by_video[video] for video in videos]
        matrix = similarities[np.ix_(indexes, columns)]
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite scores in aligned training gallery")
        entries = rank_topk(matrix, [queries[index] for index in indexes], videos, topk)
        ranks.update((entry["qid"], entry) for entry in entries)
        shards.append(videos)
    return ranks, shards


def alignment_inputs(run_dir, config, cfg, source, test, provenance, *, prepare=False):
    data = resolve(cfg.paths.data_dir)
    queries_file = data / "queries_train.jsonl"
    gallery_file = data / "videos.jsonl"
    sim_file = resolve(cfg.paths.work_dir) / "clip4clip_train_sim.npy"
    queries = read_jsonl(queries_file)
    videos = read_jsonl(gallery_file)
    gallery = [row["video_id"] for row in videos if row["split"] == "train"]
    test_gallery = {row["video_id"] for row in videos if row["split"] == "test"}
    if len(source) != len(queries) or {r["qid"] for r in source} != {r["qid"] for r in queries}:
        raise ValueError("Complete train/dev/calib annotations are required before alignment")
    if set(gallery) & test_gallery or not test or {r["gt"] for r in test} != test_gallery:
        raise ValueError("Training galleries must be disjoint from the complete test gallery")
    identity = {"source": provenance["sources"], "gallery_size": len(test_gallery), "seed": config.seed,
                "topk": cfg.retriever.topk, "deep_mode": "auto", "stage_config": digest(config.stage_config),
                "inputs": {str(path): digest(path) for path in (queries_file, gallery_file, sim_file)},
                "code": {name: digest(resolve(name)) for name in
                         ("fanganC/alignment.py", "cvr/counterfactual.py", "cvr/factory.py", "cvr/rerank.py",
                          "cvr/retriever.py", "cvr/vlm.py", "scripts/05_build_router_data.py")}}
    signature = fingerprint(identity)
    folder = Path(run_dir) / "alignment"
    receipt_file = folder / "manifest.json"
    if receipt_file.is_file():
        receipt = read_json(receipt_file)
        if receipt["signature"] != signature:
            raise ValueError("Aligned annotation inputs changed; use a new run directory")
        for name, expected in receipt["sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError(f"Aligned annotation input changed: {name}")
        return folder, read_jsonl(folder / "ranking.jsonl"), receipt
    if not prepare:
        raise ValueError("Prepare and score the aligned annotations before freezing a snapshot")
    matrix = np.load(sim_file, mmap_mode="r", allow_pickle=False)
    ranks, shards = shard_rankings(matrix, queries, gallery, len(test_gallery), cfg.retriever.topk, config.seed)
    rankings = []
    for row in source:
        ranked = ranks[row["qid"]]
        if (ranked["gt"], ranked["text"]) != (row["gt"], row["text"]):
            raise ValueError(f"Aligned query identity differs for {row['qid']}")
        rankings.append({**ranked, "split": row["split"]})
    atomic_rows(folder / "ranking.jsonl", rankings)
    atomic_json(folder / "galleries.json", shards)
    receipt = {"signature": signature, "identity": identity, "queries": len(rankings),
               "sha256": {name: digest(folder / name) for name in ("ranking.jsonl", "galleries.json")}}
    atomic_json(receipt_file, receipt)
    print(f"Aligned {len(rankings)} labelled queries in {len(shards)} galleries of {len(test_gallery)} videos", flush=True)
    return folder, rankings, receipt


def prefix(path, rankings):
    rows = read_jsonl(path) if path.is_file() else []
    if [r["qid"] for r in rows] != [r["qid"] for r in rankings[:len(rows)]]:
        raise ValueError(f"Unexpected query order in {path}")
    return rows


async def auto_hypotheses(cfg, folder, rankings):
    path = folder / "hypotheses.jsonl"
    items = prefix(path, rankings)
    if len(items) == len(rankings):
        return
    llm, nli = make_llm(cfg, "cfgen"), make_nli(cfg)
    async with llm.client:
        for start in range(len(items), len(rankings), 128):
            rows = rankings[start:start + 128]
            generated = await gather_progress(
                [llm.counterfactuals(row["text"], "auto", cfg.cf.m) for row in rows], "aligned auto hypotheses")
            filtered = nli.filter_many([row["text"] for row in rows], generated)
            items.extend({"qid": row["qid"], "hyps": hyps} for row, hyps in zip(rows, filtered, strict=True))
            atomic_rows(path, items)
            print(f"Aligned hypotheses {len(items)}/{len(rankings)}", flush=True)


async def aligned_scores(verifier, cfg, row, hyps, old):
    """Reuse a verifier answer only when the query, candidate and prompt coincide."""
    rc = cfg.rerank
    light_ids, deep_ids = row["cands"][:rc.k_light], row["cands"][:rc.k_deep]
    yes = dict(zip(old["cands"][:rc.k_light], old["light_scores"], strict=True))
    if rc.empty_cf_fallback and not old["hyps"]:
        yes.update(zip(old["cands"][:rc.k_deep], old["deep_scores"], strict=True))
    yes_ids = dict.fromkeys(light_ids + deep_ids) if rc.empty_cf_fallback and not hyps else light_ids
    missing = [video for video in yes_ids if video not in yes]
    yes.update(zip(missing, await asyncio.gather(*(verifier.yes_prob(row["text"], video) for video in missing)), strict=True))
    light = [yes[video] for video in light_ids]
    if rc.empty_cf_fallback and not hyps:
        return light, [yes[video] for video in deep_ids]
    deep = dict(zip(old["cands"][:rc.k_deep], old["deep_scores"], strict=True)) if hyps == old["hyps"] else {}
    missing = [video for video in deep_ids if video not in deep]
    scored = await asyncio.gather(*(
        verifier.forced_choice(row["text"], hyps, video, n_perm=cfg.vlm.n_perm, seed=cfg.seed) for video in missing
    ))
    deep.update(zip(missing, scored, strict=True))
    return light, [deep[video] for video in deep_ids]


def load_aligned_rows(folder, rankings, receipt, cfg):
    complete = folder / "scored_manifest.json"
    if not complete.is_file():
        raise ValueError("Aligned auto verification must finish before snapshot creation")
    proof = read_json(complete)
    path = folder / "scores.jsonl"
    if (proof["alignment"] != receipt["signature"] or proof["sha256"] != digest(path)
            or proof["hypotheses_sha256"] != digest(folder / "hypotheses.jsonl")):
        raise ValueError("Aligned auto verification differs from its completion manifest")
    rows = prefix(path, rankings)
    builder = module_from_file("scripts/05_build_router_data.py")
    for row, ranked in zip(rows, rankings, strict=True):
        if row["split"] != ranked["split"] or row["cf_mode"] != "auto":
            raise ValueError(f"Invalid aligned supervision for {row['qid']}")
        builder.validate_ranks(cfg, row, ranked)
    return rows


async def score_paths(cfg, folder, rankings, source, receipt):
    if (folder / "scored_manifest.json").is_file():
        load_aligned_rows(folder, rankings, receipt, cfg)
        return
    hyps_file = folder / "hypotheses.jsonl"
    items = prefix(hyps_file, rankings)
    if len(items) != len(rankings):
        raise ValueError("Finish generating auto hypotheses before scoring the aligned galleries")
    path = folder / "scores.jsonl"
    rows = prefix(path, rankings)
    builder = module_from_file("scripts/05_build_router_data.py")
    for row, ranked, item in zip(rows, rankings[:len(rows)], items[:len(rows)], strict=True):
        if row["split"] != ranked["split"] or row["hyps"] != item["hyps"] or row["cf_mode"] != "auto":
            raise ValueError(f"Aligned score progress differs for {row['qid']}")
        builder.validate_ranks(cfg, row, ranked)
    old = {row["qid"]: row for row in source}
    verifier = make_verifier(cfg)
    async with verifier.client:
        for start in range(len(rows), len(rankings), 32):
            batch = rankings[start:start + 32]
            hypotheses = [item["hyps"] for item in items[start:start + len(batch)]]
            scores = await gather_progress([aligned_scores(verifier, cfg, row, hyps, old[row["qid"]])
                                            for row, hyps in zip(batch, hypotheses, strict=True)], "aligned verify")
            for row, hyps, (light, deep) in zip(batch, hypotheses, scores, strict=True):
                ranks = builder.verification_ranks(row, light, deep, cfg.rerank.beta)
                scored = {**row, "hyps": hyps, "cf_mode": "auto", "light_scores": light, "deep_scores": deep,
                          "ranks": ranks, "label_action": builder.label_action(ranks)}
                builder.validate_ranks(cfg, scored, row)
                rows.append(scored)
            atomic_rows(path, rows)
            print(f"Aligned paths {len(rows)}/{len(rankings)}", flush=True)
    atomic_json(folder / "scored_manifest.json", {"alignment": receipt["signature"], "sha256": digest(path),
                "hypotheses_sha256": digest(hyps_file), "queries": len(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "text", "score"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", default="fanganC/config.yaml")
    args = parser.parse_args()
    run_dir, config = require_local_output(args.run_dir), settings(args.config)
    cfg = load_config(config.stage_config)
    rows, test, provenance = source_rows(config, cfg)
    folder, rankings, receipt = alignment_inputs(run_dir, config, cfg, rows, test, provenance,
                                                 prepare=args.phase == "prepare")
    if args.phase == "text":
        asyncio.run(auto_hypotheses(cfg, folder, rankings))
    elif args.phase == "score":
        asyncio.run(score_paths(cfg, folder, rankings, rows, receipt))


if __name__ == "__main__":
    main()
