"""Build router supervision by running all three paths offline on training queries.

label_action  = cheapest path whose top-1 is correct (accept < light < deep); if none is
                correct, the path with the best GT rank (ties -> cheaper).
label_fragile = LLM judgement of what distinguishes the query from the caption of the
                hardest retrieved negative (training videos have captions).
Needs: retrieval/train.jsonl, VLM server (GPU1), text LLM server (GPU0).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from cvr.counterfactual import FRAGILE_TYPES
from cvr.factory import make_llm, make_nli, make_verifier
from cvr.rerank import deep_scores, light_scores, new_gt_rank, rerank_topk
from cvr.router import ACTIONS
from cvr.utils import (add_common_args, gather_progress, load_config, read_json, read_jsonl, split_of,
                       work_path, write_json, write_jsonl)


RETRIEVAL_FIELDS = ("text", "gt", "gt_rank", "cands", "scores")
TEXT_LABEL_FIELDS = ("neg_caption", "label_fragile", "generated_hyps", "hyps")
SCORING_CONFIG_KEYS = ("seed", "frames", "llm", "vlm", "cf", "rerank")
TEXT_BATCH_SIZE = 256


def label_action(ranks: dict) -> str:
    for a in ACTIONS:
        if ranks[a] == 1:
            return a
    best = min(ranks.values())
    return next(a for a in ACTIONS if ranks[a] == best)


def negative_captions(cfg, rows, captions):
    rng = random.Random(cfg.seed)
    negs = []
    for r in rows:
        neg = next((v for v in r["cands"] if v != r["gt"]), None)
        caps = captions.get(neg, []) if neg else []
        negs.append(rng.choice(caps) if caps else "")
    return negs


async def stage_llm(cfg, rows, captions, negs=None):
    llm = make_llm(cfg, "llm")
    if negs is None:
        negs = negative_captions(cfg, rows, captions)
    async with llm.client:
        fragile = await gather_progress(
            [llm.fragile_label(r["text"], c) if c else asyncio.sleep(0, result="none") for r, c in zip(rows, negs)],
            "fragile labels",
        )
        hyps = await gather_progress(
            [llm.counterfactuals(r["text"], f if f != "none" else "auto", cfg.cf.m) for r, f in zip(rows, fragile)],
            "counterfactuals",
        )
    return negs, fragile, hyps


def verification_ranks(row, light, deep, beta):
    return {
        "accept": int(row["gt_rank"]),
        "light": new_gt_rank(row, rerank_topk(row["cands"], row["scores"], light, beta)),
        "deep": new_gt_rank(row, rerank_topk(row["cands"], row["scores"], deep, beta)),
    }


async def stage_vlm(cfg, rows, items, out, progress, *, split=None):
    verifier = make_verifier(cfg)
    rc = cfg.rerank

    async def score(row, hyps):
        if rc.get("empty_cf_fallback") and not hyps:
            scores = await light_scores(verifier, row, max(rc.k_light, rc.k_deep))
            return scores[:rc.k_light], scores[:rc.k_deep]
        return await asyncio.gather(
            light_scores(verifier, row, rc.k_light),
            deep_scores(verifier, row, hyps, rc.k_deep, cfg.vlm.n_perm, seed=cfg.seed),
        )

    async with verifier.client:
        for first in range(len(out), len(rows), 32):
            last = min(first + 32, len(rows))
            batch, labels = rows[first:last], items[first:last]
            scores = await gather_progress(
                [score(r, item["hyps"]) for r, item in zip(batch, labels)], f"router data {first + 1}–{last}/{len(rows)}"
            )
            for r, item, (ls, ds) in zip(batch, labels, scores):
                ranks = verification_ranks(r, ls, ds, rc.beta)
                out.append({**r, **item, "light_scores": ls, "deep_scores": ds, "ranks": ranks,
                            "label_action": label_action(ranks),
                            "split": split or split_of(r["qid"], cfg.router.dev_frac, cfg.router.calib_frac)})
            write_atomic(write_jsonl, progress, out)


def write_atomic(writer, path, content):
    staging = f"{path}.part"
    writer(staging, content)
    os.replace(staging, path)


def index_queries(rows, name):
    indexed = {row["qid"]: row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError(f"{name} must contain nonempty, unique query IDs")
    return indexed


def validate_text_labels(items):
    for item in items:
        if (not isinstance(item["neg_caption"], str)
            or item["label_fragile"] not in FRAGILE_TYPES
            or any(not isinstance(item[key], list) or any(not isinstance(h, str) for h in item[key])
                   for key in ("generated_hyps", "hyps"))):
            raise ValueError(f"Invalid text labels for {item['qid']}")


def validate_ranks(cfg, row, retrieval):
    qid = row["qid"]
    if any(row[key] != retrieval[key] for key in RETRIEVAL_FIELDS):
        raise ValueError(f"Cached retrieval fields changed for {qid}")
    light, deep = row["light_scores"], row["deep_scores"]
    for scores, k in ((light, cfg.rerank.k_light), (deep, cfg.rerank.k_deep)):
        if len(scores) != min(k, len(retrieval["cands"])) or not all(math.isfinite(s) for s in scores):
            raise ValueError(f"Incomplete verifier scores for {qid}")
    ranks = verification_ranks(retrieval, light, deep, cfg.rerank.beta)
    if row["ranks"] != ranks or row["label_action"] != label_action(ranks):
        raise ValueError(f"Inconsistent verification ranks or action label for {qid}")


def extension_inputs(cfg, pool, test, old_path):
    retrieval = index_queries(pool, "Training retrieval")
    original = read_jsonl(old_path)
    cached = index_queries(original, "Existing router data")
    old_manifest = read_json(old_path.with_name("run.json"))
    old_config = old_manifest["config"]
    if (any(cfg[key] != old_config[key] for key in SCORING_CONFIG_KEYS)
        or any(cfg.paths[key] != old_config["paths"][key] for key in ("data_dir", "frames_dir"))):
        raise ValueError("Annotation configuration differs from the existing router data")
    if list(cached) != old_manifest["qids"] or not cached.keys() <= retrieval.keys():
        raise ValueError("Existing annotations do not match their manifest and training retrieval")
    validate_text_labels(original)
    for qid, row in cached.items():
        if row["split"] not in ("train", "dev", "calib"):
            raise ValueError(f"Unknown split for {qid}")
        validate_ranks(cfg, row, retrieval[qid])
    additions = [row for row in pool if row["qid"] not in cached]
    protected = {row["gt"] for row in original if row["split"] != "train"}
    if protected & {row["gt"] for row in additions}:
        raise ValueError("Additional queries overlap the original development or calibration videos")
    if {row["gt"] for row in test} & {row["gt"] for row in pool}:
        raise ValueError("Training retrieval overlaps test videos")
    return original, additions


def read_prefix(path, rows):
    items = read_jsonl(path) if path.exists() else []
    if [item["qid"] for item in items] != [row["qid"] for row in rows[:len(items)]]:
        raise ValueError(f"Unexpected query order in {path}")
    return items


def validate_scored(cfg, scores, rows, items):
    for scored, retrieval, text in zip(scores, rows, items, strict=True):
        if scored["split"] != "train" or any(scored[key] != text[key] for key in TEXT_LABEL_FIELDS):
            raise ValueError(f"Inconsistent split or text annotation for {scored['qid']}")
        validate_ranks(cfg, scored, retrieval)


def extend_router_data(cfg, pool, captions, old_path, output_dir, phase, resume):
    if output_dir.resolve() == old_path.resolve().parent:
        raise ValueError("Use a separate output directory for expanded annotations")
    test = read_jsonl(Path(cfg.paths.work_dir) / "retrieval/test.jsonl")
    original, rows = extension_inputs(cfg, pool, test, old_path)
    original_digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps([cfg, rows, test, captions, original_digest],
                                          sort_keys=True).encode()).hexdigest()
    manifest = output_dir / "run.json"
    text_path = output_dir / "text_labels.jsonl"
    progress = output_dir / "progress.jsonl"
    destination = output_dir / "router_data.jsonl"
    if manifest.exists():
        if not resume or read_json(manifest)["signature"] != signature:
            raise ValueError("Existing extension requires --resume and identical inputs")
    else:
        if any(path.exists() for path in (text_path, progress, destination)):
            raise ValueError("Extension artifacts exist without a run manifest")
        write_atomic(write_json, manifest, {
            "signature": signature, "config": cfg, "source": str(old_path),
            "source_sha256": original_digest, "reused": len(original), "new": len(rows),
            "qids": [row["qid"] for row in rows],
        })
    print(f"Reusing {len(original)} queries and annotating {len(rows)} additional queries", flush=True)

    items = read_prefix(text_path, rows)
    validate_text_labels(items)
    if phase in ("all", "text") and len(items) < len(rows):
        negs = negative_captions(cfg, rows, captions)
        nli = make_nli(cfg)
        for start in range(len(items), len(rows), TEXT_BATCH_SIZE):
            stop = min(start + TEXT_BATCH_SIZE, len(rows))
            batch = rows[start:stop]
            print(f"New text annotation {start + 1}–{stop}/{len(rows)}", flush=True)
            negatives, fragile, generated = asyncio.run(stage_llm(cfg, batch, captions, negs[start:stop]))
            filtered = nli.filter_many([row["text"] for row in batch], generated)
            added = [dict(qid=r["qid"], neg_caption=n, label_fragile=f, generated_hyps=g, hyps=h)
                     for r, n, f, g, h in zip(batch, negatives, fragile, generated, filtered, strict=True)]
            validate_text_labels(added)
            items.extend(added)
            write_atomic(write_jsonl, text_path, items)
        del nli
    if len(items) != len(rows):
        raise ValueError("Complete the text annotation phase before scoring")
    if phase == "text":
        return

    out = read_prefix(progress, rows)
    completed = len(out)
    validate_scored(cfg, out, rows[:completed], items[:completed])
    if completed < len(rows):
        asyncio.run(stage_vlm(cfg, rows, items, out, progress, split="train"))
    if len(out) != len(rows) or [row["qid"] for row in out] != [row["qid"] for row in rows]:
        raise ValueError("Incomplete scored data")
    validate_scored(cfg, out[completed:], rows[completed:], items[completed:])
    merged = original + out
    if index_queries(merged, "Complete router data").keys() != {row["qid"] for row in pool}:
        raise ValueError("Completed annotations do not cover the entire training retrieval")
    write_atomic(write_jsonl, destination, merged)
    print(f"Published {len(merged)} annotations to {destination}", flush=True)
    print("Splits", dict(Counter(row["split"] for row in merged)), flush=True)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--n", type=int, default=None, help="default: router.n_train_queries")
    scope.add_argument("--extend-from", type=Path, help="reuse this complete dataset and label missing queries")
    ap.add_argument("--output-dir", type=Path, help="separate directory for extended data and checkpoints")
    ap.add_argument("--phase", choices=("all", "text", "score"), default="all")
    ap.add_argument("--resume", action="store_true", help="resume the same labelled queries and completed scores")
    args = ap.parse_args()
    if bool(args.extend_from) != bool(args.output_dir) or (args.phase != "all" and not args.extend_from):
        ap.error("--extend-from and --output-dir are required together for extension phases")
    cfg = load_config(args.config, args.set)
    rows = read_jsonl(work_path(cfg, "retrieval", "train.jsonl"))
    captions = read_json(os.path.join(cfg.paths.data_dir, "captions.json"))
    if args.extend_from:
        extend_router_data(cfg, rows, captions, args.extend_from, args.output_dir, args.phase, args.resume)
        return
    n = cfg.router.n_train_queries if args.n is None else args.n
    if n < 1 or not rows:
        ap.error("router supervision needs a positive sample size and nonempty training retrieval")
    rows = random.Random(cfg.seed).sample(rows, min(n, len(rows)))

    stage_path = work_path(cfg, "router_data", "text_labels.jsonl")
    progress = work_path(cfg, "router_data", "progress.jsonl")
    manifest = work_path(cfg, "router_data", "run.json")
    signature = hashlib.sha256(json.dumps([cfg, rows, captions], sort_keys=True).encode()).hexdigest()
    stored = args.resume and os.path.exists(manifest)
    if stored and read_json(manifest)["signature"] != signature:
        raise ValueError("Router-data checkpoint inputs or configuration changed")
    if not stored:
        for path in (stage_path, progress):
            if os.path.exists(path):
                os.remove(path)
        write_json(manifest, {"signature": signature, "config": cfg, "qids": [r["qid"] for r in rows]})
    if os.path.exists(stage_path):
        items = read_jsonl(stage_path)
    else:
        negs, fragile, generated = asyncio.run(stage_llm(cfg, rows, captions))
        nli = make_nli(cfg)
        hyps = nli.filter_many([r["text"] for r in rows], generated)
        del nli
        items = [dict(qid=r["qid"], neg_caption=ng, label_fragile=f, generated_hyps=g, hyps=h)
                 for r, ng, f, g, h in zip(rows, negs, fragile, generated, hyps)]
        write_jsonl(stage_path + ".part", items)
        os.replace(stage_path + ".part", stage_path)
    if [item["qid"] for item in items] != [r["qid"] for r in rows]:
        raise ValueError("Text labels do not match the requested training queries")
    print(f"Text labels ready for {len(items)} queries, {sum(bool(i['hyps']) for i in items)} with counterfactuals")

    out = read_jsonl(progress) if os.path.exists(progress) else []
    if [r["qid"] for r in out] != [r["qid"] for r in rows[:len(out)]]:
        raise ValueError("The scored query prefix does not match this run")
    if len(out) < len(rows):
        asyncio.run(stage_vlm(cfg, rows, items, out, progress))
    path = work_path(cfg, "router_data", "router_data.jsonl")
    write_jsonl(path + ".part", out)
    os.replace(path + ".part", path)
    print("action labels:", Counter(o["label_action"] for o in out))
    print("fragile labels:", Counter(o["label_fragile"] for o in out))
    print("splits:", Counter(o["split"] for o in out), "->", path)


if __name__ == "__main__":
    main()
