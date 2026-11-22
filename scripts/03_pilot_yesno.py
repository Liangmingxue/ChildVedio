"""Stage-0 pilot (go / no-go): does naive yes/no VLM re-ranking cause harmful flips?

Reports HFR (harmful flip rate), helpful flips, false-positive "Yes" rate on retrieved hard
negatives, and a breakdown by heuristic query type. Needs the VLM server (scripts/serve_vlm.sh).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import random

import numpy as np

from cvr.factory import make_verifier
from cvr.metrics import breakdown_by_tag, summarize_ranks
from cvr.rerank import new_gt_rank, rerank_topk
from cvr.utils import add_common_args, gather_progress, load_config, read_jsonl, work_path, write_json, write_jsonl


async def run(cfg, rows, k, beta):
    verifier = make_verifier(cfg)
    coros = [verifier.yes_log_odds(r["text"], v) for r in rows for v in r["cands"][:k]]
    flat = await gather_progress(coros, "yes/no")
    out, i = [], 0
    for r in rows:
        n = min(k, len(r["cands"]))
        log_odds = flat[i:i + n]
        yes = (1 / (1 + np.exp(-np.clip(log_odds, -60, 60)))).tolist()
        i += n
        scores = log_odds if beta == 1.0 else yes
        new = rerank_topk(r["cands"], r["scores"], scores, beta)
        out.append({**r, "yes": yes, "yes_log_odds": log_odds, "new_rank": new_gt_rank(r, new)})
    return out, verifier.n_calls


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=0, help="limit #queries (0 = all)")
    ap.add_argument("--beta", type=float, default=None, help="default: rerank.beta")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    rows = read_jsonl(work_path(cfg, "retrieval", f"{args.split}.jsonl"))
    if args.n:
        rows = random.Random(cfg.seed).sample(rows, min(args.n, len(rows)))
    beta = cfg.rerank.beta if args.beta is None else args.beta

    out, n_calls = asyncio.run(run(cfg, rows, args.k, beta))
    base = [r["gt_rank"] for r in out]
    new = [r["new_rank"] for r in out]

    pos_yes, neg_yes = [], []
    for r in out:
        for v, s in zip(r["cands"][: args.k], r["yes"]):
            (pos_yes if v == r["gt"] else neg_yes).append(s > 0.5)
    metrics = {
        "k": args.k, "beta": beta, "vlm_calls_per_query": n_calls / max(len(out), 1),
        "vlm_model": cfg.vlm.model, "n_frames": cfg.frames.n_frames,
        "yes_no_token_ids": cfg.vlm.get("yes_no_token_ids"),
        **summarize_ranks(base, new),
        "yes_rate_on_gt": float(np.mean(pos_yes)) if pos_yes else None,
        "yes_rate_on_hard_negatives": float(np.mean(neg_yes)) if neg_yes else None,
        "by_tag": breakdown_by_tag([r["text"] for r in out], base, new),
    }
    write_jsonl(work_path(cfg, "pilot", f"{args.split}_yesno_k{args.k}.jsonl"), out)
    write_json(work_path(cfg, "pilot", f"{args.split}_yesno_k{args.k}_metrics.json"), metrics)
    f = metrics["flips"]
    print(f"R@1 {metrics['base']['R@1']:.2f} -> {metrics['new']['R@1']:.2f} | HFR={f['HFR']:.2f}% "
          f"HelpR={f['HelpR']:.2f}% | yes-rate gt={metrics['yes_rate_on_gt']} neg={metrics['yes_rate_on_hard_negatives']}")


if __name__ == "__main__":
    main()
