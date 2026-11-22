"""End-to-end evaluation: retriever-only / always-light / always-deep / routed (ours).

--mode router uses outputs/routes/{split}.jsonl (route action + predicted fragile type).
Deep path counterfactuals come from the 'cfgen' text server (prompt-only base or GRPO LoRA).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
from collections import Counter

from cvr.factory import make_llm, make_nli, make_verifier
from cvr.metrics import breakdown_by_tag, summarize_ranks
from cvr.rerank import deep_scores, light_scores, new_gt_rank, rerank_topk
from cvr.utils import add_common_args, gather_progress, load_config, read_jsonl, work_path, write_json, write_jsonl


async def gen(cfg, items):
    llm = make_llm(cfg, "cfgen")
    hyps = await gather_progress([llm.counterfactuals(r["text"], f, cfg.cf.m) for r, f in items], "counterfactuals")
    return hyps, llm.n_calls


async def verify(cfg, rows, actions, hyps_by_qid):
    verifier = make_verifier(cfg)
    rc = cfg.rerank

    async def one(r, a):
        if a == "light":
            return await light_scores(verifier, r, rc.k_light)
        if a == "deep":
            hyps = hyps_by_qid.get(r["qid"], [])
            if not hyps and rc.get("empty_cf_fallback"):
                return await light_scores(verifier, r, rc.k_deep)
            return await deep_scores(verifier, r, hyps, rc.k_deep, cfg.vlm.n_perm, seed=cfg.seed)
        return []

    scores = await gather_progress([one(r, a) for r, a in zip(rows, actions)], "verify")
    return scores, verifier.n_calls


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="router", choices=["router", "retriever", "light", "deep"])
    ap.add_argument("--routes", default=None, help="default: outputs/routes/{split}.jsonl")
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. grpo / prompt)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    rows = read_jsonl(work_path(cfg, "retrieval", f"{args.split}.jsonl"))

    if args.mode == "router":
        routes = {r["qid"]: r for r in read_jsonl(args.routes or work_path(cfg, "routes", f"{args.split}.jsonl"))}
        actions = [routes[r["qid"]]["action"] for r in rows]
        fragiles = [routes[r["qid"]]["fragile"] for r in rows]
    else:
        a = {"retriever": "accept"}.get(args.mode, args.mode)
        actions, fragiles = [a] * len(rows), ["auto"] * len(rows)
    fragiles = ["auto" if f == "none" else f for f in fragiles]

    t0 = time.time()
    deep_items = [(r, f) for r, a, f in zip(rows, actions, fragiles) if a == "deep"]
    hyps_by_qid, llm_calls = {}, 0
    if deep_items:
        hyps, llm_calls = asyncio.run(gen(cfg, deep_items))
        nli = make_nli(cfg)
        hyps = nli.filter_many([r["text"] for r, _ in deep_items], hyps)
        del nli
        hyps_by_qid = {r["qid"]: h for (r, _), h in zip(deep_items, hyps)}
    scores, vlm_calls = asyncio.run(verify(cfg, rows, actions, hyps_by_qid))
    elapsed = time.time() - t0

    per_query, base, new = [], [], []
    for r, a, f, s in zip(rows, actions, fragiles, scores):
        rank = new_gt_rank(r, rerank_topk(r["cands"], r["scores"], s, cfg.rerank.beta)) if s else int(r["gt_rank"])
        base.append(int(r["gt_rank"]))
        new.append(rank)
        per_query.append({"qid": r["qid"], "text": r["text"], "action": a, "fragile": f,
                          "hyps": hyps_by_qid.get(r["qid"]), "scores": s, "base_rank": r["gt_rank"], "new_rank": rank})
    n = max(len(rows), 1)
    metrics = {
        "mode": args.mode, "split": args.split, "tag": args.tag,
        **summarize_ranks(base, new),
        "route_dist": dict(Counter(actions)),
        "vlm_calls_per_query": vlm_calls / n, "llm_calls_per_query": llm_calls / n,
        "sec_per_query": elapsed / n,
        "by_tag": breakdown_by_tag([r["text"] for r in rows], base, new),
    }
    name = f"{args.split}_{args.mode}{'_' + args.tag if args.tag else ''}"
    write_jsonl(work_path(cfg, "pipeline", f"{name}.jsonl"), per_query)
    write_json(work_path(cfg, "pipeline", f"{name}_metrics.json"), metrics)
    f = metrics["flips"]
    print(f"[{name}] R@1={metrics['new']['R@1']:.2f} (base {metrics['base']['R@1']:.2f}) HFR={f['HFR']:.2f}% "
          f"VLM calls/q={metrics['vlm_calls_per_query']:.2f} routes={metrics['route_dist']}")


if __name__ == "__main__":
    main()
