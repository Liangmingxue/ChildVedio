"""Compare calibrated, random, and score-margin accept/deep routing at one VLM budget.

All policies replay the same always-deep results and use the same number of deep
queries in each call-cost group. The score-margin baseline gives deep verification
to queries with the smallest Top-1 to Top-2 retrieval score gaps.
"""
import argparse
from collections import Counter, defaultdict
from math import isclose
from pathlib import Path
import random
from statistics import mean, pstdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvr.metrics import summarize_ranks
from cvr.retriever import score_stats
from cvr.utils import add_common_args, load_config, read_json, read_jsonl, work_path, write_json


def index_queries(path):
    rows = read_jsonl(path)
    by_qid = {row["qid"]: row for row in rows}
    if not rows or len(by_qid) != len(rows):
        raise ValueError(f"{path} must contain a nonempty set of unique query IDs")
    return by_qid


def load_replay(cfg, retrieval_path, routes_path, deep_path, metrics_path, split):
    retrieval = index_queries(retrieval_path)
    routes = index_queries(routes_path)
    deep = index_queries(deep_path)
    metrics = read_json(metrics_path)
    if retrieval.keys() != routes.keys() or retrieval.keys() != deep.keys():
        raise ValueError("Retrieval, routes, and always-deep results must cover the same queries")
    if metrics["mode"] != "deep" or metrics["split"] != split or metrics["new"]["n"] != len(deep):
        raise ValueError(f"{metrics_path} does not describe this always-deep evaluation")

    base_ranks, deep_ranks, deep_calls, margins = [], [], [], []
    router_deep = set()
    for index, (qid, row) in enumerate(retrieval.items()):
        scored = deep[qid]
        action = routes[qid]["action"]
        if action not in ("accept", "deep"):
            raise ValueError(f"{qid} has action {action!r}. This control expects accept/deep routes")
        if scored["action"] != "deep" or scored["text"] != row["text"] or scored["base_rank"] != row["gt_rank"]:
            raise ValueError(f"Always-deep result does not match retrieval for {qid}")
        n_scores = len(scored["scores"])
        if n_scores != min(cfg.rerank.k_deep, len(row["cands"])):
            raise ValueError(f"Unexpected number of verification scores for {qid}")
        calls_per_candidate = max(1, cfg.vlm.n_perm)
        if not scored["hyps"] and cfg.rerank.get("empty_cf_fallback"):
            calls_per_candidate = 1
        deep_calls.append(n_scores * calls_per_candidate)
        base_ranks.append(row["gt_rank"])
        deep_ranks.append(scored["new_rank"])
        margins.append(score_stats(row["scores"], k=2)["margin"])
        if action == "deep":
            router_deep.add(index)

    if not isclose(sum(deep_calls), metrics["vlm_calls_per_query"] * len(deep)):
        raise ValueError("Reconstructed VLM call count differs from the always-deep metrics. Check the config")
    return base_ranks, deep_ranks, deep_calls, margins, router_deep


def replay(base_ranks, deep_ranks, selected):
    ranks = [deep if index in selected else base
             for index, (base, deep) in enumerate(zip(base_ranks, deep_ranks))]
    return summarize_ranks(base_ranks, ranks)


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--split", default="test")
    parser.add_argument("--tag", default="prompt", help="tag of the saved always-deep evaluation")
    parser.add_argument("--trials", type=int, default=1000, help="number of random assignments")
    parser.add_argument("--output", type=Path, help="write the report to this path")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    cfg = load_config(args.config, args.set)
    work_dir = Path(cfg.paths.work_dir)
    tag = f"_{args.tag}" if args.tag else ""
    deep_name = f"{args.split}_deep{tag}"
    deep_path = work_dir / "pipeline" / f"{deep_name}.jsonl"
    metrics_path = work_dir / "pipeline" / f"{deep_name}_metrics.json"
    routes_path = work_dir / "routes" / f"{args.split}.jsonl"
    retrieval_path = work_dir / "retrieval" / f"{args.split}.jsonl"
    try:
        base_ranks, deep_ranks, deep_calls, margins, router_deep = load_replay(
            cfg, retrieval_path, routes_path, deep_path, metrics_path, args.split
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    pool_by_cost = defaultdict(list)
    for index, calls in enumerate(deep_calls):
        pool_by_cost[calls].append(index)
    budget_by_cost = Counter(deep_calls[index] for index in router_deep)
    router_metrics = replay(base_ranks, deep_ranks, router_deep)
    margin_deep = {
        index
        for cost, pool in pool_by_cost.items()
        for index in sorted(pool, key=margins.__getitem__)[: budget_by_cost[cost]]
    }
    margin_metrics = replay(base_ranks, deep_ranks, margin_deep)
    trials = []
    for seed in range(cfg.seed, cfg.seed + args.trials):
        rng = random.Random(seed)
        selected = {
            index
            for cost, pool in sorted(pool_by_cost.items())
            for index in rng.sample(pool, budget_by_cost[cost])
        }
        result = replay(base_ranks, deep_ranks, selected)
        trials.append({"seed": seed, "R@1": result["new"]["R@1"], "HFR": result["flips"]["HFR"]})

    random_summary = {}
    for name in ("R@1", "HFR"):
        values = [trial[name] for trial in trials]
        random_summary[name] = {"mean": mean(values), "std": pstdev(values)}

    n_queries = len(base_ranks)
    total_calls = sum(cost * count for cost, count in budget_by_cost.items())
    report = {
        "split": args.split,
        "routes": str(routes_path),
        "shared_deep_results": str(deep_path),
        "shared_budget": {
            "queries": n_queries,
            "route_dist": {"accept": n_queries - len(router_deep), "deep": len(router_deep)},
            "deep_query_count_by_vlm_calls": dict(sorted(budget_by_cost.items())),
            "vlm_calls": total_calls,
            "vlm_calls_per_query": total_calls / n_queries,
        },
        "router_replay": router_metrics,
        "margin_control": margin_metrics,
        "random_control": {"summary": random_summary, "trials": trials},
    }
    out_path = args.output or Path(work_path(cfg, "routing_control", f"{args.split}_matched_budget{tag}.json"))
    write_json(out_path, report)
    print(f"Shared budget: {len(router_deep)}/{n_queries} deep queries, {total_calls / n_queries:.2f} VLM calls/query")
    print(f"Router replay: R@1={router_metrics['new']['R@1']:.2f}% HFR={router_metrics['flips']['HFR']:.2f}%")
    print(f"Score margin: R@1={margin_metrics['new']['R@1']:.2f}% HFR={margin_metrics['flips']['HFR']:.2f}%")
    print(f"Random control ({args.trials} trials): "
          f"R@1={random_summary['R@1']['mean']:.2f}% ± {random_summary['R@1']['std']:.2f}, "
          f"HFR={random_summary['HFR']['mean']:.2f}% ± {random_summary['HFR']['std']:.2f}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
