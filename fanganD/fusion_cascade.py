"""Learned listwise fusion + light->deep cascade under the margin baseline's exact VLM budget (offline replay)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import (BUDGET, ListwiseFusion, beta_ranks, build, cached_ranks, coarse_margin, dump,  # noqa: E402
                 fill_budget, load_run, metrics, paired_bootstrap)


def flips(ranks, base):
    hit, orig = ranks <= 1, base <= 1
    return {"helpful": int((hit & ~orig).sum()), "harmful": int((~hit & orig).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="fanganC/runs/aligned-full-20261122")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results/fusion_cascade.json"))
    args = ap.parse_args()
    t0 = time.time()
    rows, test, cost, ma = load_run(args.run_dir)
    train = [x for x in rows if x.get("split") == "train"]  # dev/calib held out, never used here
    cached = cached_ranks(test)
    base = cached["accept"]
    margin_ranks = np.where(ma == "deep", cached["deep"], base)
    margin_calls = int(cost[ma == "deep", 2].sum())
    out = {"train_rows": len(train), "test_rows": len(test), "budget": BUDGET,
           "cached": {k: metrics(v) for k, v in cached.items()},
           "margin_baseline": {**metrics(margin_ranks), **flips(margin_ranks, base), "calls": margin_calls},
           "beta_baseline": {}, "fusion_all_queries": {}, "budget_matched": {}, "cascade_curve": {}, "seeds": {}}

    for key in ("light_scores", "deep_scores"):
        grid = np.round(np.linspace(0, 1, 21), 2)
        best = max(grid, key=lambda b: (beta_ranks(train, key, b) <= 1).mean())
        out["beta_baseline"][key] = {"beta_tuned_on_train": float(best),
                                     "test_tuned": metrics(beta_ranks(test, key, best)),
                                     "test_beta_0.7": metrics(beta_ranks(test, key, .7))}

    configs = {"light": (True, False), "deep": (False, True), "light+deep": (True, True)}
    test_data = {name: build(test, *cfg) for name, cfg in configs.items()}
    train_data = {name: build(train, *cfg) for name, cfg in configs.items()}
    marg = coarse_margin(test)
    margin_hit = margin_ranks <= 1

    for seed in args.seeds:
        models = {name: ListwiseFusion(args.hidden, args.epochs, seed=seed).fit(train_data[name]) for name in configs}
        fused = {name: models[name].ranks(test_data[name], test) for name in configs}
        if seed == args.seeds[0]:
            for name in configs:
                out["fusion_all_queries"][name] = metrics(fused[name])
            linear = {name: ListwiseFusion(0, args.epochs, seed=seed).fit(train_data[name]) for name in configs}
            for name in configs:
                out["fusion_all_queries"][name + " (linear)"] = metrics(linear[name].ranks(test_data[name], test))
        light_conf = models["light"].top_gap(test_data["light"])
        light_calls = int(cost[:, 1].sum())
        policies = {}
        r = np.where(ma == "deep", fused["deep"], base)
        policies["margin_route+deep_fusion"] = (r, margin_calls)
        rng = np.random.default_rng(seed)
        for name, priority in (("cascade_by_light_conf", light_conf), ("cascade_by_coarse_margin", marg),
                               ("cascade_random", rng.random(len(test)))):
            sel, spent = fill_budget(priority, cost[:, 2], light_calls)
            policies[name] = (np.where(sel, fused["light+deep"], fused["light"]), spent, int(sel.sum()))
        out["seeds"][seed] = {}
        for name, value in policies.items():
            ranks, spent = value[0], value[1]
            record = {**metrics(ranks), **flips(ranks, base), "calls": spent,
                      "vs_margin": paired_bootstrap(ranks <= 1, margin_hit, seed=seed)}
            if len(value) > 2:
                record["deep_queries"] = value[2]
            out["seeds"][seed][name] = record
        if seed == args.seeds[0]:
            out["budget_matched"] = out["seeds"][seed]
            for budget in (5000, 10000, 15000, 20000, 25000, 27824, 30000, 37000):
                if budget < light_calls:
                    continue
                sel, spent = fill_budget(light_conf, cost[:, 2], light_calls, budget)
                ranks = np.where(sel, fused["light+deep"], fused["light"])
                out["cascade_curve"][budget] = {**metrics(ranks), "calls": spent, "deep_queries": int(sel.sum())}
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = time.time() - t0
    dump(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
