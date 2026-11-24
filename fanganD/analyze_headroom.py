"""Diagnose why routing cannot beat the margin baseline at the 87.6% deep operating point."""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import auc, cached_ranks, coarse_margin, dump, load_run, parse, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="fanganC/runs/aligned-full-20261122")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results/headroom.json"))
    args = ap.parse_args()
    _, test, cost, ma = load_run(args.run_dir)
    ranks = cached_ranks(test)
    acc, lig, dep = (ranks[k] == 1 for k in ("accept", "light", "deep"))
    marg = coarse_margin(test)
    patterns = collections.Counter(f"accept={a},light={l},deep={d}" for a, l, d in zip(acc, lig, dep))
    gain = dep.astype(int) - acc.astype(int)
    out = {
        "correct_patterns": dict(patterns),
        "single_path_R@1": {"accept": acc.mean() * 100, "light": lig.mean() * 100, "deep": dep.mean() * 100},
        "margin_accept_set": {"n": int((ma == "accept").sum()), "accept_correct": int(acc[ma == "accept"].sum()),
                              "deep_would_fix": int((dep & ~acc)[ma == "accept"].sum()),
                              "deep_would_harm": int((acc & ~dep)[ma == "accept"].sum())},
        "margin_deep_set": {"n": int((ma == "deep").sum()), "helpful": int((dep & ~acc)[ma == "deep"].sum()),
                            "harmful": int((acc & ~dep)[ma == "deep"].sum())},
        "oracle_R@1": {"accept_or_deep": float(np.where(gain > 0, dep, acc).mean() * 100),
                       "any_of_three": float((acc | lig | dep).mean() * 100)},
        "light_right_deep_wrong": int((lig & ~dep).sum()), "deep_right_light_wrong": int((dep & ~lig).sum()),
        "auc": {"neg_margin_for_helpful": auc(-marg, dep & ~acc), "margin_for_harmful": auc(marg, acc & ~dep),
                "margin_for_harmful_in_deep_set": auc(marg[ma == "deep"], (acc & ~dep)[ma == "deep"])},
        "budget_curve_query_count": {},
    }
    run = Path(args.run_dir)
    scores = {}
    for variant in ("text_only", "candidate"):
        pred = {x["qid"]: x for x in read_jsonl(run / variant / "test_predictions.jsonl")}
        prob = np.array([pred[x["qid"]]["probabilities_light_deep_unchanged_fix_harm"] for x in test])
        scores[variant] = prob[:, 1, 1] - prob[:, 1, 2]
        out["auc"][f"{variant}_fix_for_helpful"] = auc(prob[:, 1, 1], dep & ~acc)
        out["auc"][f"{variant}_harm_for_harmful"] = auc(prob[:, 1, 2], acc & ~dep)
        out["auc"][f"{variant}_harm_for_harmful_in_deep_set"] = auc(prob[ma == "deep", 1, 2], (acc & ~dep)[ma == "deep"])
    for frac in (.1, .2, .3, .4, .5, .6, .7, .8, .876):
        k = int(round(frac * len(test)))
        row = {}
        for name, s in (("margin", -marg), ("text_only", scores["text_only"]),
                        ("candidate", scores["candidate"]), ("oracle", gain - 1e-6 * marg)):
            sel = np.argsort(-s, kind="stable")[:k]
            hit = acc.copy()
            hit[sel] = dep[sel]
            row[name] = float(hit.mean() * 100)
        out["budget_curve_query_count"][str(frac)] = row
    dump(args.out, out)
    print(f"wrote {args.out}")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
