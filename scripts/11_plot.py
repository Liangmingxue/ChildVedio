"""Paper figures: accuracy-cost Pareto plot and router reliability diagram."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvr.calibration import ece_score
from cvr.utils import add_common_args, load_config, read_json, work_path


def pareto(cfg, pattern):
    files = sorted(glob.glob(pattern))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for f in files:
        m = read_json(f)
        label = m["mode"] + (f"-{m['tag']}" if m.get("tag") else "")
        ax.scatter(m["vlm_calls_per_query"], m["new"]["R@1"], s=40)
        ax.annotate(label, (m["vlm_calls_per_query"], m["new"]["R@1"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("VLM calls per query")
    ax.set_ylabel("R@1 (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = work_path(cfg, "figures", "pareto.pdf")
    fig.savefig(out)
    print("->", out)


def reliability(cfg, npz_path):
    d = np.load(npz_path)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    for key, name in (("action_raw", "uncalibrated"), ("action_cal", "temperature-scaled")):
        e, bins = ece_score(d[key], d["action_y"])
        ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], marker="o", ms=3, label=f"{name} (ECE={e:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = work_path(cfg, "figures", "reliability.pdf")
    fig.savefig(out)
    print("->", out)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("what", choices=["pareto", "reliability"])
    ap.add_argument("--pattern", default=None, help="glob of *_metrics.json for pareto")
    ap.add_argument("--npz", default=None, help="router probs npz for reliability")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.what == "pareto":
        pareto(cfg, args.pattern or os.path.join(cfg.paths.work_dir, "pipeline", "*_metrics.json"))
    else:
        reliability(cfg, args.npz or work_path(cfg, "router", "dev_probs.npz"))


if __name__ == "__main__":
    main()
