"""Train the Jev-style router (LoRA, restricted-softmax supervision). Single GPU, no servers needed."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvr.router import train_router
from cvr.utils import add_common_args, load_config, read_jsonl, set_seed, work_path


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    rows = read_jsonl(work_path(cfg, "router_data", "router_data.jsonl"))
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    print(f"train={len(train)} dev={len(dev)}")
    best = train_router(cfg, train, dev, work_path(cfg, "router", "x")[:-2])
    print("best adapter ->", best)


if __name__ == "__main__":
    main()
