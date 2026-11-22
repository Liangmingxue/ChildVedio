"""RL-train the counterfactual generator with GRPO + DAPO dynamic sampling.

GPU0: policy (LoRA) + NLI filter.  GPU1: VLM server (scripts/serve_vlm.sh). No text server needed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time

from cvr.factory import make_nli, make_verifier
from cvr.grpo import CFGRPOTrainer, build_grpo_rows
from cvr.utils import add_common_args, load_config, read_jsonl, set_seed, work_path


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    g = cfg.grpo
    set_seed(cfg.seed)
    rows = build_grpo_rows(read_jsonl(work_path(cfg, "router_data", "router_data.jsonl")), g.n_neg, g.max_gt_rank)
    print(f"GRPO training queries: {len(rows)}")
    out_dir = work_path(cfg, "cf_grpo", "x")[:-2]
    trainer = CFGRPOTrainer(cfg, make_verifier(cfg), make_nli(cfg, device=g.device), out_dir)

    rng = random.Random(cfg.seed)
    order, ptr = list(range(len(rows))), len(rows)
    for step in range(1, g.steps + 1):
        if ptr + g.batch_queries > len(order):
            rng.shuffle(order)
            ptr = 0
        batch = [rows[i] for i in order[ptr:ptr + g.batch_queries]]
        ptr += g.batch_queries
        t0 = time.time()
        stats = trainer.step(batch, step)
        stats["sec"] = round(time.time() - t0, 2)
        trainer.log(stats)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()})
        if step % g.save_every == 0:
            trainer.save(f"step_{step}")
    print("final adapter ->", trainer.save("final"))


if __name__ == "__main__":
    main()
