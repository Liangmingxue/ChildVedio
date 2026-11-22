"""Route a retrieval split with the calibrated router or a fixed score-margin budget."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from cvr.calibration import apply_temperature
from cvr.counterfactual import FRAGILE_TYPES
from cvr.retriever import score_stats
from cvr.router import ACTIONS, JevFormatter, decide_action, load_router, predict_router
from cvr.utils import add_common_args, load_config, read_json, read_jsonl, work_path, write_jsonl


def calibrated_routes(cfg, rows, tau):
    rc = cfg.router
    cal = read_json(work_path(cfg, "router", "calib.json"))
    tau = cal["tau_accept"] if tau is None else tau
    tok, model = load_router(rc.base_model, adapter=cal["adapter"], device=rc.device)
    fmt = JevFormatter(tok, max_len=rc.max_len)
    pred = predict_router(model, fmt, rows, rc.device, rc.eval_batch_size, cfg.retriever.stat_temp, cfg.retriever.stat_k)
    pa = apply_temperature(pred["action"], cal["T_action"])
    pf = apply_temperature(pred["fragile"], cal["T_fragile"])
    out = []
    for r, a, f in zip(rows, pa, pf):
        out.append({
            "qid": r["qid"], "p_action": dict(zip(ACTIONS, map(float, a))), "action": decide_action(a, tau),
            "fragile": FRAGILE_TYPES[int(f.argmax())], "p_fragile": dict(zip(FRAGILE_TYPES, map(float, f))),
        })
    return out


def margin_routes(rows, deep_count):
    ordered = sorted(rows, key=lambda row: score_stats(row["scores"], k=2)["margin"])
    deep = {row["qid"] for row in ordered[:deep_count]}
    return [
        {"qid": row["qid"], "action": "deep" if row["qid"] in deep else "accept", "fragile": "auto"}
        for row in rows
    ]


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="test")
    policy = ap.add_mutually_exclusive_group()
    policy.add_argument("--tau", type=float, help="override CRC threshold (for sweeps)")
    policy.add_argument("--margin-deep-count", type=int, help="send this many smallest-margin queries to deep")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    rows = read_jsonl(work_path(cfg, "retrieval", f"{args.split}.jsonl"))

    if args.margin_deep_count is None:
        out = calibrated_routes(cfg, rows, args.tau)
        suffix = "" if args.tau is None else f"_tau{args.tau:g}"
    else:
        if not 0 <= args.margin_deep_count <= len(rows):
            ap.error(f"--margin-deep-count must be between 0 and {len(rows)}")
        out = margin_routes(rows, args.margin_deep_count)
        suffix = f"_margin_{args.margin_deep_count}"
    path = work_path(cfg, "routes", f"{args.split}{suffix}.jsonl")
    write_jsonl(path, out)
    print("routes:", Counter(o["action"] for o in out), "->", path)


if __name__ == "__main__":
    main()
