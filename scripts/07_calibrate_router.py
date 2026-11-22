"""Fit router temperatures on dev and choose the CRC accept threshold on calib.

Loss for CRC (B = 1): accepting the first stage is a 'missed fix' when its top-1 is wrong
while the light or deep path would have put the GT at rank 1.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cvr.calibration import apply_temperature, brier_score, crc_threshold, ece_score, fit_temperature, nll_score
from cvr.counterfactual import FRAGILE_TYPES
from cvr.router import ACTIONS, JevFormatter, load_router, predict_router
from cvr.utils import add_common_args, load_config, read_jsonl, work_path, write_json


def labels(rows):
    a = np.array([ACTIONS.index(r["label_action"]) for r in rows])
    f = np.array([FRAGILE_TYPES.index(r["label_fragile"]) if r["label_fragile"] in FRAGILE_TYPES else 6 for r in rows])
    return a, f


def report(logits, y, T):
    p0, p1 = apply_temperature(logits, 1.0), apply_temperature(logits, T)
    return {
        "acc": float((logits.argmax(1) == y).mean()),
        "nll_before": nll_score(logits, y), "nll_after": nll_score(logits, y, T),
        "ece_before": ece_score(p0, y)[0], "ece_after": ece_score(p1, y)[0],
        "brier_before": brier_score(p0, y), "brier_after": brier_score(p1, y),
    }


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--adapter", default=None, help="default: outputs/router/best")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    rc = cfg.router
    adapter = args.adapter or work_path(cfg, "router", "best")
    rows = read_jsonl(work_path(cfg, "router_data", "router_data.jsonl"))
    calib = [r for r in rows if r["split"] == "calib"]
    dev = [r for r in rows if r["split"] == "dev"]
    if not calib:
        ap.error("router data must contain a calibration split")

    tok, model = load_router(rc.base_model, adapter=adapter, device=rc.device)
    fmt = JevFormatter(tok, max_len=rc.max_len)
    st = (cfg.retriever.stat_temp, cfg.retriever.stat_k)
    pc = predict_router(model, fmt, calib, rc.device, rc.eval_batch_size, *st)
    pd = predict_router(model, fmt, dev, rc.device, rc.eval_batch_size, *st) if dev else None

    ya, yf = labels(calib)
    Ta = Tf = 1.0
    if pd is not None:
        da, df = labels(dev)
        Ta, _ = fit_temperature(pd["action"], da)
        Tf, _ = fit_temperature(pd["fragile"], df)
    p_action = apply_temperature(pc["action"], Ta)

    loss = np.array([float(r["ranks"]["accept"] != 1 and (r["ranks"]["light"] == 1 or r["ranks"]["deep"] == 1))
                     for r in calib])
    tau = crc_threshold(p_action[:, 0], loss, rc.alpha)
    accept = p_action[:, 0] >= tau
    out = {
        "adapter": adapter, "T_action": Ta, "T_fragile": Tf, "tau_accept": tau, "alpha": rc.alpha,
        "temperature_fit_split": "dev" if pd is not None else None,
        "temperature_fit_n": len(dev), "crc_loss": "missed_fix",
        "calib_n": len(calib), "calib_accept_rate": float(accept.mean()),
        "calib_missed_fix_rate": float((loss * accept).mean()),
        "calib_action": report(pc["action"], ya, Ta), "calib_fragile": report(pc["fragile"], yf, Tf),
    }
    if pd is not None:
        out["dev_action"] = report(pd["action"], da, Ta)
        out["dev_fragile"] = report(pd["fragile"], df, Tf)
        np.savez(work_path(cfg, "router", "dev_probs.npz"),
                 action_raw=apply_temperature(pd["action"], 1.0), action_cal=apply_temperature(pd["action"], Ta),
                 action_y=da)
    np.savez(work_path(cfg, "router", "calib_probs.npz"),
             action_raw=apply_temperature(pc["action"], 1.0), action_cal=p_action, action_y=ya)
    write_json(work_path(cfg, "router", "calib.json"), out)
    print(f"T_action={Ta:.3f} T_fragile={Tf:.3f} tau_accept={tau:.4f} accept_rate={accept.mean():.3f}")


if __name__ == "__main__":
    main()
