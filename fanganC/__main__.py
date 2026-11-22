"""Run from repository root: python -m fanganC --help."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import traceback

from .common import atomic_json, read_json, require_local_output, snapshot_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["snapshot", "features", "train", "evaluate", "run", "smoke"])
    parser.add_argument("--config", default="fanganC/config.yaml")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", choices=["text_only", "candidate"])
    args = parser.parse_args()
    run_dir = require_local_output(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    phase = args.command

    def status(state, **extra):
        atomic_json(run_dir / "status.json", {"state": state, "phase": phase,
                    "utc": datetime.now(timezone.utc).isoformat(), **extra})

    try:
        if args.command == "smoke":
            from .smoke import smoke
            smoke(run_dir, args.device)
            return
        from .data import create_snapshot
        if args.command in ("snapshot", "run"):
            status("running")
            create_snapshot(args.config, run_dir)
        if args.command in ("train", "evaluate", "run"):
            config, _ = snapshot_config(run_dir)
            if args.variant and args.variant not in config.variants:
                raise ValueError(f"Variant {args.variant} is outside the frozen experiment configuration")
            variants = [args.variant] if args.variant else config.variants
            if args.command == "run" and args.variant:
                missing = [name for name in config.variants if name != args.variant
                           and not (run_dir / name / "training_complete.json").is_file()]
                if missing:
                    raise ValueError("Run all configured variants before evaluation, or use train --variant "
                                     "for a single variant in this experiment")
        if args.command in ("features", "run"):
            from .features import clip_features, text_features
            phase = "clip_features"
            status("running")
            clip_features(run_dir, args.device)
            phase = "text_features"
            status("running")
            text_features(run_dir, args.device)
        if args.command in ("train", "run"):
            from .training import train_variant
            for variant in variants:
                phase = f"train_{variant}"
                status("running")
                train_variant(run_dir, variant, args.device)
        if args.command in ("evaluate", "run"):
            from .evaluation import evaluate_variant
            results_file = run_dir / "results.json"
            results = read_json(results_file) if args.variant and results_file.is_file() else {}
            for variant in variants:
                phase = f"evaluate_{variant}"
                status("running")
                results[variant] = evaluate_variant(run_dir, variant, args.device)
            atomic_json(results_file, results)
        status("complete")
    except BaseException as error:
        status("failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
