"""Train and evaluate the fanganE JEV-style distillation model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from cvr.router import load_router
from cvr.utils import set_seed
from fanganC.common import atomic_json, digest, resolve
from fanganD.lib import BUDGET, metrics, paired_bootstrap, parse

from .data import load_bundle
from .losses import compute_loss
from .model import JevCandidateRanker, OUTCOME_NAMES
from .policy import allocate_budget, rank_from_path, softmax_np


ROOT = Path(__file__).resolve().parents[1]


def evidence_text(row: dict) -> str:
    coarse = " ".join(f"{value:.2f}" for value in parse(row["scores"]))
    light = " ".join(f"{value:.3f}" for value in parse(row["light_scores"]))
    return (
        f"Query: {row['text']}\n"
        f"Candidates in retrieval order: {coarse}\n"
        f"Yes probabilities for the first five: {light}\n"
        "Decide whether the current light result should be upgraded and which candidate to trust."
    )


def unwrap_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def hidden_batch(model, tokenizer, token_rows, ids: np.ndarray, device: torch.device) -> torch.Tensor:
    batch = tokenizer.pad(
        {"input_ids": [token_rows[int(index)] for index in ids]},
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    body = unwrap_model(model).model
    outputs = body(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    positions = torch.arange(input_ids.shape[1], device=device).expand_as(attention_mask)
    last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    row_ids = torch.arange(len(ids), device=device)
    return outputs.last_hidden_state[row_ids, last_positions].float()


def precompute_hidden(model, tokenizer, token_rows, count: int, batch_size: int, device: torch.device):
    model.eval()
    cache = None
    with torch.no_grad():
        for start in range(0, count, batch_size):
            ids = np.arange(start, min(start + batch_size, count), dtype=np.int64)
            hidden = hidden_batch(model, tokenizer, token_rows, ids, device).cpu().numpy()
            if cache is None:
                cache = np.empty((count, hidden.shape[-1]), dtype=np.float32)
            cache[start : start + len(ids)] = hidden
    if cache is None:
        raise ValueError("Cannot precompute hidden states for an empty dataset")
    return cache


def capture_trainable(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable(model, state: dict[str, torch.Tensor]) -> None:
    if not state:
        return
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def fit_route_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    calib_ids: np.ndarray,
    grid: list[float],
) -> tuple[float, float]:
    if len(calib_ids) == 0:
        raise ValueError("Calibration gallery is empty")
    best_temperature, best_loss = None, float("inf")
    target = targets[calib_ids].astype(np.float64)
    for temperature in grid:
        probs = softmax_np(logits[calib_ids], float(temperature))
        loss = float(-(target * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1).mean())
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), loss
    return best_temperature, best_loss


def summarize_split(
    ranks: np.ndarray,
    original: np.ndarray,
    chosen: np.ndarray,
    costs: np.ndarray,
    mask: np.ndarray,
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Cannot summarize an empty split")
    new_hit = ranks[mask] == 1
    old_hit = original[mask] == 1
    calls = int(costs[mask, 1].sum() + costs[mask & chosen, 2].sum())
    split_budget = int(round(BUDGET * int(mask.sum()) / 1000))
    if calls > split_budget:
        raise RuntimeError(f"VLM budget exceeded: {calls} > {split_budget}")
    return {
        **metrics(ranks[mask]),
        "queries": int(mask.sum()),
        "deep_queries": int(chosen[mask].sum()),
        "vlm_calls": calls,
        "budget": split_budget,
        "fixes_vs_original": int((~old_hit & new_hit).sum()),
        "harms_vs_original": int((old_hit & ~new_hit).sum()),
        "route_counts": {
            "light": int((~chosen[mask]).sum()),
            "deep": int(chosen[mask].sum()),
        },
    }


def _to_device(bundle: dict, device: torch.device) -> dict:
    return {
        "numeric": torch.as_tensor(bundle["numeric"], dtype=torch.float32, device=device),
        "initial": torch.as_tensor(bundle["initial_scores"], dtype=torch.float32, device=device),
        "teacher": torch.as_tensor(bundle["teacher"], dtype=torch.float32, device=device),
        "teacher_mask": torch.as_tensor(bundle["teacher_mask"], dtype=torch.bool, device=device),
        "route_soft": torch.as_tensor(bundle["route_soft"], dtype=torch.float32, device=device),
        "route_hard": torch.as_tensor(bundle["route_hard"], dtype=torch.long, device=device),
        "gt": torch.as_tensor(bundle["gt_index"], dtype=torch.long, device=device),
        "candidate_ids": torch.as_tensor(bundle["candidate_ids"], dtype=torch.long, device=device),
        "videos": torch.as_tensor(bundle["video_features"], dtype=torch.float32, device=device),
    }


def _batch_forward(
    ids: np.ndarray,
    model,
    tokenizer,
    token_rows,
    hidden_cache,
    head,
    tensors,
    device,
):
    index = torch.as_tensor(ids, dtype=torch.long, device=device)
    if hidden_cache is None:
        hidden = hidden_batch(model, tokenizer, token_rows, ids, device)
    else:
        hidden = torch.as_tensor(hidden_cache[ids], dtype=torch.float32, device=device)
    candidate_videos = tensors["videos"][tensors["candidate_ids"][index]]
    return head(
        hidden,
        tensors["numeric"][index],
        tensors["initial"][index],
        candidate_videos,
    )


def _evaluate_loss(
    ids: np.ndarray,
    model,
    tokenizer,
    token_rows,
    hidden_cache,
    head,
    tensors,
    device,
    config,
    class_weights,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    head.eval()
    collected_scores, collected_outcomes = [], []
    totals = {"total": 0.0, "rank_kl": 0.0, "rank_ce": 0.0, "route_kl": 0.0, "route_ce": 0.0}
    seen = 0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            index = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
            scores, outcomes = _batch_forward(
                batch_ids, model, tokenizer, token_rows, hidden_cache, head, tensors, device
            )
            losses = compute_loss(
                scores,
                outcomes,
                tensors["gt"][index],
                tensors["teacher"][index],
                tensors["teacher_mask"][index],
                tensors["route_soft"][index],
                tensors["route_hard"][index],
                config,
                class_weights,
            )
            size = len(batch_ids)
            for key in totals:
                totals[key] += float(losses[key].detach()) * size
            seen += size
            collected_scores.append(scores.float().cpu().numpy())
            collected_outcomes.append(outcomes.float().cpu().numpy())
    if seen == 0:
        raise ValueError("Evaluation split is empty")
    averages = {key: value / seen for key, value in totals.items()}
    return averages, np.concatenate(collected_scores), np.concatenate(collected_outcomes)


def _predict_all(
    count,
    batch_size,
    model,
    tokenizer,
    token_rows,
    hidden_cache,
    head,
    tensors,
    device,
):
    model.eval()
    head.eval()
    score_parts, outcome_parts = [], []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            ids = np.arange(start, min(start + batch_size, count), dtype=np.int64)
            scores, outcomes = _batch_forward(
                ids, model, tokenizer, token_rows, hidden_cache, head, tensors, device
            )
            score_parts.append(scores.float().cpu().numpy())
            outcome_parts.append(outcomes.float().cpu().numpy())
    return np.concatenate(score_parts), np.concatenate(outcome_parts)


def run(config: dict, out_dir: Path) -> dict:
    mode = config["mode"]
    if mode not in ("frozen", "lora"):
        raise ValueError("mode must be 'frozen' or 'lora'")

    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This training recipe requires BF16 support on the selected GPU")

    set_seed(int(config["seed"]))
    started = time.time()
    log_path = out_dir / "train.log"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    log(f"Loading source snapshot: {config['source']}")
    bundle = load_bundle(config["source"], float(config["teacher_temperature"]))
    tensors = _to_device(bundle, device)
    rows = bundle["rows"]
    count = len(rows)
    train_ids = bundle["train_ids"]
    dev_ids = bundle["dev_ids"]
    calib_ids = bundle["calib_ids"]
    holdout_ids = bundle["holdout_ids"]

    log(
        "Gallery split sizes: "
        f"train={len(train_ids)} (galleries 0-3), "
        f"decision_fit=2000 (galleries 4-5, untouched), "
        f"dev={len(dev_ids)}, calib={len(calib_ids)}, holdout={len(holdout_ids)}"
    )

    log(f"Loading Qwen backbone: {bundle['base_model']} ({mode})")
    tokenizer, backbone = load_router(
        bundle["base_model"],
        device=str(device),
        train=(mode == "lora"),
        lora_r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
    )
    backbone.config.use_cache = False
    token_rows = tokenizer(
        [evidence_text(row) for row in rows],
        truncation=True,
        max_length=int(config["max_len"]),
    )["input_ids"]

    hidden_cache = None
    if mode == "frozen":
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()
        log("Precomputing frozen Qwen hidden states")
        hidden_cache = precompute_hidden(
            backbone,
            tokenizer,
            token_rows,
            count,
            int(config["eval_batch_size"]),
            device,
        )
        hidden_dim = int(hidden_cache.shape[-1])
    else:
        hidden_dim = int(unwrap_model(backbone).config.hidden_size)
        log("LoRA enabled with gradient checkpointing; micro-batches are kept small")

    head = JevCandidateRanker(
        hidden_dim=hidden_dim,
        numeric_dim=int(bundle["numeric"].shape[-1]),
        clip_dim=int(bundle["video_features"].shape[-1]),
        pointer_dim=int(config["pointer_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)

    route_counts = np.bincount(bundle["route_hard"][train_ids], minlength=3).astype(np.float64)
    class_weights_np = np.sqrt(route_counts.sum() / np.maximum(route_counts, 1.0))
    class_weights_np /= class_weights_np.mean()
    class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)

    head_params = list(head.parameters())
    lora_params = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    if mode == "lora" and not lora_params:
        raise RuntimeError("LoRA mode was requested, but no trainable adapter parameters were found")
    optimizer_groups = [{"params": head_params, "lr": float(config["head_lr"])}]
    if mode == "lora":
        optimizer_groups.append({"params": lora_params, "lr": float(config["lora_lr"])})
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(config["weight_decay"]),
    )
    trainable_params = head_params + lora_params

    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_head = None
    best_adapter = {}
    best_dev_summary = None
    stale_epochs = 0
    rng = np.random.default_rng(int(config["seed"]))

    for epoch in range(1, int(config["epochs"]) + 1):
        head.train()
        backbone.train(mode == "lora")
        order = rng.permutation(train_ids)
        micro_batches = [
            order[start : start + int(config["micro_batch"])]
            for start in range(0, len(order), int(config["micro_batch"]))
        ]
        epoch_total = 0.0
        seen = 0
        optimizer_steps = 0

        for window_start in range(0, len(micro_batches), int(config["grad_accum"])):
            window = micro_batches[window_start : window_start + int(config["grad_accum"])]
            optimizer.zero_grad(set_to_none=True)
            window_total = 0.0
            for batch_ids in window:
                index = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
                scores, outcomes = _batch_forward(
                    batch_ids, backbone, tokenizer, token_rows, hidden_cache, head, tensors, device
                )
                losses = compute_loss(
                    scores,
                    outcomes,
                    tensors["gt"][index],
                    tensors["teacher"][index],
                    tensors["teacher_mask"][index],
                    tensors["route_soft"][index],
                    tensors["route_hard"][index],
                    config,
                    class_weights,
                )
                (losses["total"] / len(window)).backward()
                window_total += float(losses["total"].detach())
                seen += len(batch_ids)

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                float(config["max_grad_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer_steps += 1
            epoch_total += window_total

        dev_losses, dev_scores, dev_outcomes = _evaluate_loss(
            dev_ids,
            backbone,
            tokenizer,
            token_rows,
            hidden_cache,
            head,
            tensors,
            device,
            config,
            class_weights,
            int(config["eval_batch_size"]),
        )
        dev_probs = softmax_np(dev_outcomes)
        dev_priority = dev_probs[:, 2] - dev_probs[:, 0]
        dev_chosen = allocate_budget(
            dev_priority,
            bundle["costs"][dev_ids],
            bundle["gallery"][dev_ids],
        )
        dev_ranks = rank_from_path(
            dev_scores,
            dev_chosen,
            bundle["gt_index"][dev_ids],
            bundle["original_rank"][dev_ids],
        )
        dev_r1 = float(metrics(dev_ranks)["R@1"])
        record = {
            "epoch": epoch,
            "train_loss": epoch_total / max(optimizer_steps, 1),
            "train_queries": int(seen),
            "optimizer_steps": int(optimizer_steps),
            "dev": dev_losses,
            "dev_cascade_R1": dev_r1,
            "dev_deep_queries": int(dev_chosen.sum()),
        }
        history.append(record)
        atomic_json(out_dir / "history.json", history)
        log(json.dumps(record, ensure_ascii=False))

        if dev_losses["total"] < best_loss - float(config["min_delta"]):
            best_loss = dev_losses["total"]
            best_epoch = epoch
            best_head = copy.deepcopy(head.state_dict())
            best_adapter = capture_trainable(backbone) if mode == "lora" else {}
            best_dev_summary = record
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                log(f"Early stopping after epoch {epoch}; best epoch={best_epoch}")
                break

    if best_head is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    head.load_state_dict(best_head)
    if mode == "lora":
        restore_trainable(backbone, best_adapter)
        backbone.save_pretrained(str(out_dir / "best_adapter"))
    torch.save(
        {
            "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "mode": mode,
            "best_epoch": best_epoch,
            "best_dev_loss": best_loss,
        },
        out_dir / "best_head.pt",
    )
    atomic_json(out_dir / "history.json", history)

    log("Running offline predictions for calibration and the untouched holdout gallery")
    all_scores, all_outcomes = _predict_all(
        count,
        int(config["eval_batch_size"]),
        backbone,
        tokenizer,
        token_rows,
        hidden_cache,
        head,
        tensors,
        device,
    )

    route_temperature, calibration_loss = fit_route_temperature(
        all_outcomes,
        bundle["route_soft"],
        calib_ids,
        [float(value) for value in config["route_temperature_grid"]],
    )
    route_probs = softmax_np(all_outcomes, route_temperature)
    priority = route_probs[:, 2] - route_probs[:, 0]
    learned_route = allocate_budget(priority, bundle["costs"], bundle["gallery"])
    learned_ranks = rank_from_path(
        all_scores,
        learned_route,
        bundle["gt_index"],
        bundle["original_rank"],
    )
    fixed_ranks = rank_from_path(
        bundle["initial_scores"],
        bundle["fixed_route"],
        bundle["gt_index"],
        bundle["original_rank"],
    )

    holdout_mask = bundle["gallery"] == 8
    learned_summary = summarize_split(
        learned_ranks,
        bundle["original_rank"],
        learned_route,
        bundle["costs"],
        holdout_mask,
    )
    fixed_summary = summarize_split(
        fixed_ranks,
        bundle["original_rank"],
        bundle["fixed_route"],
        bundle["costs"],
        holdout_mask,
    )
    paired = paired_bootstrap(
        learned_ranks[holdout_mask] == 1,
        fixed_ranks[holdout_mask] == 1,
    )

    calib_probs = route_probs[calib_ids]
    calib_targets = bundle["route_soft"][calib_ids]
    calibration_report = {
        "gallery": 7,
        "temperature": route_temperature,
        "soft_target_cross_entropy": calibration_loss,
        "soft_target_brier": float(((calib_probs - calib_targets) ** 2).sum(axis=1).mean()),
        "hard_outcome_accuracy": float(
            (calib_probs.argmax(axis=1) == bundle["route_hard"][calib_ids]).mean()
        ),
        "outcome_names": list(OUTCOME_NAMES),
    }

    results = {
        "method": "fanganE",
        "mode": mode,
        "selection": {
            "criterion": "minimum development composite distillation loss",
            "best_epoch": best_epoch,
            "best_dev_loss": best_loss,
            "best_dev": best_dev_summary,
        },
        "calibration": calibration_report,
        "holdout_gallery_8": {
            "evaluation_protocol": "one-time offline replay; no new VLM calls",
            "fanganE": learned_summary,
            "D0_light_confidence_cascade": fixed_summary,
            "paired_bootstrap_vs_D0": paired,
        },
        "budget": {
            "calls": learned_summary["vlm_calls"],
            "maximum": learned_summary["budget"],
            "within_budget": learned_summary["vlm_calls"] <= learned_summary["budget"],
        },
    }
    atomic_json(out_dir / "results.json", results)
    np.savez_compressed(
        out_dir / "predictions.npz",
        gallery=bundle["gallery"],
        gt_index=bundle["gt_index"],
        original_rank=bundle["original_rank"],
        rank_scores=all_scores,
        outcome_logits=all_outcomes,
        route_probabilities=route_probs,
        learned_route=learned_route,
        fixed_route=bundle["fixed_route"],
        learned_ranks=learned_ranks,
        fixed_ranks=fixed_ranks,
    )
    np.savez_compressed(out_dir / "d0_models.npz", **bundle["d0_models"])

    code_paths = (
        "fanganE/model.py",
        "fanganE/losses.py",
        "fanganE/data.py",
        "fanganE/policy.py",
        "fanganE/train.py",
        "fanganD/lib.py",
        "fanganC/vendor/pointer.py",
        "cvr/router.py",
    )
    code_hashes = {
        relative: digest(resolve(relative))
        for relative in code_paths
        if resolve(relative).is_file()
    }
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(bundle["source"]),
        "source_snapshot_manifest_sha256": digest(bundle["source"] / "snapshot" / "manifest.json"),
        "clip_archive_sha256": bundle["clip_manifest"]["sha256"],
        "source_split": {
            "train_galleries": [0, 1, 2, 3],
            "decision_fit_reserved": [4, 5],
            "development": [6],
            "calibration": [7],
            "holdout": [8],
        },
        "test_set_used": False,
        "new_vlm_calls": 0,
        "base_model": bundle["base_model"],
        "mode": mode,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "config": config,
        "code_sha256": code_hashes,
        "command": "python -m fanganE --config fanganE/config.yaml --mode "
                   + mode + " --out " + str(out_dir),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda" else None
        ),
    }
    atomic_json(out_dir / "protocol.json", protocol)
    log(
        f"Complete: holdout R@1={learned_summary['R@1']:.3f}, "
        f"D0 R@1={fixed_summary['R@1']:.3f}, "
        f"calls={learned_summary['vlm_calls']}/{learned_summary['budget']}"
    )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="fanganE/config.yaml")
    parser.add_argument("--source")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("frozen", "lora"))
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-len", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    overrides = {
        "source": args.source,
        "mode": args.mode,
        "device": args.device,
        "epochs": args.epochs,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "max_len": args.max_len,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    runs_root = (ROOT / "fanganE" / "runs").resolve()
    if not out_dir.is_relative_to(runs_root):
        raise ValueError(f"Run output must be inside {runs_root}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=False)

    atomic_json(out_dir / "config.json", config)
    try:
        run(config, out_dir)
    except BaseException as error:
        atomic_json(
            out_dir / "failure.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
