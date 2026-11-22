"""Two-stage optimization and development-only checkpoint selection."""
from __future__ import annotations

import gc
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers import get_cosine_schedule_with_warmup

from .budget import structured_budget_loss
from .common import atomic_json, digest, fingerprint, legacy, read_json, resolve
from .features import FeatureStore, encode
from .model import CandidateRouter
from cvr.router import JevFormatter, field_states, load_router
from cvr.utils import set_seed


def new_head(store, variant):
    indices = store.train_indices
    config = store.config
    rerank = store.cfg.rerank
    return CandidateRouter(store.hidden.shape[1], store.query.shape[1], store.numeric[indices],
                           store.margins[indices], store.correct[indices, 0], variant,
                           config.hidden_dim, config.pointer_dim, config.dropout,
                           k_light=rerank.k_light, k_deep=rerank.k_deep,
                           temporal_candidates=config.get("temporal_candidates", False))


def optimizer_groups(head, backbone, config, phase):
    groups = []
    sources = [(head, config.warmup_lr if phase == "warmup" else config.head_lr)]
    if backbone is not None:
        sources.append((backbone, config.lora_lr))
    for module, rate in sources:
        for decay in (True, False):
            parameters = [p for p in module.parameters() if p.requires_grad and (p.ndim >= 2) == decay]
            if parameters:
                groups.append({"params": parameters, "lr": rate,
                               "weight_decay": config.weight_decay if decay else 0.})
    return groups


def replay(store, indices):
    return legacy().Replay.from_rows([store.rows[i] for i in indices], store.cfg)


@torch.inference_mode()
def predict(store, head, indices, device, backbone=None, formatter=None, encoded=None):
    head.eval()
    if backbone is not None:
        backbone.eval()
    include_candidates = head.variant == "candidate"
    predictions = []
    start_time = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
    for start in range(0, len(indices), store.config.eval_batch_size):
        ids = indices[start:start + store.config.eval_batch_size]
        if backbone is None:
            states = torch.as_tensor(np.array(store.hidden[ids]), device=device)
        else:
            _, states = field_states(backbone, formatter, [encoded[i] for i in ids], device)
            states = states[:, 0]
        logits = head(states, store.batch(ids, device, include_candidates=include_candidates))
        predictions.append(logits.cpu().numpy())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return np.concatenate(predictions), time.perf_counter() - start_time


def experiment_signature(store):
    folder = Path(__file__).parent
    return fingerprint({"snapshot": store.manifest["frozen_sha256"],
                        "features": {name: {"signature": value["signature"], "sha256": value["sha256"]}
                                     for name, value in store.feature_manifests.items()},
                        "code": {p.name: digest(p) for p in sorted(folder.glob("*.py"))},
                        "pointer": digest(folder / "vendor/pointer.py"),
                        "config": store.config})


def checkpoint_hashes(checkpoint, phase):
    hashes = {"head": digest(checkpoint / "head.pt")}
    if phase == "joint":
        adapter = checkpoint / "adapter"
        hashes["adapter"] = digest(adapter / "adapter_model.safetensors")
        hashes["adapter_config"] = digest(adapter / "adapter_config.json")
    return hashes


def publish_best(folder, head, backbone):
    best, pending, previous = (folder / name for name in ("best", "best.part", "best.previous"))
    if previous.exists() and not best.exists():
        previous.replace(best)
    for stale in (pending, previous):
        if stale.exists():
            shutil.rmtree(stale)
    pending.mkdir()
    torch.save(head.state_dict(), pending / "head.pt")
    if backbone is not None:
        backbone.save_pretrained(pending / "adapter")
    if best.exists():
        best.replace(previous)
    try:
        pending.replace(best)
    except OSError:
        if previous.exists():
            previous.replace(best)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def training_logits(head, store, ids, device, backbone, formatter, encoded, *, checkpointed):
    batch = store.batch(ids, device, include_candidates=head.variant == "candidate")
    if backbone is None:
        states = torch.as_tensor(np.array(store.hidden[ids]), device=device)
        return head(states, batch)
    texts = [encoded[i] for i in ids]

    def forward(inputs):
        _, states = field_states(backbone, formatter, texts, device)
        return head(states[:, 0], inputs)

    return activation_checkpoint(forward, batch, use_reentrant=False) if checkpointed else forward(batch)


def train_variant(run_dir, variant, device):
    store = FeatureStore(run_dir)
    config, cfg = store.config, store.cfg
    signature = experiment_signature(store)
    folder = Path(run_dir) / variant
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / "training_complete.json"
    if completed.exists():
        result = read_json(completed)
        if result["signature"] != signature:
            raise ValueError("Training inputs or code changed; start a new run")
        return result
    set_seed(config.seed)
    head = new_head(store, variant).to(device)
    dev = replay(store, store.dev_indices)
    baseline = dev.metrics(dev.margin_actions(config.deep_fraction))
    report = {"variant": variant, "signature": signature, "dev_margin": baseline,
              "snapshot_splits": store.manifest["splits"], "epochs": [], "validations": [], "selection": None}
    best = None
    select_joint_only = config.get("select_joint_only", False)
    if select_joint_only and config.joint_epochs < 1:
        raise ValueError("select_joint_only requires at least one joint epoch")
    patience = config.get("early_stopping_patience", 0)
    budget_weight = config.get("budget_loss_weight", 0.)
    if patience < 0 or not math.isfinite(budget_weight) or budget_weight < 0:
        raise ValueError("Invalid early-stopping patience or budget loss weight")
    if budget_weight:
        calls = np.zeros((len(store.rows), 3), dtype=np.int64)
        calls[store.train_indices] = replay(store, store.train_indices).calls
    stopped, stale_checks = False, 0
    dev_targets = torch.as_tensor(store.correct[store.dev_indices])
    training_start = time.perf_counter()
    backbone = formatter = encoded = None
    steps_per_epoch = math.ceil(len(store.train_indices) / config.effective_batch)
    validation_steps = {steps_per_epoch}
    if patience:
        validation_steps.add((steps_per_epoch + 1) // 2)
    for phase, epochs in (("warmup", config.warmup_epochs), ("joint", config.joint_epochs)):
        eligible_for_selection = phase == "joint" or not select_joint_only
        if phase == "joint":
            tokenizer, backbone = load_router(str(resolve(cfg.router.base_model)), device=device, train=True,
                                               lora_r=cfg.router.lora_r, lora_alpha=cfg.router.lora_alpha)
            backbone.config.use_cache = False
            formatter = JevFormatter(tokenizer, max_len=cfg.router.max_len)
            encoded = encode(store.rows, cfg, formatter)
        optimizer = torch.optim.AdamW(optimizer_groups(head, backbone, config, phase))
        steps = steps_per_epoch * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(config.warmup_ratio * steps)), steps)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        for epoch in range(epochs):
            head.train()
            if backbone is not None:
                backbone.train()
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            order = store.train_indices.copy()
            random.Random(config.seed + epoch + (100 if phase == "joint" else 0)).shuffle(order)
            total_loss = 0.
            total_budget_loss = 0.
            for step, start in enumerate(range(0, len(order), config.effective_batch), 1):
                window = order[start:start + config.effective_batch]
                optimizer.zero_grad(set_to_none=True)
                parts = []
                for offset in range(0, len(window), config.microbatch):
                    ids = window[offset:offset + config.microbatch]
                    logits = training_logits(head, store, ids, device, backbone, formatter, encoded,
                                             checkpointed=bool(budget_weight))
                    if budget_weight:
                        parts.append(logits)
                        continue
                    target = torch.as_tensor(store.correct[ids], device=device)
                    loss = head.loss(logits, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    (loss * len(ids) / len(window)).backward()
                    total_loss += float(loss.detach()) * len(ids)
                if budget_weight:
                    logits = torch.cat(parts)
                    correct = store.correct[window]
                    bce = head.loss(logits, torch.as_tensor(correct, device=device))
                    budget_loss = structured_budget_loss(head.probabilities(logits), correct, calls[window],
                                                         config.deep_fraction, config.harm_weights)
                    loss = bce + budget_weight * budget_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite budget-aware training loss")
                    loss.backward()
                    total_loss += float(bce.detach()) * len(window)
                    total_budget_loss += float(budget_loss.detach()) * len(window)
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                scheduler.step()
                seen = min(start + len(window), len(order))
                if step % 20 == 0:
                    print(f"{variant} {phase} epoch={epoch} step={step} seen={seen}/{len(order)}", flush=True)
                if step not in validation_steps:
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits, elapsed = predict(store, head, store.dev_indices, device, backbone, formatter, encoded)
                dev_logits = torch.as_tensor(logits)
                dev_loss = float(head.loss(dev_logits, dev_targets))
                probabilities = head.probabilities(dev_logits).numpy()
                record = {"phase": phase, "epoch": epoch, "step": step, "train_queries": seen,
                          "loss": total_loss / seen, "budget_loss": total_budget_loss / seen,
                          "dev_loss": dev_loss, "dev_seconds": elapsed, "policies": []}
                if str(device).startswith("cuda"):
                    record["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                for weight in config.harm_weights:
                    result = legacy().eval_policy(dev, legacy().gain_utility(probabilities, weight), baseline)
                    candidate = {"phase": phase, "epoch": epoch, "step": step, "harm_weight": weight, "dev": result}
                    record["policies"].append(candidate)
                    print(f"{variant} {phase} epoch={epoch} step={step} weight={weight} "
                          f"dev_R1={result['new']['R@1']:.3f} harms={result['flips']['harmful_flips']}", flush=True)
                candidate = max(record["policies"], key=lambda item: legacy().selection_key(item["dev"], baseline))
                improved = eligible_for_selection and (best is None or
                    legacy().selection_key(candidate["dev"], baseline) > legacy().selection_key(best["dev"], baseline))
                if improved:
                    publish_best(folder, head, backbone)
                    best = report["selection"] = candidate
                    stale_checks = 0
                elif phase == "joint":
                    stale_checks += 1
                report["validations"].append(record)
                atomic_json(folder / "training.json", report)
                if phase == "joint" and patience and stale_checks >= patience:
                    print(f"{variant} early stopping after {stale_checks} development checks without improvement", flush=True)
                    stopped = True
                    break
                head.train()
                if backbone is not None:
                    backbone.train()
            report["epochs"].append(record)
            atomic_json(folder / "training.json", report)
            if stopped:
                break
        del optimizer, scheduler
    report.update(training_seconds=time.perf_counter() - training_start,
                  early_stopping={"patience": patience, "triggered": stopped, "checks_without_improvement": stale_checks})
    report["checkpoint_sha256"] = checkpoint_hashes(folder / "best", best["phase"])
    atomic_json(folder / "training.json", report)
    atomic_json(completed, report)
    del backbone, head, store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def load_selected(run_dir, variant, store, device):
    folder = Path(run_dir) / variant
    report = read_json(folder / "training_complete.json")
    if report["signature"] != experiment_signature(store):
        raise ValueError("Evaluation inputs or implementation differ from training")
    checkpoint = folder / "best"
    phase = report["selection"]["phase"]
    if checkpoint_hashes(checkpoint, phase) != report["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint files changed")
    head = new_head(store, variant).to(device)
    head.load_state_dict(torch.load(checkpoint / "head.pt", map_location=device, weights_only=True))
    backbone = formatter = encoded = None
    if phase == "joint":
        tokenizer, backbone = load_router(str(resolve(store.cfg.router.base_model)),
                                          adapter=str(checkpoint / "adapter"), device=device)
        backbone.config.use_cache = False
        formatter = JevFormatter(tokenizer, max_len=store.cfg.router.max_len)
        encoded = encode(store.rows, store.cfg, formatter)
    return head, backbone, formatter, encoded, report
