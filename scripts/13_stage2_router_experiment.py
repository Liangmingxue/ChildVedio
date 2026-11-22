"""Compare saved LoRA checkpoints and train an outcome router on the complete training split."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from peft import PeftModel
from torch import nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from cvr.calibration import apply_temperature, fit_temperature
from cvr.metrics import summarize_ranks
from cvr.retriever import score_stats
from cvr.router import (
    ACTIONS, JevFormatter, _labels, action_classification, build_state, field_states,
    load_router, predict_router,
)
from cvr.utils import add_common_args, load_config, read_json, read_jsonl, set_seed, write_json


@dataclass
class Replay:
    rows: list
    ranks: np.ndarray
    calls: np.ndarray
    margins: np.ndarray

    @classmethod
    def from_rows(cls, rows, cfg):
        repeats = max(1, cfg.vlm.n_perm)
        fallback = cfg.rerank.get("empty_cf_fallback")
        ranks = [[row["ranks"][a] for a in ACTIONS] for row in rows]
        calls = [[0, len(row["light_scores"]), len(row["deep_scores"]) *
                  (1 if fallback and not row["hyps"] else repeats)] for row in rows]
        margins = [score_stats(row["scores"], k=2)["margin"] for row in rows]
        return cls(rows, np.asarray(ranks), np.asarray(calls), np.asarray(margins))

    def margin_actions(self, fraction, quotas=None):
        by_cost = defaultdict(list)
        for i, cost in enumerate(self.calls[:, 2]):
            by_cost[cost].append(i)
        if quotas is not None and any(cost not in by_cost or not 0 <= quota <= len(by_cost[cost])
                                      for cost, quota in quotas.items()):
            raise ValueError("The test budget does not match the cached deep-verification costs")
        actions = np.zeros(len(self.rows), dtype=np.uint8)
        for cost, group in by_cost.items():
            quota = round(len(group) * fraction) if quotas is None else quotas.get(cost, 0)
            ordered = sorted(group, key=self.margins.__getitem__)
            actions[ordered[:quota]] = 2
        return actions

    def metrics(self, actions):
        idx = np.arange(len(self.rows))
        calls = int(self.calls[idx, actions].sum())
        return {
            **summarize_ranks(self.ranks[:, 0], self.ranks[idx, actions]),
            "route_dist": dict(Counter(ACTIONS[a] for a in actions)),
            "vlm_calls": calls,
            "vlm_calls_per_query": calls / len(self.rows),
        }


def allocate(utilities, calls, budget):
    """Maximize one predicted utility per query at the exact integer call budget."""
    if not np.isfinite(utilities).all() or (calls[:, 1:] <= 0).any():
        raise ValueError("Routing requires finite utilities and positive verification costs")
    values = np.full(budget + 1, -np.inf)
    values[0] = 0
    choices = np.zeros((len(calls), budget + 1), dtype=np.uint8)
    for i, (row_values, row_calls) in enumerate(zip(utilities, calls)):
        current = values.copy()
        for action in (1, 2):
            cost = row_calls[action]
            if cost > budget:
                continue
            candidate = values[:-cost] + row_values[action]
            better = candidate > current[cost:]
            current[cost:][better] = candidate[better]
            choices[i, cost:][better] = action
        values = current
    if not np.isfinite(values[budget]):
        raise ValueError(f"No routing allocation uses exactly {budget} VLM calls")
    actions = np.zeros(len(calls), dtype=np.uint8)
    for index in range(len(calls) - 1, -1, -1):
        actions[index] = choices[index, budget]
        budget -= calls[index, actions[index]]
    return actions


def eval_policy(replay, utilities, baseline):
    budget = baseline["vlm_calls"]
    result = replay.metrics(allocate(utilities, replay.calls, budget))
    if result["vlm_calls"] != budget:
        raise RuntimeError("Replayed policy used a different VLM budget")
    return result


def selection_key(result, baseline):
    harms = result["flips"]["harmful_flips"]
    return (harms <= baseline["flips"]["harmful_flips"], result["new"]["R@1"], -harms)


def print_result(name, result):
    print(f"{name} R@1={result['new']['R@1']:.2f}% HFR={result['flips']['HFR']:.2f}% "
          f"calls/q={result['vlm_calls_per_query']:.3f} {result['route_dist']}", flush=True)


def save_report(path, report):
    temp = path.with_suffix(".tmp")
    write_json(temp, report)
    temp.replace(path)


def classifier_utility(logits, temperature):
    probabilities = apply_temperature(logits, temperature)
    return probabilities - probabilities[:, :1]


class GainHead(nn.Module):
    """Predict unchanged, corrected and harmful outcomes for light and deep."""

    def __init__(self, hidden_size, margins, outputs=6):
        super().__init__()
        self.register_buffer("margin_mean", torch.tensor(float(margins.mean())))
        self.register_buffer("margin_scale", torch.tensor(max(float(margins.std()), 1e-6)))
        self.linear = nn.Linear(hidden_size + 1, outputs)

    def features(self, states, margins):
        states = F.layer_norm(states.float(), (states.shape[-1],))
        margins = ((margins - self.margin_mean) / self.margin_scale).unsqueeze(1)
        return torch.cat((states, margins), dim=1)

    def forward(self, states, margins):
        return self.linear(self.features(states, margins)).reshape(-1, 2, 3)

    def loss(self, logits, correct):
        base, verified = correct[:, :1], correct[:, 1:]
        labels = (~base & verified).long() + 2 * (base & ~verified).long()
        return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))

    def probabilities(self, logits):
        return logits.softmax(dim=-1)


class FactorizedGainHead(GainHead):
    """Predict coarse correctness and verification success conditional on it."""

    def __init__(self, hidden_size, margins):
        super().__init__(hidden_size, margins, outputs=5)
        self.register_buffer("margin_prior", torch.zeros(2))
        nn.init.zeros_(self.linear.weight[0])
        nn.init.zeros_(self.linear.bias[0])

    def fit_margin_prior(self, margins, coarse_correct):
        margin = (torch.as_tensor(margins, dtype=torch.float64) - self.margin_mean) / self.margin_scale
        target = torch.as_tensor(coarse_correct, dtype=torch.float64)
        coefficients = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([coefficients], max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = coefficients[0] + coefficients[1] * margin
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.margin_prior.copy_(coefficients.detach())

    def forward(self, states, margins):
        features = self.features(states, margins)
        logits = self.linear(features)
        prior = self.margin_prior[0] + self.margin_prior[1] * features[:, -1]
        return torch.cat((logits[:, :1] + prior[:, None], logits[:, 1:]), dim=1)

    def loss(self, logits, correct):
        coarse = correct[:, 0]
        conditional = logits[:, 1:].reshape(-1, 2, 2)
        condition = coarse.long().reshape(-1, 1, 1).expand(-1, 2, 1)
        observed = conditional.gather(dim=2, index=condition).squeeze(2)
        return (
            F.binary_cross_entropy_with_logits(logits[:, 0], coarse.float())
            + F.binary_cross_entropy_with_logits(observed, correct[:, 1:].float())
        )

    def probabilities(self, logits):
        coarse = logits[:, :1].sigmoid()
        conditional = logits[:, 1:].reshape(-1, 2, 2).sigmoid()
        fixes = (1 - coarse) * conditional[:, :, 0]
        harms = coarse * (1 - conditional[:, :, 1])
        return torch.stack((1 - fixes - harms, fixes, harms), dim=-1)


GAIN_HEADS = {"joint": GainHead, "factorized": FactorizedGainHead}


def encode_rows(cfg, fmt, rows):
    return [fmt.encode(build_state(row, cfg.retriever.stat_temp, cfg.retriever.stat_k)) for row in rows]


@torch.inference_mode()
def predict_gain(model, fmt, head, cfg, replay):
    model.eval()
    head.eval()
    encoded = encode_rows(cfg, fmt, replay.rows)
    rc = cfg.router
    device = rc.device
    batch_size = min(rc.eval_batch_size, rc.get("gain_microbatch", rc.batch_size))
    probs = []
    for start in range(0, len(encoded), batch_size):
        end = start + batch_size
        _, states = field_states(model, fmt, encoded[start:end], device)
        margin = torch.as_tensor(replay.margins[start:end], dtype=torch.float32, device=device)
        logits = head(states[:, 0], margin)
        probs.append(head.probabilities(logits).cpu().numpy())
    return np.concatenate(probs)


def gain_utility(probabilities, harm_weight):
    gains = probabilities[:, :, 1] - harm_weight * probabilities[:, :, 2]
    return np.column_stack((np.zeros(len(gains)), gains))


def save_gain_weights(out_dir, epoch, model, tok, head):
    adapter = out_dir / f"gain_epoch_{epoch}"
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    torch.save(head.state_dict(), adapter / "gain_head.pt")
    return str(adapter)


def train_gain(cfg, train, dev, baseline, out_dir, report, report_path, *, fixed_selection=None):
    rc = cfg.router
    head_kind = rc.get("gain_head", "joint")
    device = torch.device(rc.device)
    effective_batch = rc.batch_size * rc.grad_accum
    microbatch = min(rc.get("gain_microbatch", rc.batch_size), effective_batch)
    epochs = rc.epochs if fixed_selection is None else fixed_selection["epoch"] + 1
    print(f"Training {head_kind} head on {len(train.rows)} queries per epoch for {epochs} epochs "
          f"(lr={rc.lr:g}, microbatch={microbatch}, effective_batch={effective_batch})", flush=True)
    tok, model = load_router(
        rc.base_model, device=rc.device, train=True, lora_r=rc.lora_r, lora_alpha=rc.lora_alpha
    )
    model.config.use_cache = False
    fmt = JevFormatter(tok, max_len=rc.max_len)
    head = GAIN_HEADS[head_kind](model.config.hidden_size, train.margins)
    correct = torch.as_tensor(train.ranks == 1)
    if head_kind == "factorized":
        head.fit_margin_prior(train.margins, correct[:, 0])
        print(f"Fitted coarse margin prior {head.margin_prior.tolist()}", flush=True)
    head.to(rc.device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=rc.lr)
    batches = (len(train.rows) + effective_batch - 1) // effective_batch
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(0.03 * batches * rc.epochs)),
                                                batches * rc.epochs)
    encoded = encode_rows(cfg, fmt, train.rows)
    fragile = torch.as_tensor(_labels(train.rows)[1])
    margins = torch.as_tensor(train.margins, dtype=torch.float32)
    best = None
    for epoch in range(epochs):
        model.train()
        head.train()
        order = list(range(len(encoded)))
        random.Random(cfg.seed + epoch).shuffle(order)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seen = 0
        for step, start in enumerate(range(0, len(order), effective_batch), 1):
            window = order[start:start + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0
            for offset in range(0, len(window), microbatch):
                ids = window[offset:offset + microbatch]
                base, states = field_states(model, fmt, [encoded[i] for i in ids], device)
                outcome = head(states[:, 0], margins[ids].to(device))
                fragile_logits = base.lm_head(states[:, 1]).float()[:, fmt.cand_ids[1]]
                loss = head.loss(outcome, correct[ids].to(device))
                loss = loss + rc.fragile_weight * F.cross_entropy(fragile_logits, fragile[ids].to(device))
                loss = loss * (len(ids) / len(window))
                loss.backward()
                step_loss = step_loss + loss.detach()
                seen += len(ids)
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            if step % 20 == 0:
                print(f"gain epoch {epoch} step {step} queries={seen}/{len(train.rows)} "
                      f"loss={step_loss.item():.4f}", flush=True)
        optimizer.zero_grad(set_to_none=True)

        probabilities = predict_gain(model, fmt, head, cfg, dev) if fixed_selection is None else None
        training_stats = {"epoch": epoch, "queries_seen": seen, "optimizer_steps": step}
        if device.type == "cuda":
            training_stats["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        if fixed_selection is not None:
            report.setdefault("refit_training", []).append(training_stats)
            save_report(report_path, report)
            continue
        report.setdefault("gain_training", []).append(training_stats)
        for weight in (1.0, 2.0):
            result = eval_policy(dev, gain_utility(probabilities, weight), baseline)
            print_result(f"gain epoch {epoch} harm_weight={weight:g}", result)
            report["gain_epochs"].append({"epoch": epoch, "harm_weight": weight, "dev": result})
            if best is None or selection_key(result, baseline) > selection_key(best["dev"], baseline):
                adapter = save_gain_weights(out_dir, epoch, model, tok, head)
                best = {"type": "gain", "gain_head": head_kind, "adapter": adapter, "epoch": epoch,
                        "harm_weight": weight, "dev": result}
                report["selection"] = best
            save_report(report_path, report)
    if fixed_selection is not None:
        adapter = save_gain_weights(out_dir, epoch, model, tok, head)
        return {**fixed_selection, "adapter": adapter}
    return best


def load_test(cfg, stage):
    retrieval = read_jsonl(stage / "retrieval/test.jsonl")
    light = {r["qid"]: r for r in read_jsonl(stage / "pipeline/test_light.jsonl")}
    deep = {r["qid"]: r for r in read_jsonl(stage / "pipeline/test_deep_prompt.jsonl")}
    qids = {r["qid"] for r in retrieval}
    if qids != light.keys() or qids != deep.keys() or len(qids) != len(retrieval):
        raise ValueError("Test retrieval and the complete light/deep results must cover the same queries")
    rows = []
    for row in retrieval:
        l, d = light[row["qid"]], deep[row["qid"]]
        if any(x["text"] != row["text"] or x["base_rank"] != row["gt_rank"] for x in (l, d)):
            raise ValueError(f"Cached test results do not match {row['qid']}")
        rows.append({**row, "ranks": {"accept": row["gt_rank"], "light": l["new_rank"], "deep": d["new_rank"]},
                     "light_scores": l["scores"], "deep_scores": d["scores"], "hyps": d["hyps"]})
    return Replay.from_rows(rows, cfg)


def split_router_holdout(rows, fraction, seed):
    qids = {row["qid"] for row in rows}
    if len(qids) != len(rows):
        raise ValueError("Router supervision contains duplicate query IDs")
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["qid"])
    test_count = round(len(rows) * fraction)
    if not 0 < test_count < len(train):
        raise ValueError("The holdout must leave some of the training rows available for LoRA")
    held_out = {row["qid"] for row in random.Random(seed).sample(train, test_count)}
    test = [row for row in rows if row["qid"] in held_out]
    development = [row for row in rows if row["qid"] not in held_out]
    if {row["gt"] for row in test} & {row["gt"] for row in development}:
        raise ValueError("Holdout videos overlap the training, development or calibration data")
    return development, test


def sample_margin_accuracy(replay, actions, seed, sample_size=1000, repeats=1000):
    """Slice one fixed set of predictions to isolate variation from test sample size."""
    correct = replay.ranks[np.arange(len(actions)), actions] == 1
    rng = np.random.default_rng(seed)
    results = np.array([
        100 * correct[rng.choice(len(correct), sample_size, replace=False)].mean()
        for _ in range(repeats)
    ])
    low, high = np.quantile(results, [0.025, 0.975])
    return {
        "sample_size": sample_size, "repeats": repeats, "seed": seed,
        "mean_R@1": float(results.mean()),
        "central_95pct_R@1": [float(low), float(high)],
    }


def replay_checkpoints(cfg, checkpoints, dev, baseline, report, report_path):
    rc = cfg.router
    tok, model = load_router(rc.base_model, device=rc.device)
    model.config.use_cache = False
    fmt = JevFormatter(tok, max_len=rc.max_len)
    ya, _ = _labels(dev.rows)
    winners = []
    for path in checkpoints:
        print(f"Replaying {path}", flush=True)
        adapter = PeftModel.from_pretrained(model, path).eval()
        logits = predict_router(adapter, fmt, dev.rows, rc.device, rc.eval_batch_size,
                                cfg.retriever.stat_temp, cfg.retriever.stat_k)["action"]
        model = adapter.unload()
        del adapter
        temperature, _ = fit_temperature(logits, ya)
        result = eval_policy(dev, classifier_utility(logits, temperature), baseline)
        candidate = {"type": "classifier", "adapter": str(path), "temperature": temperature, "dev": result,
                     "action_metrics": action_classification(logits, ya)}
        report["checkpoints"].append(candidate)
        print_result(path.name, result)
        if selection_key(result, baseline) > selection_key(baseline, baseline):
            winners.append(candidate)
        save_report(report_path, report)
    return winners


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--router-data", type=Path, help="path to the router supervision JSONL")
    parser.add_argument("--deep-fraction", required=True, type=float, help="margin baseline deep quota per cost group")
    test_options = parser.add_mutually_exclusive_group()
    test_options.add_argument("--test-budget-report", type=Path, help="reuse a previous matched-budget test report")
    test_options.add_argument("--holdout-fraction", type=float, default=0.0,
                              help="take this fraction of the router dataset from its training rows for testing")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--gain-if-needed", action="store_true", help="skip gain training if a saved adapter wins")
    start.add_argument("--skip-checkpoint-replay", action="store_true", help="start gain training immediately")
    parser.add_argument("--gain-head", choices=GAIN_HEADS, default="joint")
    parser.add_argument("--gain-microbatch", type=int, default=4, help="maximum gain examples per forward pass")
    parser.add_argument("--learning-rate", type=float, help="override the router learning rate")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0 < args.deep_fraction <= 1:
        parser.error("--deep-fraction must be greater than zero and at most one")
    if not 0 <= args.holdout_fraction < 1:
        parser.error("--holdout-fraction must be at least zero and less than one")
    if args.holdout_fraction and not args.skip_checkpoint_replay:
        parser.error("--holdout-fraction requires --skip-checkpoint-replay to train on the new split")
    cfg = load_config(args.config, args.set)
    if args.learning_rate is not None:
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("--learning-rate must be finite and positive")
        cfg.router.lr = args.learning_rate
    cfg.router.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    stage, out_dir = Path(cfg.paths.work_dir), args.output_dir
    if min(cfg.router.epochs, cfg.router.batch_size, cfg.router.grad_accum, cfg.router.eval_batch_size,
           args.gain_microbatch) < 1:
        parser.error("Training epochs, batch sizes and gradient accumulation must be positive")
    cfg.router.gain_microbatch = min(args.gain_microbatch, cfg.router.batch_size)
    cfg.router.gain_head = args.gain_head
    set_seed(cfg.seed)
    data_path = args.router_data or stage / "router_data/router_data.jsonl"
    source = read_jsonl(data_path)
    holdout_rows = []
    if args.holdout_fraction:
        source, holdout_rows = split_router_holdout(source, args.holdout_fraction, cfg.seed)
    training_splits = ("train", "calib") if holdout_rows else ("train",)
    train_rows = [row for row in source if row["split"] in training_splits]
    dev_rows = [row for row in source if row["split"] == "dev"]
    if not train_rows or not dev_rows:
        parser.error("The saved router data must contain train and dev rows")
    checkpoints = sorted(p.parent for p in (stage / "router").glob("epoch_*/adapter_model.safetensors"))
    if not checkpoints and not args.skip_checkpoint_replay:
        parser.error("The stage directory has no saved epoch adapters")
    report_path = out_dir / "report.json"
    if report_path.exists():
        parser.error("Choose a new --output-dir for this experiment")

    dev = Replay.from_rows(dev_rows, cfg)
    baseline = dev.metrics(dev.margin_actions(args.deep_fraction))
    test_quotas = None
    if args.test_budget_report:
        budget = read_json(args.test_budget_report)["shared_budget"]
        test_quotas = {int(cost): count for cost, count in budget["deep_query_count_by_vlm_calls"].items()}
    phase = "gain_training" if args.skip_checkpoint_replay else "checkpoint_replay"
    if holdout_rows:
        phase = "development_selection"
    report = {"phase": phase, "device": cfg.router.device, "deep_fraction": args.deep_fraction,
              "router_data": str(data_path),
              "dev_margin": baseline, "test_deep_quotas": test_quotas, "checkpoints": [], "gain_epochs": [],
              "train_queries": len(train_rows), "gain_head": args.gain_head, "gain_epochs_planned": cfg.router.epochs,
              "learning_rate": cfg.router.lr,
              "gain_microbatch": cfg.router.gain_microbatch,
              "gain_effective_batch": cfg.router.batch_size * cfg.router.grad_accum}
    if holdout_rows:
        split_qids = {
            "selection_train": [row["qid"] for row in train_rows],
            "selection_dev": [row["qid"] for row in dev_rows],
            "test": [row["qid"] for row in holdout_rows],
            "final_fit": [row["qid"] for row in source],
        }
        save_report(out_dir / "split.json", split_qids)
        report["holdout"] = {
            "fraction": args.holdout_fraction, "seed": cfg.seed,
            "counts": {name: len(qids) for name, qids in split_qids.items()},
        }
        report["selection_train_queries"] = len(train_rows)
        report["train_queries"] = len(source)
    save_report(report_path, report)
    print_result(f"dev margin ({cfg.router.device})", baseline)
    winners = []
    if not args.skip_checkpoint_replay:
        winners = replay_checkpoints(cfg, checkpoints, dev, baseline, report, report_path)
    if winners and args.gain_if_needed:
        selected = max(winners, key=lambda item: selection_key(item["dev"], baseline))
    else:
        report["phase"] = "development_selection" if holdout_rows else "gain_training"
        save_report(report_path, report)
        print("Training the gain/risk heads from saved route outcomes", flush=True)
        train = Replay.from_rows(train_rows, cfg)
        selected = train_gain(cfg, train, dev, baseline, out_dir, report, report_path)

    if holdout_rows:
        report.update(phase="full_refit", validation_selection=selected,
                      refit_epochs_planned=selected["epoch"] + 1)
        save_report(report_path, report)
        print(f"Refitting all {len(source)} training-side queries", flush=True)
        set_seed(cfg.seed)
        full_train = Replay.from_rows(source, cfg)
        selected = train_gain(cfg, full_train, None, None, out_dir / "refit", report, report_path,
                              fixed_selection=selected)

    report.update(phase="test_replay", selection=selected)
    save_report(report_path, report)
    test = Replay.from_rows(holdout_rows, cfg) if holdout_rows else load_test(cfg, stage)
    baseline_actions = test.margin_actions(args.deep_fraction, test_quotas)
    test_baseline = test.metrics(baseline_actions)
    if len(holdout_rows) > 1000:
        report["test_margin_subsamples"] = sample_margin_accuracy(test, baseline_actions, cfg.seed)
    tok, model = load_router(cfg.router.base_model, adapter=selected["adapter"], device=cfg.router.device)
    model.config.use_cache = False
    fmt = JevFormatter(tok, max_len=cfg.router.max_len)
    if selected["type"] == "gain":
        head = GAIN_HEADS[selected["gain_head"]](model.config.hidden_size, np.array([0.0])).to(cfg.router.device)
        head.load_state_dict(torch.load(Path(selected["adapter"]) / "gain_head.pt", map_location=cfg.router.device,
                                        weights_only=True))
        scores = gain_utility(predict_gain(model, fmt, head, cfg, test), selected["harm_weight"])
    else:
        logits = predict_router(model, fmt, test.rows, cfg.router.device, cfg.router.eval_batch_size,
                                cfg.retriever.stat_temp, cfg.retriever.stat_k)["action"]
        scores = classifier_utility(logits, selected["temperature"])
    result = eval_policy(test, scores, test_baseline)
    print_result("test margin", test_baseline)
    print_result("test selected router", result)
    report.update(phase="complete", test_margin=test_baseline, test_selected=result)
    save_report(report_path, report)


if __name__ == "__main__":
    main()
