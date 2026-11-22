"""Jev-style calibrated decision router (method 1).

Following PyTrio-Jev: the model never generates free text. The assistant turn is a JSON
skeleton whose answer slots are a fixed marker token; supervision is applied only at the
position right before each marker, and inference reads a restricted softmax over the
option letters at that position. One forward pass answers both questions:

* action  : accept (trust first stage) / light (yes-no check) / deep (counterfactual check)
* fragile : which compositional constraint is most likely to be confused (steers method 2)
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Sequence

import numpy as np

from .counterfactual import FRAGILE_TYPES
from .retriever import score_stats
from .utils import log_softmax_np

ACTIONS = ["accept", "light", "deep"]
ACTION_DESC = {
    "accept": "trust the first-stage ranking (no VLM call)",
    "light": "quick yes/no check of the top candidates with a VLM",
    "deep": "counterfactual forced-choice verification of more candidates with a VLM",
}
FIELDS = [
    ("action", "Which is the cheapest path that puts the correct video at rank 1?", ACTIONS),
    ("fragile", "Which constraint of the query is most likely to be confused by near-miss videos?", FRAGILE_TYPES),
]
LETTERS = "ABCDEFGHIJ"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_state(row: dict, stat_temp: float = 0.01, stat_k: int = 10, n_show: int = 5) -> str:
    st = score_stats(row["scores"], temp=stat_temp, k=stat_k)
    return "\n".join([
        f'Query: "{row["text"]}"',
        f"Query length (words): {len(row['text'].split())}",
        f"Top-{n_show} first-stage similarities: " + ", ".join(f"{s:.4f}" for s in st["topk"][:n_show]),
        f"Top-1 minus top-2 margin: {st['margin']:.4f}",
        f"Normalized entropy over top-{stat_k}: {st['entropy']:.4f}",
        f"Top-1 probability over top-{stat_k}: {st['p_top1']:.4f}",
    ])


class JevFormatter:
    def __init__(self, tokenizer, marker: str = "decision", max_len: int = 768):
        self.tok = tokenizer
        self.max_len = max_len
        self.marker_id = tokenizer.encode(marker, add_special_tokens=False)[0]
        self.letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
        self.cand_ids = [self.letter_ids[: len(opts)] for _, _, opts in FIELDS]

    def prompt_text(self, state: str) -> str:
        paths = "\n".join(f"- {a}: {ACTION_DESC[a]}" for a in ACTIONS)
        qs = []
        for name, q, opts in FIELDS:
            optline = " ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts))
            qs.append(f"[{name}] {q} Options: {optline}")
        return (
            "<|im_start|>user\n### State\n" + state + "\n\n### Paths\n" + paths
            + "\n\n### Questions\n" + "\n".join(qs)
            + "\nAnswer each question with one option letter in the JSON below.<|im_end|>\n<|im_start|>assistant\n"
        )

    def encode(self, state: str):
        ids = self.tok.encode(self.prompt_text(state), add_special_tokens=False)
        tail_budget = 48
        if len(ids) > self.max_len - tail_budget:
            ids = ids[: self.max_len - tail_budget]
        positions = []
        for fi, (name, _, _) in enumerate(FIELDS):
            pre = ('{"' if fi == 0 else '", "') + name + '": "'
            ids += self.tok.encode(pre, add_special_tokens=False)
            positions.append(len(ids) - 1)  # logits here predict the slot
            ids.append(self.marker_id)
        ids += self.tok.encode('"}', add_special_tokens=False)
        return ids, positions


def load_router(base_model: str, adapter: str | None = None, device: str = "cuda:0",
                train: bool = False, lora_r: int = 32, lora_alpha: int = 64):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
        ))
    if train:
        model.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
    else:
        model.eval()
    return tok, model


def _collate(encoded: Sequence[tuple], pad_id: int, device):
    import torch

    L = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(encoded), L), dtype=torch.long)
    pos = torch.zeros((len(encoded), len(FIELDS)), dtype=torch.long)
    for i, (ids, p) in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        att[i, : len(ids)] = 1
        pos[i] = torch.tensor(p)
    return input_ids.to(device), att.to(device), pos.to(device)


def field_states(model, fmt: JevFormatter, encoded: Sequence[tuple], device):
    """Return the causal LM and its hidden states just before the answer slots."""
    import torch

    input_ids, att, pos = _collate(encoded, fmt.tok.pad_token_id, device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=input_ids, attention_mask=att).last_hidden_state
    rows = torch.arange(hidden.size(0), device=hidden.device)[:, None]
    return base, hidden[rows, pos]


def field_logits(model, fmt: JevFormatter, encoded: Sequence[tuple], device):
    """Restricted logits per field: list of (B, n_options) float tensors."""
    base, states = field_states(model, fmt, encoded, device)
    return [base.lm_head(states[:, slot]).float()[:, candidates]
            for slot, candidates in enumerate(fmt.cand_ids)]


def _labels(rows: Sequence[dict]):
    a = [ACTIONS.index(r["label_action"]) for r in rows]
    f = [FRAGILE_TYPES.index(r["label_fragile"]) if r["label_fragile"] in FRAGILE_TYPES else FRAGILE_TYPES.index("none")
         for r in rows]
    return np.asarray(a), np.asarray(f)


def action_classification(logits: np.ndarray, labels: np.ndarray) -> dict:
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = confusion.diagonal()
    present = support > 0
    per_example_nll = -log_softmax_np(logits)[np.arange(len(labels)), labels]
    class_nll = np.bincount(labels, weights=per_example_nll, minlength=len(ACTIONS))
    return {
        "accuracy": float(correct.sum() / len(labels)),
        "macro_f1": float((2 * correct[present] / (support + predicted)[present]).mean()),
        "balanced_nll": float((class_nll[present] / support[present]).mean()),
        "prediction_counts": dict(zip(ACTIONS, predicted.tolist())),
        "recall": {action: float(correct[i] / support[i]) if present[i] else None
                   for i, action in enumerate(ACTIONS)},
    }


def predict_router(model, fmt: JevFormatter, rows: Sequence[dict], device, batch_size: int = 16,
                   stat_temp: float = 0.01, stat_k: int = 10) -> Dict[str, np.ndarray]:
    import torch

    model.eval()
    outs = {name: [] for name, _, _ in FIELDS}
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            enc = [fmt.encode(build_state(r, stat_temp, stat_k)) for r in rows[i:i + batch_size]]
            for (name, _, _), lg in zip(FIELDS, field_logits(model, fmt, enc, device)):
                outs[name].append(lg.cpu().numpy())
    return {k: np.concatenate(v, 0) if v else np.zeros((0, 1)) for k, v in outs.items()}


def train_router(cfg, train_rows: List[dict], dev_rows: List[dict], out_dir: str) -> str:
    import torch
    import torch.nn.functional as F
    from transformers import get_cosine_schedule_with_warmup

    rc = cfg.router
    device = torch.device(rc.device)
    tok, model = load_router(rc.base_model, device=rc.device, train=True, lora_r=rc.lora_r, lora_alpha=rc.lora_alpha)
    fmt = JevFormatter(tok, max_len=rc.max_len)
    st_t, st_k = cfg.retriever.stat_temp, cfg.retriever.stat_k

    enc = [fmt.encode(build_state(r, st_t, st_k)) for r in train_rows]
    ya, yf = _labels(train_rows)
    counts = np.bincount(ya, minlength=len(ACTIONS)).astype(np.float64)
    w = np.sqrt(counts.sum() / (len(ACTIONS) * np.maximum(counts, 1.0)))
    w_action = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print("[router] action label counts:", dict(zip(ACTIONS, counts.astype(int).tolist())))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rc.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(enc) / (rc.batch_size * rc.grad_accum))
    total = max(1, steps_per_epoch * rc.epochs)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total)), total)

    best_dir = os.path.join(out_dir, "best")
    best_key, step = None, 0
    log_path = os.path.join(out_dir, "train_log.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    order = list(range(len(enc)))
    for ep in range(rc.epochs):
        model.train()
        random.Random(cfg.seed + ep).shuffle(order)
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, len(order), rc.batch_size)):
            idx = order[i:i + rc.batch_size]
            la, lf = field_logits(model, fmt, [enc[j] for j in idx], device)
            ta = torch.tensor(ya[idx], device=device)
            tf = torch.tensor(yf[idx], device=device)
            loss = F.cross_entropy(la, ta, weight=w_action) + rc.fragile_weight * F.cross_entropy(lf, tf)
            (loss / rc.grad_accum).backward()
            if (bi + 1) % rc.grad_accum == 0 or i + rc.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"epoch": ep, "step": step, "loss": float(loss.item()),
                                            "lr": sched.get_last_lr()[0]}) + "\n")
        epoch_report = {"epoch": ep}
        if dev_rows:
            pred = predict_router(model, fmt, dev_rows, device, rc.eval_batch_size, st_t, st_k)
            da, df = _labels(dev_rows)
            action = action_classification(pred["action"], da)
            acc_f = float((pred["fragile"].argmax(1) == df).mean())
            selection_key = (action["macro_f1"], -action["balanced_nll"], acc_f)
            epoch_report.update(
                dev_action_acc=action["accuracy"],
                dev_fragile_acc=acc_f,
                dev_action=action,
            )
            print(f"[router] epoch {ep}: dev action acc={action['accuracy']:.4f} "
                  f"macro-F1={action['macro_f1']:.4f} fragile acc={acc_f:.4f} "
                  f"action predictions={action['prediction_counts']}")
        else:
            selection_key = (ep,)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_report) + "\n")
        model.save_pretrained(os.path.join(out_dir, f"epoch_{ep}"))
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            model.save_pretrained(best_dir)
            tok.save_pretrained(best_dir)
    return best_dir


def decide_action(p_action: np.ndarray, tau_accept: float) -> str:
    """Accept if calibrated p(accept) >= tau (CRC threshold), else the likelier VLM path."""
    if p_action[0] >= tau_accept:
        return "accept"
    return "deep" if p_action[2] >= p_action[1] else "light"
