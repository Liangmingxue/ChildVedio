"""Shared helpers for the fanganD offline cascade + learned-fusion experiments (numpy only)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

K = 16  # coarse candidates kept by the retriever (stage config retriever.topk)
BUDGET = 27824  # VLM calls used by the margin baseline on the 1000 test queries


def parse(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_run(run_dir):
    """Load train-side rows, test rows and per-query cached costs/margin actions from a fanganC run."""
    run = Path(run_dir)
    rows = read_jsonl(run / "snapshot/router_data.jsonl")
    test = read_jsonl(run / "snapshot/test.jsonl")
    pred = {x["qid"]: x for x in read_jsonl(run / "candidate/test_predictions.jsonl")}
    cost = np.array([pred[x["qid"]]["cached_call_costs"] for x in test])  # [accept, light, deep]
    margin_action = np.array([pred[x["qid"]]["margin_action"] for x in test])
    return rows, test, cost, margin_action


def cached_ranks(rows):
    return {k: np.array([int(parse(x["ranks"])[k]) for x in rows]) for k in ("accept", "light", "deep")}


def coarse_margin(rows):
    out = []
    for x in rows:
        s = parse(x["scores"])
        out.append(s[0] - s[1])
    return np.array(out)


def metrics(ranks):
    r = np.asarray(ranks)
    return {"R@1": float((r <= 1).mean() * 100), "R@5": float((r <= 5).mean() * 100),
            "R@10": float((r <= 10).mean() * 100), "MdR": float(np.median(r)), "MnR": float(r.mean())}


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + .5 * (pos[:, None] == neg[None]).mean())


def features(row, use_light, use_deep):
    """Per-candidate features over the coarse top-16. No GT, caption or label is used."""
    cands = list(parse(row["cands"]))[:K]
    s = np.array(parse(row["scores"])[:K], float)
    n = len(s)
    minmax = (s - s.min()) / (s.max() - s.min() + 1e-9)
    z = (s - s.mean()) / (s.std() + 1e-9)
    gap = s[0] - s[1]
    pos = np.arange(n)
    light, light_mask = np.zeros(n), np.zeros(n)
    if use_light:
        v = np.array(parse(row["light_scores"]), float)
        light[:len(v)], light_mask[:len(v)] = v, 1
    deep, deep_mask = np.zeros(n), np.zeros(n)
    if use_deep:
        v = np.array(parse(row["deep_scores"]), float)
        deep[:len(v)], deep_mask[:len(v)] = v, 1

    def log(v):
        return np.log(np.clip(v, 1e-6, 1))

    def within_rank(v, m):
        out = np.zeros(n)
        if m.any():
            out[m > 0] = np.argsort(np.argsort(-v[m > 0]))
        return out

    light_rank, deep_rank = within_rank(light, light_mask), within_rank(deep, deep_mask)
    columns = [
        minmax, z, s - s[0], np.log1p(pos), (pos == 0) * 1., (pos < 5) * 1.,
        light, log(light) * light_mask, light_mask, (light_rank == 0) * light_mask, light * minmax,
        deep, log(deep) * deep_mask, deep_mask, (deep_rank == 0) * deep_mask, deep * minmax,
        (deep - deep.max()) * deep_mask,
        deep * light, log(deep) * deep_mask * gap / 10, log(light) * light_mask * gap / 10,
        (pos == 0) * gap / 10, deep * (pos == 0), light * (pos == 0),
    ]
    gt = cands.index(row["gt"]) if row["gt"] in cands else -1
    return np.stack(columns, 1), gt


def build(rows, use_light, use_deep):
    return [features(x, use_light, use_deep) for x in rows]


class ListwiseFusion:
    """Listwise softmax (ListNet top-1) scorer: linear (hidden=0) or one tanh hidden layer, Adam, full batch."""

    def __init__(self, hidden=16, epochs=400, lr=.05, l2=1e-3, seed=0):
        self.hidden, self.epochs, self.lr, self.l2, self.seed = hidden, epochs, lr, l2, seed

    def fit(self, data):
        rng = np.random.default_rng(self.seed)
        kept = [(X, g) for X, g in data if g >= 0]  # GT outside top-16 cannot be moved by reranking
        stacked = np.concatenate([X for X, _ in kept])
        self.mu, self.sd = stacked.mean(0), stacked.std(0) + 1e-6
        d, n = stacked.shape[1], len(kept)
        X = np.zeros((n, K, d))
        mask = np.zeros((n, K), bool)
        for i, (x, _) in enumerate(kept):
            X[i, :len(x)], mask[i, :len(x)] = (x - self.mu) / self.sd, True
        gt = np.array([g for _, g in kept])
        h = self.hidden
        self.W1 = rng.normal(0, .1, (d, h)) if h else None
        self.w = rng.normal(0, .01, h if h else d)
        params = [self.w] + ([self.W1] if h else [])
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        target = np.zeros((n, K))
        target[np.arange(n), gt] = 1
        for t in range(1, self.epochs + 1):
            H = np.tanh(X @ self.W1) if h else None
            score = H @ self.w if h else X @ self.w
            score = np.where(mask, score, -1e9)
            score -= score.max(1, keepdims=True)
            p = np.exp(score) * mask
            p /= p.sum(1, keepdims=True)
            ds = (p - target) / n
            if h:
                grads = [np.einsum("nk,nkh->h", ds, H) + self.l2 * self.w]
                dH = ds[..., None] * self.w * (1 - H ** 2)
                grads.append(np.einsum("nkh,nkd->dh", dH, X) + self.l2 * self.W1)
            else:
                grads = [np.einsum("nk,nkd->d", ds, X) + self.l2 * self.w]
            for i, (param, grad) in enumerate(zip(params, grads)):
                m1[i] = .9 * m1[i] + .1 * grad
                m2[i] = .999 * m2[i] + .001 * grad ** 2
                param -= self.lr * (m1[i] / (1 - .9 ** t)) / (np.sqrt(m2[i] / (1 - .999 ** t)) + 1e-8)
        return self

    def score(self, X):
        X = (X - self.mu) / self.sd
        return np.tanh(X @ self.W1) @ self.w if self.hidden else X @ self.w

    def ranks(self, data, rows):
        out = []
        for (X, gt), row in zip(data, rows):
            if gt < 0:
                out.append(int(row["gt_rank"]))
                continue
            s = self.score(X)
            out.append(int((s > s[gt]).sum()) + 1)
        return np.array(out)

    def top_gap(self, data):
        """Softmax top-1 minus top-2 probability: a post-light confidence used for escalation."""
        out = []
        for X, _ in data:
            s = self.score(X)
            p = np.exp(s - s.max())
            p = np.sort(p / p.sum())
            out.append(p[-1] - p[-2])
        return np.array(out)


def beta_ranks(rows, key, beta):
    """Reproduce cvr.rerank.rerank_topk: beta*vlm + (1-beta)*minmax(coarse) over the verified prefix."""
    out = []
    for x in rows:
        cands = list(parse(x["cands"]))
        s = np.array(parse(x["scores"]), float)
        v = np.array(parse(x[key]), float)
        k = len(v)
        base = s[:k]
        comb = beta * v + (1 - beta) * (base - base.min()) / (base.max() - base.min() + 1e-9)
        order = list(np.array(cands[:k])[np.argsort(-comb, kind="stable")]) + cands[k:]
        out.append(order.index(x["gt"]) + 1 if x["gt"] in order else int(x["gt_rank"]))
    return np.array(out)


def fill_budget(priority, extra_cost, spent, budget=BUDGET):
    """Greedily select queries in priority order (lowest first) until the next one exceeds the budget."""
    selected = np.zeros(len(priority), bool)
    for i in np.argsort(priority, kind="stable"):
        if spent + extra_cost[i] > budget:
            break
        spent += extra_cost[i]
        selected[i] = True
    return selected, int(spent)


def paired_bootstrap(new_hit, ref_hit, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(new_hit), (n, len(new_hit)))
    delta = (new_hit[idx].astype(float) - ref_hit[idx]).mean(1) * 100
    return {"delta": float((new_hit.astype(float) - ref_hit).mean() * 100),
            "ci95": [float(x) for x in np.percentile(delta, [2.5, 97.5])],
            "p_delta_le_0": float((delta <= 0).mean())}


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
