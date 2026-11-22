"""Read-only audit of saved Stage 2 outcomes; writes only beside this script."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cvr.metrics import summarize_ranks

STAGE = ROOT / "outputs/stage2_clip4clip_qwen3vl8b"
ACTIONS = ("accept", "light", "deep")
CONFIG = yaml.safe_load((STAGE / "config.yaml").read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def existing_allocator():
    """Execute the repository's exact allocator without importing GPU dependencies."""
    path = ROOT / "scripts/13_stage2_router_experiment.py"
    source = path.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "allocate")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["allocate"], hashlib.sha256(source.encode()).hexdigest()


def arrays(rows):
    ranks = np.array([[row["ranks"][a] for a in ACTIONS] for row in rows])
    calls = np.array([[0, len(row["light_scores"]), len(row["deep_scores"]) *
                       (1 if CONFIG["rerank"].get("empty_cf_fallback") and not row["hyps"]
                        else max(1, CONFIG["vlm"]["n_perm"]))] for row in rows])
    scores = np.array([row["scores"] for row in rows])
    return ranks, calls, scores[:, 0] - scores[:, 1]


def metrics(rows, actions):
    ranks, calls, _ = arrays(rows)
    index = np.arange(len(rows))
    total = int(calls[index, actions].sum())
    return {**summarize_ranks(ranks[:, 0], ranks[index, actions]),
            "route_dist": dict(Counter(ACTIONS[a] for a in actions)),
            "vlm_calls": total, "vlm_calls_per_query": total / len(rows)}


def margin_actions(rows, fraction, quotas=None):
    _, calls, margins = arrays(rows)
    actions = np.zeros(len(rows), dtype=np.uint8)
    for cost in np.unique(calls[:, 2]):
        indices = np.flatnonzero(calls[:, 2] == cost)
        count = round(len(indices) * fraction) if quotas is None else quotas[int(cost)]
        indices = sorted(indices, key=lambda i: margins[i])
        actions[indices[:count]] = 2
    return actions


def features(rows, margin_only=False):
    s = np.array([r["scores"] for r in rows])
    if margin_only:
        return s[:, :1] - s[:, 1:2]
    p = np.exp(s[:, :10] - s[:, :1])
    p /= p.sum(1, keepdims=True)
    entropy = -(p * np.log(p + 1e-12)).sum(1) / np.log(10)
    return np.column_stack((s[:, :5], s[:, :1] - s[:, 1:5], entropy, p[:, 0],
                            [len(r["text"].split()) for r in rows]))


def fit_factorized(rows, kind):
    x = features(rows, margin_only=kind == "margin_logistic")
    correct = arrays(rows)[0] == 1

    def fit(mask, labels):
        if kind == "numeric_tree":
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                                   min_samples_leaf=40, l2_regularization=10,
                                                   random_state=42)
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
        return model.fit(x[mask], labels)

    models = [fit(np.ones(len(rows), dtype=bool), correct[:, 0])]
    for action in (1, 2):
        for coarse in (False, True):
            mask = correct[:, 0] == coarse
            models.append(fit(mask, correct[mask, action]))
    return models


def predict_utility(models, rows, kind, harm_weight):
    x = features(rows, margin_only=kind == "margin_logistic")
    p = [model.predict_proba(x)[:, 1] for model in models]
    return np.column_stack((np.zeros(len(rows)),
                            (1 - p[0]) * p[1] - harm_weight * p[0] * (1 - p[2]),
                            (1 - p[0]) * p[3] - harm_weight * p[0] * (1 - p[4])))


def profile(rows):
    ranks, calls, margins = arrays(rows)
    correct = ranks == 1
    result = {"n": len(rows), "outcome_correct_counts": dict(zip(ACTIONS, correct.sum(0).tolist())),
              "action_labels": dict(Counter(r["label_action"] for r in rows)) if "label_action" in rows[0] else None,
              "margin_quantiles_0_10_50_90_100": np.quantile(margins, [0, .1, .5, .9, 1]).tolist(),
              "top1_score_mean": float(np.mean([r["scores"][0] for r in rows])),
              "deep_call_costs": dict(Counter(calls[:, 2].tolist())),
              "outcomes": {}}
    for a, name in ((1, "light"), (2, "deep")):
        result["outcomes"][name] = {
            "fixes": int((~correct[:, 0] & correct[:, a]).sum()),
            "harms": int((correct[:, 0] & ~correct[:, a]).sum()),
            "unchanged": int((correct[:, 0] == correct[:, a]).sum())}
    return result


def test_rows():
    light = {r["qid"]: r for r in read_rows(STAGE / "pipeline/test_light.jsonl")}
    deep = {r["qid"]: r for r in read_rows(STAGE / "pipeline/test_deep_prompt.jsonl")}
    rows = read_rows(STAGE / "retrieval/test.jsonl")
    assert set(light) == set(deep) == {r["qid"] for r in rows}
    for row in rows:
        for cache in (light, deep):
            assert cache[row["qid"]]["text"] == row["text"]
            assert cache[row["qid"]]["base_rank"] == row["gt_rank"]
    return [{**r, "ranks": {"accept": r["gt_rank"], "light": light[r["qid"]]["new_rank"],
                            "deep": deep[r["qid"]]["new_rank"]},
             "light_scores": light[r["qid"]]["scores"], "deep_scores": deep[r["qid"]]["scores"],
             "hyps": deep[r["qid"]]["hyps"]} for r in rows]


def gallery_resampling(rows):
    queries = read_rows(ROOT / "data/msrvtt/queries_train.jsonl")
    gallery = [r for r in read_rows(ROOT / "data/msrvtt/videos.jsonl") if r["split"] == "train"]
    qi = {r["qid"]: i for i, r in enumerate(queries)}
    gi = {r["video_id"]: i for i, r in enumerate(gallery)}
    sim = np.load(STAGE / "clip4clip_train_sim.npy", mmap_mode="r")
    assert sim.shape == (len(queries), len(gallery))
    records = []
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        correct, margins = [], []
        for row in rows:
            index, truth = qi[row["qid"]], gi[row["gt"]]
            # Include GT and sample 999 other training videos; no test video is used.
            negatives = rng.choice(len(gallery) - 1, 999, replace=False)
            negatives += negatives >= truth
            scores = sim[index, np.concatenate(([truth], negatives))]
            top = np.partition(scores, -2)[-2:]
            # Match the original stable gallery-order tie break.
            correct.append(not bool(((scores[1:] > scores[0]) |
                                     ((scores[1:] == scores[0]) & (negatives < truth))).any()))
            margins.append(float(top.max() - top.min()))
        records.append({"seed": seed, "n": len(rows), "R@1": 100 * float(np.mean(correct)),
                        "margin_median": float(np.median(margins))})
    return {"protocol": "per-query GT plus 999 uniformly sampled non-GT training videos; stable gallery-order ties",
            "warning": "Retrieval-only diagnostic. Existing VLM labels do not apply to changed candidate sets.",
            "runs": records}


def main():
    allocate, source_hash = existing_allocator()
    # Verify exact-budget DP against complete enumeration on independent small examples.
    rng = np.random.default_rng(42)
    checked = 0
    for _ in range(20):
        utility = np.column_stack((np.zeros(5), rng.normal(size=(5, 2))))
        costs = np.tile([0, 2, 3], (5, 1))
        assignments = list(itertools.product(range(3), repeat=5))
        for budget in (4, 7, 10):
            feasible = [a for a in assignments if costs[np.arange(5), a].sum() == budget]
            actual = allocate(utility, costs, budget)
            expected = max(utility[np.arange(5), a].sum() for a in feasible)
            assert costs[np.arange(5), actual].sum() == budget
            assert np.isclose(utility[np.arange(5), actual].sum(), expected)
            checked += 1

    original = read_rows(STAGE / "router_data/router_data.jsonl")
    splits = {s: [r for r in original if r["split"] == s] for s in ("train", "dev", "calib")}
    report_b = json.loads((STAGE / "routing_experiment/gpu0-plan-b-yUV47YLg/report.json").read_text())
    fraction = report_b["deep_fraction"]
    quotas = {int(k): v for k, v in report_b["test_deep_quotas"].items()}
    train, dev, test = splits["train"], splits["dev"], test_rows()
    paths = [STAGE / "config.yaml", STAGE / "router_data/router_data.jsonl",
             STAGE / "retrieval/test.jsonl", STAGE / "pipeline/test_light.jsonl",
             STAGE / "pipeline/test_deep_prompt.jsonl"]
    report = {"input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "allocator_sha256": source_hash, "allocator_bruteforce_cases": checked,
              "profiles": {**{k: profile(v) for k, v in splits.items()}, "test": profile(test)},
              "diagnostic_models": {"train": "original train only", "selection": "dev only",
                                    "test": "fixed margin probe and dev-selected probe, no test tuning"}}
    for name, rows, quota in (("dev", dev, None), ("test", test, quotas)):
        actions = margin_actions(rows, fraction, quota)
        baseline = metrics(rows, actions)
        ranks, calls, _ = arrays(rows)
        correct = ranks == 1
        gains = correct.astype(float) - correct[:, :1]
        harms = correct[:, :1] & ~correct
        oracle = allocate(gains - 1e-6 * harms, calls, baseline["vlm_calls"])
        # Fixed deep quotas: best possible accept/deep allocation with the same group sizes.
        fixed = np.zeros(len(rows), dtype=np.uint8)
        for cost in np.unique(calls[:, 2]):
            pool = np.flatnonzero(calls[:, 2] == cost)
            count = int((actions[pool] == 2).sum())
            ordered = sorted(pool, key=lambda i: (gains[i, 2], -int(harms[i, 2])), reverse=True)
            fixed[ordered[:count]] = 2
        report[name] = {"margin": baseline, "always_deep": metrics(rows, np.full(len(rows), 2)),
                        "oracle_exact_budget_three_actions": metrics(rows, oracle),
                        "oracle_fixed_deep_quotas": metrics(rows, fixed),
                        "oracle_warning": "Uses GT outcomes; diagnostic upper bound, not an achievable model result."}
    assert np.isclose(report["test"]["margin"]["new"]["R@1"], report_b["test_margin"]["new"]["R@1"])
    assert report["test"]["margin"]["vlm_calls"] == report_b["test_margin"]["vlm_calls"]
    correct = arrays(test)[0] == 1
    report["test"]["outcome_headroom"] = {
        "any_path_correct": int(correct.any(1).sum()),
        "light_correct_base_and_deep_wrong": int((correct[:, 1] & ~correct[:, 0] & ~correct[:, 2]).sum()),
        "deep_harm_light_safe": int((correct[:, 0] & ~correct[:, 2] & correct[:, 1]).sum())}

    baseline = report["dev"]["margin"]
    candidates = []
    models = {}
    for kind in ("margin_logistic", "numeric_logistic", "numeric_tree"):
        models[kind] = fit_factorized(train, kind)
        for weight in (1.0, 2.0):
            utility = predict_utility(models[kind], dev, kind, weight)
            actions = allocate(utility, arrays(dev)[1], baseline["vlm_calls"])
            result = metrics(dev, actions)
            candidates.append({"kind": kind, "harm_weight": weight, "dev": result})

    def key(candidate):
        result = candidate["dev"]
        harms = result["flips"]["harmful_flips"]
        return (harms <= baseline["flips"]["harmful_flips"], result["new"]["R@1"], -harms)

    selected = max(candidates, key=key)
    report["diagnostic_models"].update(dev_candidates=candidates, dev_selection=selected)
    for label, kind, weight in (("fixed_margin_probe", "margin_logistic", 1.0),
                                ("dev_selected_probe", selected["kind"], selected["harm_weight"])):
        utility = predict_utility(models[kind], test, kind, weight)
        actions = allocate(utility, arrays(test)[1], report["test"]["margin"]["vlm_calls"])
        reference = margin_actions(test, fraction, quotas)
        ranks = arrays(test)[0]
        indices = np.arange(len(test))
        report["diagnostic_models"][label] = {
            "kind": kind, "harm_weight": weight, "test": metrics(test, actions),
            "different_actions_from_margin": int((actions != reference).sum()),
            "different_top1_correctness_from_margin": int(((ranks[indices, actions] == 1) !=
                                                           (ranks[indices, reference] == 1)).sum())}
    report["gallery_resampling"] = gallery_resampling(original)
    destination = Path(__file__).with_name("audit_results.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(destination)
    for split in ("dev", "test"):
        for name, value in report[split].items():
            if isinstance(value, dict) and "new" in value:
                print(split, name, value["new"]["R@1"], value["flips"]["harmful_flips"], value["vlm_calls"])
    print("diagnostic selection", selected["kind"], selected["harm_weight"])
    for name in ("fixed_margin_probe", "dev_selected_probe"):
        print(name, json.dumps(report["diagnostic_models"][name], ensure_ascii=False))
    print("gallery", report["gallery_resampling"]["runs"])


if __name__ == "__main__":
    main()
