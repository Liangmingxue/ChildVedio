"""Frozen-selection evaluation at the existing exact VLM call budget."""
import gc
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, atomic_rows, legacy, read_json
from .features import FeatureStore
from .training import load_selected, predict, replay


def probability_report(head, logits, correct):
    tensor = torch.as_tensor(logits)
    targets = torch.as_tensor(correct)
    coarse = tensor[:, 0].sigmoid()
    conditional = tensor[:, 1:].reshape(-1, 2, 2).sigmoid()
    observed = conditional.gather(2, targets[:, :1].long()[:, None].expand(-1, 2, 1)).squeeze(2)
    probabilities = torch.cat((coarse[:, None], observed), 1)
    return {"conditional_loss": float(head.loss(tensor, targets)),
            "brier_per_output": ((probabilities - targets.float()) ** 2).mean(0).tolist(),
            "n": len(correct), "temperature": 1.0, "used_for_selection": False}


def evaluate_variant(run_dir, variant, device):
    store = FeatureStore(run_dir)
    folder = Path(run_dir) / variant
    if (folder / "evaluation.json").exists():
        # Validate signature/checkpoint even when reusing an already completed report.
        head, backbone, _, _, report = load_selected(run_dir, variant, store, device)
        del head, backbone
        return read_json(folder / "evaluation.json")
    # All variants must be selected before any test metric is exposed.
    for name in store.config.variants:
        if not (Path(run_dir) / name / "training_complete.json").exists():
            raise ValueError("Finish development selection for both variants before testing")
    head, backbone, formatter, encoded, training = load_selected(run_dir, variant, store, device)
    logits, elapsed = predict(store, head, store.test_indices, device, backbone, formatter, encoded)
    probabilities = head.probabilities(torch.as_tensor(logits)).detach().numpy()
    weight = training["selection"]["harm_weight"]
    utilities = legacy().gain_utility(probabilities, weight)
    test = replay(store, store.test_indices)
    budget = read_json(Path(run_dir) / "snapshot/budget.json")
    quotas = {int(k): v for k, v in budget["deep_query_count_by_vlm_calls"].items()}
    reference = test.margin_actions(store.config.deep_fraction, quotas)
    baseline = test.metrics(reference)
    if baseline["vlm_calls"] != budget["vlm_calls"]:
        raise ValueError("Reconstructed baseline uses a different budget")
    actions = legacy().allocate(utilities, test.calls, baseline["vlm_calls"])
    result = test.metrics(actions)
    indices = np.arange(len(actions))
    old_correct = test.ranks[indices, reference] == 1
    new_correct = test.ranks[indices, actions] == 1
    per_query = []
    for i, global_index in enumerate(store.test_indices):
        row = store.rows[global_index]
        per_query.append({"qid": row["qid"], "text": row["text"], "logits": logits[i].tolist(),
                          "probabilities_light_deep_unchanged_fix_harm": probabilities[i].tolist(),
                          "utilities": utilities[i].tolist(), "cached_call_costs": test.calls[i].tolist(),
                          "action": legacy().ACTIONS[actions[i]], "margin_action": legacy().ACTIONS[reference[i]],
                          "base_rank": int(test.ranks[i, 0]), "new_rank": int(test.ranks[i, actions[i]]),
                          "margin_rank": int(test.ranks[i, reference[i]])})
    calibration_logits, _ = predict(store, head, store.calib_indices, device, backbone, formatter, encoded)
    report = {"variant": variant, "evaluation": "offline replay of frozen existing paths; no new VLM calls",
              "selection": training["selection"], "test_margin": baseline, "test_selected": result,
              "routing_seconds_per_query": elapsed / len(actions),
              "different_actions": int((actions != reference).sum()),
              "paired_improved": int((~old_correct & new_correct).sum()),
              "paired_worsened": int((old_correct & ~new_correct).sum()),
              "beats_margin_under_harm_constraint": bool(result["new"]["R@1"] > baseline["new"]["R@1"] and
                  result["flips"]["harmful_flips"] <= baseline["flips"]["harmful_flips"]),
              "calibration": probability_report(head, calibration_logits, store.correct[store.calib_indices])}
    atomic_rows(folder / "test_predictions.jsonl", per_query)
    atomic_json(folder / "evaluation.json", report)
    print(f"{variant} TEST R1={result['new']['R@1']:.3f} harms={result['flips']['harmful_flips']} calls={result['vlm_calls']}", flush=True)
    del backbone, head, store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
