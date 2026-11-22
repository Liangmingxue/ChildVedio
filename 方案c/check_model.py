"""CPU structural checks using the real tokenizer and a tiny random Qwen3 backbone.

This checks the production functions, not the quality of the trained 4B checkpoint.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data/models/router-python-deps"), str(ROOT)]

import numpy as np
import peft
import torch
import transformers
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from cvr.router import JevFormatter, _collate, build_state, field_states, load_router


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    spec = importlib.util.spec_from_file_location("stage2_experiment_audit", ROOT / "scripts/13_stage2_router_experiment.py")
    experiment = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = experiment
    spec.loader.exec_module(experiment)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/models/qwen3-4b", local_files_only=True)
    formatter = JevFormatter(tokenizer)
    tokens = {text: tokenizer.encode(text, add_special_tokens=False)
              for text in ["decision", *"ABCDEFGHIJ"]}
    assert all(len(ids) == 1 for ids in tokens.values())
    rows = [json.loads(line) for line in (ROOT / "outputs/stage2_clip4clip_qwen3vl8b/router_data/router_data.jsonl").read_text().splitlines()]
    encoded = [formatter.encode(build_state(row, 1.0000000762939454, 10)) for row in rows]
    lengths = [len(ids) for ids, _ in encoded]
    assert all(ids[pos + 1] == formatter.marker_id for ids, positions in encoded for pos in positions)
    assert max(lengths) < formatter.max_len

    config = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, attention_dropout=0.0, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    tiny = Qwen3ForCausalLM(config).to(torch.bfloat16)
    original = copy.deepcopy(tiny.state_dict())
    # Exercise load_router's existing PEFT, checkpointing and gradient setup.
    with patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer), \
            patch.object(AutoModelForCausalLM, "from_pretrained", return_value=tiny):
        _, model = load_router("tiny-cpu-probe", device="cpu", train=True, lora_r=4, lora_alpha=8)
    model.config.use_cache = False
    head = experiment.FactorizedGainHead(32, np.array([.5, 1.5, 2.5, 3.5]))
    correct = torch.tensor([[False, True, False], [True, False, True],
                            [False, False, True], [True, True, False]])
    margins = torch.tensor([.5, 1.5, 2.5, 3.5])
    head.fit_margin_prior(margins.numpy(), correct[:, 0])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), lr=.001)
    gradients = []
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        base, states = field_states(model, formatter, encoded[:4], "cpu")
        logits = head(states[:, 0], margins)
        fragile_logits = base.lm_head(states[:, 1]).float()[:, formatter.cand_ids[1]]
        loss = head.loss(logits, correct) + .5 * torch.nn.functional.cross_entropy(fragile_logits, torch.arange(4))
        loss.backward()
        grad = {name: float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
                if "lora_" in name and parameter.grad is not None}
        assert grad and all(np.isfinite(v) for v in grad.values())
        assert any(value > 0 for name, value in grad.items() if "lora_B" in name)
        if step:
            assert any(value > 0 for name, value in grad.items() if "lora_A" in name)
        assert head.linear.weight.grad.norm() > 0
        gradients.append({"step": step, "loss": float(loss.detach()),
                          "nonzero_lora_A": sum(v > 0 for n, v in grad.items() if "lora_A" in n),
                          "nonzero_lora_B": sum(v > 0 for n, v in grad.items() if "lora_B" in n)})
        optimizer.step()

    model.eval()
    head.eval()
    with torch.inference_mode():
        _, active = field_states(model, formatter, encoded[:4], "cpu")
        with model.disable_adapter():
            _, disabled = field_states(model, formatter, encoded[:4], "cpu")
        adapter_effect = float((active.float() - disabled.float()).abs().max())
        assert adapter_effect > 0
        ids, mask, pos = _collate(encoded[:4], tokenizer.pad_token_id, "cpu")
        # A later placeholder cannot influence the causal action readout.
        changed = ids.clone()
        changed[torch.arange(4), pos[:, 0] + 1] = tokenizer.encode("different", add_special_tokens=False)[0]
        backbone = model.get_base_model()
        before = backbone.model(input_ids=ids, attention_mask=mask).last_hidden_state
        after = backbone.model(input_ids=changed, attention_mask=mask).last_hidden_state
        causal_difference = float((before[torch.arange(4), pos[:, 0]] -
                                   after[torch.arange(4), pos[:, 0]]).abs().max())
        assert causal_difference == 0
        arbitrary_logits = torch.randn(100, 5)
        probability = head.probabilities(arbitrary_logits)
        assert probability.min() >= 0
        torch.testing.assert_close(probability.sum(-1), torch.ones(100, 2))
        coarse = arbitrary_logits[:, :1].sigmoid()
        cond = arbitrary_logits[:, 1:].reshape(-1, 2, 2).sigmoid()
        expected_success = (1 - coarse) * cond[:, :, 0] + coarse * cond[:, :, 1]
        torch.testing.assert_close(probability[:, :, 1] - probability[:, :, 2], expected_success - coarse)

    with tempfile.TemporaryDirectory(prefix="childvedio-model-audit-") as folder:
        model.save_pretrained(folder)
        torch.save(head.state_dict(), Path(folder) / "gain_head.pt")
        restored_base = Qwen3ForCausalLM(config).to(torch.bfloat16)
        restored_base.load_state_dict(original)
        with patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer), \
                patch.object(AutoModelForCausalLM, "from_pretrained", return_value=restored_base):
            _, restored = load_router("tiny-cpu-probe", adapter=folder, device="cpu")
        restored.config.use_cache = False
        restored_head = experiment.FactorizedGainHead(32, np.array([0.0]))
        restored_head.load_state_dict(torch.load(Path(folder) / "gain_head.pt", weights_only=True))
        with torch.inference_mode():
            _, restored_states = field_states(restored, formatter, encoded[:4], "cpu")
            before = head.probabilities(head(active[:, 0], margins))
            after = restored_head.probabilities(restored_head(restored_states[:, 0], margins))
            roundtrip_difference = float((before - after).abs().max())
            torch.testing.assert_close(before, after, atol=0, rtol=0)

    trained_head = torch.load(ROOT / "outputs/stage2_clip4clip_qwen3vl8b/routing_experiment/gpu0-plan-b-yUV47YLg/gain_epoch_2/gain_head.pt",
                              map_location="cpu", weights_only=True)
    checkpoint = ROOT / "outputs/stage2_clip4clip_qwen3vl8b/routing_experiment/gpu0-plan-b-yUV47YLg/gain_epoch_2/adapter_model.safetensors"
    adapter_norms = {"lora_A": [], "lora_B": []}
    with safe_open(checkpoint, framework="pt", device="cpu") as saved:
        for name in saved.keys():
            value = saved.get_tensor(name)
            assert torch.isfinite(value).all()
            for kind in adapter_norms:
                if kind in name:
                    adapter_norms[kind].append(float(value.float().norm()))
    assert adapter_norms["lora_B"] and min(adapter_norms["lora_B"]) > 0
    adapter_summary = {kind: {"count": len(values), "nonzero_count": sum(v > 0 for v in values),
                              "min_norm": min(values), "max_norm": max(values)}
                       for kind, values in adapter_norms.items()}
    report = {"scope": "Real tokenizer and production functions; tiny randomly initialized 2-layer Qwen3, not pretrained 4B inference",
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
              "cuda_available": torch.cuda.is_available(), "token_ids": tokens,
              "training_input_lengths": {"n": len(lengths), "min": min(lengths), "max": max(lengths),
                                         "mean": float(np.mean(lengths)), "max_len": formatter.max_len},
              "gradients": gradients, "adapter_effect_max_abs_hidden": adapter_effect,
              "causal_action_readout_difference": causal_difference,
              "probability_and_gain_formula_checks": "passed",
              "save_reload_max_abs_probability_difference": roundtrip_difference,
              "saved_4B_adapter_tensor_checks": adapter_summary,
              "trained_B_head": {k: v.tolist() for k, v in trained_head.items() if v.numel() <= 10},
              "trained_B_head_weight_row_norms": trained_head["linear.weight"].norm(dim=1).tolist()}
    destination = Path(__file__).with_name("model_checks.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
