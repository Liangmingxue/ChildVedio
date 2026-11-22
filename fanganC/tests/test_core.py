import copy
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from fanganC.__main__ import main
from fanganC.common import atomic_json, legacy, read_json, require_local_output, settings, snapshot_config
from fanganC.data import merge_completed, published_rows
from fanganC.features import FeatureStore, clip_features, numeric_features, text_features
from fanganC.model import CandidateRouter
from fanganC.training import checkpoint_hashes, experiment_signature, load_selected, optimizer_groups, predict
from cvr.utils import Cfg


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def make_head(self, variant="candidate", **kwargs):
        return CandidateRouter(16, 8, np.zeros((4, 12)), np.arange(4.),
                               np.array([0, 1, 0, 1]), variant, hidden_dim=8, pointer_dim=4, dropout=0., **kwargs)

    def batch(self):
        return {"query": torch.randn(4, 8), "videos": torch.randn(4, 7, 8),
                "mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0]] * 4).bool(),
                "numeric": torch.zeros(4, 12), "margin": torch.arange(4.)}

    def test_candidate_padding_does_not_change_prediction_or_gradient(self):
        head = self.make_head().eval()
        batch = self.batch()
        states = torch.randn(4, 16)
        baseline = head(states, batch)
        changed = copy.deepcopy(batch)
        changed["videos"][:, 5:] = float("nan")
        changed["videos"].requires_grad_()
        actual = head(states, changed)
        torch.testing.assert_close(actual, baseline)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(changed["videos"].grad).all())
        self.assertEqual(int(changed["videos"].grad[:, 5:].count_nonzero()), 0)

    def test_candidate_evidence_changes_outputs_and_text_control_ignores_it(self):
        states, batch = torch.randn(4, 16), self.batch()
        changed = copy.deepcopy(batch)
        changed["videos"] *= -1
        head = self.make_head().eval()
        self.assertGreater(float((head(states, batch) - head(states, changed)).abs().max().detach()), 1e-5)
        head = self.make_head("text_only").eval()
        torch.testing.assert_close(head(states, batch), head(states, changed), atol=0, rtol=0)

    def test_candidate_context_respects_both_verification_limits(self):
        head = self.make_head(k_light=6, k_deep=3).eval()
        batch = self.batch()
        batch["mask"].fill_(True)
        videos = batch["videos"]
        scores = head.pointer(videos, batch["query"])
        light_context = (scores[:, :6].softmax(-1)[..., None] * videos[:, :6]).sum(1)
        deep_context = (scores[:, :3].softmax(-1)[..., None] * videos[:, :3]).sum(1)
        with patch.object(head.evidence, "forward", wraps=head.evidence.forward) as evidence:
            head(torch.randn(4, 16), batch)
        used = evidence.call_args.args[0]
        torch.testing.assert_close(used[:, 16:24], videos[:, 0] - deep_context)
        torch.testing.assert_close(used[:, 24:], light_context - deep_context)

    def test_text_only_predict_skips_candidate_arrays(self):
        store = FeatureStore.__new__(FeatureStore)
        store.config = SimpleNamespace(eval_batch_size=2)
        store.hidden = np.zeros((4, 16), dtype=np.float32)
        store.numeric = np.zeros((4, 12), dtype=np.float32)
        store.margins = np.arange(4, dtype=np.float32)
        logits, _ = predict(store, self.make_head("text_only"), [0, 1, 2, 3], "cpu")
        self.assertEqual(logits.shape, (4, 5))

    def test_probability_gain_identity_and_roundtrip(self):
        head = self.make_head().eval()
        logits = torch.randn(20, 5)
        p = head.probabilities(logits)
        self.assertGreaterEqual(float(p.min()), 0.)
        torch.testing.assert_close(p.sum(-1), torch.ones(20, 2))
        coarse, conditional = logits[:, :1].sigmoid(), logits[:, 1:].reshape(-1, 2, 2).sigmoid()
        torch.testing.assert_close(p[:, :, 1] - p[:, :, 2],
                                  (1 - coarse) * conditional[:, :, 0] + coarse * conditional[:, :, 1] - coarse)
        clone = self.make_head().eval()
        clone.load_state_dict(head.state_dict())
        states, batch = torch.randn(4, 16), self.batch()
        torch.testing.assert_close(head(states, batch), clone(states, batch), atol=0, rtol=0)

    def test_numeric_input_does_not_read_labels(self):
        cfg = Cfg(retriever=Cfg(stat_temp=1., stat_k=10))
        row = {"text": "a person runs", "scores": list(np.linspace(10, 1, 16)), "gt": "secret", "gt_rank": 1,
               "ranks": {"accept": 1, "light": 9, "deep": 2}, "neg_caption": "secret caption"}
        changed = {**row, "gt": "other", "gt_rank": 999, "ranks": {}, "neg_caption": "other"}
        np.testing.assert_array_equal(numeric_features([row], cfg), numeric_features([changed], cfg))

    def test_clip_replays_original_batches_when_retrying_an_old_run(self):
        queries = {split: [{"qid": f"{split}-{i}", "text": str(i)} for i in range(size)]
                   for split, size in (("train", 65), ("test", 3))}

        def embedding(values):
            return torch.stack((values.float() + 1, torch.full_like(values, len(values)).float()), dim=-1)

        def expected_score(query, query_batch, video, video_batch):
            q = torch.tensor([query + 1, query_batch], dtype=torch.float32)
            v = torch.tensor([video + 1, video_batch], dtype=torch.float32)
            return float(100 * torch.dot(q / q.norm(), v / v.norm()))

        rows = []
        cases = [("train", 64, 1, 4, 16), ("train", 10, 64, 16, 1), ("test", 1, 3, 1, 3)]
        for split, query, query_batch, video, video_batch in cases:
            rows.append({**queries[split][query], "cands": [f"{split}-{video}"],
                         "scores": [expected_score(query, query_batch, video, video_batch)]})

        model = Mock()
        model.clip.logit_scale = torch.tensor(100.).log()
        model.clip.encode_text.side_effect = lambda tokens: embedding(tokens[:, 1])
        model.get_visual_output.side_effect = lambda frames, mask, **_: embedding(frames).unsqueeze(1)
        model._mean_pooling_for_similarity_visual.side_effect = lambda visual, mask: visual[:, 0]
        tokenizer = SimpleNamespace(encoder={"<|startoftext|>": 1000, "<|endoftext|>": 1001},
                                    encode=lambda text: [int(text)])
        upstream = SimpleNamespace(load_model=Mock(return_value=model), SimpleTokenizer=lambda: tokenizer,
                                   cv2=Mock(), MAX_WORDS=4, MAX_FRAMES=1,
                                   Gallery=lambda videos: [(torch.tensor([v["value"]]), torch.tensor([True]))
                                                           for v in videos])

        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            features = run_dir / "features"
            features.mkdir()
            old_query_file = features / "clip_queries.npz"
            old_query_file.write_bytes(b"legacy cache")
            manifest = features / "clip_manifest.json"
            manifest.write_text(json.dumps({"signature": "legacy", "complete": False}))
            videos = [{"split": split, "video_id": f"{split}-{i}", "path": str(old_query_file), "value": i}
                      for split, size in (("train", 17), ("test", 3)) for i in range(size)]
            inputs = {"videos.jsonl": videos, **{f"queries_{s}.jsonl": q for s, q in queries.items()}}
            for name, records in inputs.items():
                (run_dir / name).write_text("".join(json.dumps(r) + "\n" for r in records))
            config = SimpleNamespace(workers=0, video_batch_size=8, temporal_candidates=True)
            cfg = SimpleNamespace(paths=SimpleNamespace(data_dir=run_dir))
            with patch.multiple("fanganC.features", module_from_file=Mock(return_value=upstream),
                                clip_source_identity=Mock(return_value={}), snapshot_config=Mock(return_value=(config, cfg)),
                                frozen_rows=Mock(return_value=(rows[:2], rows[2:], {"frozen_sha256": {}}))):
                clip_features(run_dir, "cpu")
                clip_features(run_dir, "cpu")

            self.assertEqual([call.args[0][:, 1].tolist() for call in model.clip.encode_text.call_args_list],
                             [list(range(64)), [64], [0, 1, 2]])
            self.assertEqual([call.args[0].tolist() for call in model.get_visual_output.call_args_list],
                             [list(range(16)), [16], [0, 1, 2]])
            upstream.load_model.assert_called_once_with("cpu")
            self.assertEqual(old_query_file.read_bytes(), b"legacy cache")
            self.assertLess(json.loads(manifest.read_text())["score_max_abs_error"], 1e-4)
            with np.load(features / "clip.npz") as saved:
                self.assertEqual(saved["qids"].tolist(), [r["qid"] for r in rows])
                self.assertEqual(saved["video_ids"].tolist(), sorted(r["cands"][0] for r in rows))
                self.assertEqual(saved["frames"].shape, (3, 1, 2))
                self.assertEqual(saved["frames"].dtype, np.float16)
                self.assertTrue(saved["frame_mask"].all())

    def test_text_cache_reassembles_missing_output_from_chunks(self):
        rows = [{"qid": "train-0"}]
        cfg = SimpleNamespace(router=SimpleNamespace(base_model="local", max_len=32))
        config = SimpleNamespace(eval_batch_size=1)
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=2, use_cache=False))
        states = torch.tensor([[[1., 2.]]])
        field_states = Mock(return_value=(model, states))
        with tempfile.TemporaryDirectory() as folder, patch.multiple(
                "fanganC.features", snapshot_config=Mock(return_value=(config, cfg)),
                frozen_rows=Mock(return_value=(rows, [], {"frozen_sha256": {}})),
                backbone_signature=Mock(return_value={}), load_router=Mock(return_value=(None, model)),
                JevFormatter=Mock(), encode=Mock(return_value=[None]), field_states=field_states):
            output = Path(folder) / "features/text.npy"
            text_features(folder, "cpu")
            output.unlink()
            text_features(folder, "cpu")
            field_states.assert_called_once()
            np.testing.assert_array_equal(np.load(output), [[1., 2.]])

    def test_snapshot_prefix_frozen_duplicates_and_split_leakage(self):
        base = [{"qid": "a", "gt": "v1", "split": "dev"}]
        extra = [{"qid": "b", "gt": "v2", "split": "train"}]
        self.assertEqual(len(merge_completed(base, extra)), 2)
        with self.assertRaises(ValueError):
            merge_completed(base, [{"qid": "b", "gt": "v1", "split": "train"}])
        with self.assertRaises(ValueError):
            merge_completed(base, extra * 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.jsonl"
            path.write_text('{"qid": "first"}\n')
            rows, checksum = published_rows(path)
            replacement = path.with_suffix(".part")
            replacement.write_text('{"qid": "second"}\n')
            replacement.replace(path)
            self.assertEqual(rows, [{"qid": "first"}])
            self.assertNotEqual(checksum, published_rows(path)[1])
            path.write_text('{"qid":')
            with self.assertRaises(ValueError):
                published_rows(path)

    def test_exact_budget_matches_exhaustive_assignments(self):
        rng = np.random.default_rng(42)
        calls = np.tile([0, 2, 3], (4, 1))
        for _ in range(10):
            utility = np.column_stack((np.zeros(4), rng.normal(size=(4, 2))))
            actions = legacy().allocate(utility, calls, 7)
            expected = max(utility[np.arange(4), a].sum() for a in itertools.product(range(3), repeat=4)
                           if calls[np.arange(4), a].sum() == 7)
            self.assertEqual(calls[np.arange(4), actions].sum(), 7)
            self.assertAlmostEqual(utility[np.arange(4), actions].sum(), expected)
        with self.assertRaises(ValueError):
            legacy().allocate(np.zeros((1, 3)), np.array([[0, 2, 4]]), 3)

    def test_optimizer_groups_and_output_boundary(self):
        head = self.make_head()
        backbone = torch.nn.Linear(3, 3)
        cfg = Cfg(warmup_lr=3e-4, head_lr=1e-4, lora_lr=2e-5, weight_decay=.01)
        groups = optimizer_groups(head, backbone, cfg, "joint")
        params = [p for group in groups for p in group["params"]]
        self.assertEqual(len(params), len(set(map(id, params))))
        self.assertEqual({g["lr"] for g in groups}, {1e-4, 2e-5})
        for group in groups:
            if group["weight_decay"]:
                self.assertTrue(all(p.ndim >= 2 for p in group["params"]))
        with self.assertRaises(ValueError):
            require_local_output("outputs/not-allowed")

    def test_snapshot_preserves_scientific_notation_learning_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "snapshot"
            atomic_json(snapshot / "experiment.yaml", {"lora_lr": 2e-5})
            atomic_json(snapshot / "stage.yaml", {"router": {"lr": 2e-5}})
            config, stage = snapshot_config(folder)
            self.assertEqual(config.lora_lr, 2e-5)
            self.assertEqual(stage.router.lr, 2e-5)

    def test_custom_and_old_snapshot_learning_rates_reach_scheduler_as_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            custom = path / "custom.yaml"
            custom.write_text("stage_config: stage.yaml\nwarmup_lr: 3e-4\nhead_lr: 1e-4\nlora_lr: 2e-5\n")
            self.assertEqual(settings(custom).lora_lr, 2e-5)
            atomic_json(path / "snapshot/experiment.yaml", {"warmup_lr": "3e-4", "head_lr": "1e-4",
                                                            "lora_lr": "2e-5", "weight_decay": .01})
            atomic_json(path / "snapshot/stage.yaml", {})
            config, _ = snapshot_config(path)
            groups = optimizer_groups(self.make_head(), torch.nn.Linear(3, 3), config, "joint")
            scheduler = get_cosine_schedule_with_warmup(torch.optim.AdamW(groups), 2, 10)
            self.assertEqual(set(scheduler.base_lrs), {1e-4, 2e-5})

    def test_selected_checkpoint_rejects_changed_adapter_config(self):
        with tempfile.TemporaryDirectory() as folder:
            training = Path(folder) / "candidate"
            checkpoint = training / "best"
            adapter = checkpoint / "adapter"
            adapter.mkdir(parents=True)
            (checkpoint / "head.pt").write_bytes(b"head")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            atomic_json(adapter / "adapter_config.json", {"lora_alpha": 64})
            atomic_json(training / "training_complete.json", {"signature": "experiment", "selection": {"phase": "joint"},
                        "checkpoint_sha256": checkpoint_hashes(checkpoint, "joint")})
            atomic_json(adapter / "adapter_config.json", {"lora_alpha": 32})
            with patch("fanganC.training.experiment_signature", return_value="experiment"):
                with self.assertRaisesRegex(ValueError, "Selected checkpoint files changed"):
                    load_selected(folder, "candidate", None, "cpu")

    def test_evaluating_one_variant_preserves_other_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            summary = path / "results.json"
            atomic_json(summary, {"text_only": {"result": 1}, "candidate": {"result": 2}})
            config = SimpleNamespace(variants=["text_only", "candidate"])
            args = ["fanganC", "evaluate", "--run-dir", folder, "--variant", "candidate"]
            with (patch.object(sys, "argv", args),
                  patch("fanganC.__main__.require_local_output", return_value=path),
                  patch("fanganC.__main__.snapshot_config", return_value=(config, None)),
                  patch("fanganC.evaluation.evaluate_variant", return_value={"result": 3})):
                main()
            self.assertEqual(read_json(summary), {"text_only": {"result": 1}, "candidate": {"result": 3}})

    def test_single_variant_run_checks_other_training_before_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            config = SimpleNamespace(variants=["text_only", "candidate"])
            args = ["fanganC", "run", "--run-dir", folder, "--variant", "candidate"]
            with (patch.object(sys, "argv", args),
                  patch("fanganC.__main__.require_local_output", return_value=path),
                  patch("fanganC.__main__.snapshot_config", return_value=(config, None)),
                  patch("fanganC.data.create_snapshot"), patch("fanganC.features.clip_features") as features):
                with self.assertRaisesRegex(ValueError, "Run all configured variants before evaluation"):
                    main()
                features.assert_not_called()

    def test_training_signature_binds_feature_provenance(self):
        store = SimpleNamespace(manifest={"frozen_sha256": {"data": "fixed"}}, config={},
                                feature_manifests={"clip": {"signature": "encoder-v1", "sha256": "data-v1"},
                                                   "text": {"signature": "qwen-v1", "sha256": "states-v1"}})
        before = experiment_signature(store)
        store.feature_manifests["clip"]["signature"] = "encoder-v2"
        self.assertNotEqual(before, experiment_signature(store))
        store.feature_manifests["clip"]["signature"] = "encoder-v1"
        store.feature_manifests["text"]["sha256"] = "states-v2"
        self.assertNotEqual(before, experiment_signature(store))


if __name__ == "__main__":
    unittest.main()
