import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

from cvr.utils import Cfg
from fanganC.alignment import aligned_scores, shard_rankings
from fanganC.budget import structured_budget_loss
from fanganC.common import legacy
from fanganC.features import FeatureStore
from fanganC.model import CandidateRouter
from fanganC.training import train_variant


class AlignedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)

    def test_rankings_use_exactly_the_query_own_gallery(self):
        gallery = [f"v{i}" for i in range(6)]
        queries = [{"qid": f"q{i}", "video_id": video, "text": video} for i, video in enumerate(gallery)]
        matrix = np.broadcast_to(np.arange(6, dtype=np.float32), (6, 6))
        rankings, shards = shard_rankings(matrix, queries, gallery, 3, 2, 42)
        self.assertEqual(sorted(v for shard in shards for v in shard), gallery)
        for query in queries:
            shard = next(s for s in shards if query["video_id"] in s)
            ordered = sorted(shard, reverse=True)
            row = rankings[query["qid"]]
            self.assertEqual(row["cands"], ordered[:2])
            self.assertEqual(row["gt_rank"], ordered.index(query["video_id"]) + 1)

    def test_alignment_reuses_light_scores_but_requeries_changed_deep_prompts(self):
        cfg = Cfg(seed=42, rerank=Cfg(k_light=1, k_deep=2, empty_cf_fallback=True), vlm=Cfg(n_perm=2))
        old = {"cands": ["a", "b"], "hyps": ["fragile"], "light_scores": [.1], "deep_scores": [.4, .5]}
        row = {"text": "query", "cands": ["a", "c"]}
        verifier = SimpleNamespace(yes_prob=AsyncMock(return_value=.3), forced_choice=AsyncMock(return_value=.8))
        light, deep = asyncio.run(aligned_scores(verifier, cfg, row, ["auto result"], old))
        self.assertEqual(light, [.1])
        self.assertEqual(deep, [.8, .8])
        self.assertEqual([call.args[2] for call in verifier.forced_choice.await_args_list], ["a", "c"])
        verifier.yes_prob.assert_not_awaited()

    def test_budget_loss_penalizes_allocating_the_only_deep_slot_to_wrong_query(self):
        correct = np.array([[False, False, True], [True, False, False]])
        calls = np.array([[0, 1, 2], [0, 1, 2]])
        predicted = torch.tensor([[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [0., 1., 0.]]], requires_grad=True)
        loss = structured_budget_loss(predicted, correct, calls, .5, [1.])
        loss.backward()
        self.assertGreater(loss.item(), 0.)
        self.assertLess(predicted.grad[0, 1, 1].item(), 0.)
        self.assertGreater(predicted.grad[1, 1, 1].item(), 0.)

    def test_temporal_context_reads_frame_order_and_ignores_padding(self):
        head = CandidateRouter(4, 4, np.zeros((2, 3)), np.array([0., 1.]), [False, True],
                               hidden_dim=4, pointer_dim=4, temporal_candidates=True)
        query = torch.randn(1, 4)
        frames = torch.randn(1, 2, 4, 4)
        frame_mask = torch.tensor([[[True, True, True, False], [True, True, False, False]]])
        mask = torch.tensor([[True, False]])
        expected = head.temporal_context(query, frames, frame_mask, mask)
        masked = frames.clone()
        masked[0, 0, 3] = torch.nan
        masked[0, 1] = torch.nan
        torch.testing.assert_close(head.temporal_context(query, masked, frame_mask, mask), expected)
        reversed_frames = frames.clone()
        reversed_frames[0, 0, :3] = frames[0, 0, :3].flip(0)
        self.assertFalse(torch.allclose(head.temporal_context(query, reversed_frames, frame_mask, mask), expected))

    def test_joint_stops_after_three_half_epoch_validations_and_keeps_one_checkpoint(self):
        store = FeatureStore.__new__(FeatureStore)
        store.config = Cfg(seed=42, hidden_dim=4, pointer_dim=4, dropout=0., warmup_epochs=1, joint_epochs=8,
                           warmup_lr=3e-4, lora_lr=2e-5, head_lr=1e-4, weight_decay=.01, warmup_ratio=.03,
                           effective_batch=2, microbatch=1, eval_batch_size=2, deep_fraction=.5, harm_weights=[1., 2.],
                           select_joint_only=True, early_stopping_patience=3, budget_loss_weight=1.)
        store.cfg = Cfg(router=Cfg(base_model="local", lora_r=2, lora_alpha=4, max_len=32),
                        rerank=Cfg(k_light=1, k_deep=2))
        store.manifest = {"splits": {"train": 4, "dev": 2}}
        store.rows = [None] * 6
        store.hidden = np.ones((6, 4), dtype=np.float32)
        store.query = np.ones((6, 2), dtype=np.float32)
        store.numeric = np.ones((6, 2), dtype=np.float32)
        store.margins = np.arange(6, dtype=np.float32)
        store.correct = np.array([[True, True, False], [False, False, True]] * 3)
        store.train_indices, store.dev_indices = [0, 1, 2, 3], [4, 5]

        class Backbone(torch.nn.Linear):
            def __init__(self):
                super().__init__(4, 4)
                self.config = SimpleNamespace(use_cache=True)

            def save_pretrained(self, path):
                path.mkdir()
                (path / "adapter_model.safetensors").write_bytes(b"adapter fixture")
                (path / "adapter_config.json").write_text("{}")

        baseline = {"new": {"R@1": 50.}, "flips": {"harmful_flips": 1}}

        def replay(_, indices):
            return SimpleNamespace(calls=np.tile([0, 1, 2], (len(indices), 1)),
                                   margin_actions=lambda fraction: [0, 2], metrics=lambda actions: baseline)

        def field_states(model, formatter, encoded, device):
            return model, model(torch.ones((len(encoded), 1, 4)))

        with (TemporaryDirectory() as folder, patch.multiple("fanganC.training", FeatureStore=lambda path: store,
              replay=replay, experiment_signature=lambda store: "fixture", encode=lambda rows, cfg, fmt: rows,
              load_router=lambda *args, **kwargs: (None, Backbone()), JevFormatter=lambda *args, **kwargs: None,
              field_states=field_states), patch.object(legacy(), "eval_policy", return_value=baseline)):
            report = train_variant(folder, "text_only", "cpu")
            self.assertEqual(report["selection"]["phase"], "joint")
            self.assertEqual((report["selection"]["epoch"], report["selection"]["step"]), (0, 1))
            self.assertEqual(sum(r["phase"] == "joint" for r in report["validations"]), 4)
            self.assertTrue(report["early_stopping"]["triggered"])
            self.assertEqual(sorted(p.name for p in (Path(folder) / "text_only").glob("best*")), ["best"])


if __name__ == "__main__":
    unittest.main()
