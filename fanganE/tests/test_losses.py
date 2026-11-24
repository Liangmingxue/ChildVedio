import unittest

import numpy as np
import torch

from fanganE.losses import build_teacher_targets, compute_loss, masked_rank_kl


class TeacherTargetTests(unittest.TestCase):
    def test_teacher_and_route_targets_are_normalized(self):
        rows = [
            {
                "qid": "harm-case",
                "light_scores": [0.95, 0.10],
                "deep_scores": [0.10, 0.20, 0.30],
                "ranks": {"light": 1, "deep": 2},
            },
            {
                "qid": "fix-case",
                "light_scores": [0.10, 0.20],
                "deep_scores": [0.95, 0.10, 0.20],
                "ranks": {"light": 2, "deep": 1},
            },
        ]
        teacher, mask, route_soft, route_hard = build_teacher_targets(
            rows, np.asarray([0, 0]), num_candidates=4, temperature=0.25
        )
        self.assertEqual(teacher.shape, (2, 2, 4))
        self.assertTrue(mask[0, 0, :2].all())
        self.assertFalse(mask[0, 0, 2:].any())
        np.testing.assert_allclose(teacher.sum(axis=-1), np.ones((2, 2)), atol=1e-6)
        np.testing.assert_allclose(route_soft.sum(axis=-1), np.ones(2), atol=1e-6)
        self.assertGreater(route_soft[0, 0], route_soft[0, 2])
        self.assertGreater(route_soft[1, 2], route_soft[1, 0])
        self.assertEqual(route_hard.tolist(), [0, 2])

    def test_masked_kl_is_zero_for_matching_distribution(self):
        teacher = torch.tensor([[[0.7, 0.3, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]]])
        mask = torch.tensor([[[True, True, False, False], [True, True, False, False]]])
        student = torch.log(teacher.clamp_min(1e-8))
        loss = masked_rank_kl(student, teacher, mask, temperature=1.0)
        self.assertLess(float(loss), 1e-6)

    def test_composite_loss_backpropagates(self):
        torch.manual_seed(3)
        scores = torch.randn(2, 2, 4, requires_grad=True)
        outcomes = torch.randn(2, 3, requires_grad=True)
        teacher = torch.tensor(
            [
                [[0.7, 0.3, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0]],
                [[0.2, 0.8, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]],
            ]
        )
        mask = teacher > 0
        route_soft = torch.tensor([[0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        route_hard = torch.tensor([1, 2])
        gt = torch.tensor([0, 1])
        loss = compute_loss(
            scores,
            outcomes,
            gt,
            teacher,
            mask,
            route_soft,
            route_hard,
            {
                "rank_temperature": 1.0,
                "rank_kd_weight": 1.0,
                "rank_hard_weight": 0.25,
                "route_kd_weight": 0.5,
                "route_hard_weight": 0.25,
                "route_label_smoothing": 0.02,
            },
        )
        self.assertTrue(torch.isfinite(loss["total"]))
        loss["total"].backward()
        self.assertIsNotNone(scores.grad)
        self.assertIsNotNone(outcomes.grad)


if __name__ == "__main__":
    unittest.main()
