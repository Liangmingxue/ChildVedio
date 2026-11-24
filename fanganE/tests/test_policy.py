import unittest

import numpy as np

from fanganE.policy import allocate_budget, rank_from_path


class PolicyTests(unittest.TestCase):
    def test_budget_allocator_respects_incremental_costs(self):
        priority = np.asarray([0.1, 0.9, 0.5, 0.2])
        costs = np.asarray([[0, 5, 32]] * 4)
        gallery = np.asarray([0, 0, 0, 0])
        chosen = allocate_budget(priority, costs, gallery)
        self.assertEqual(chosen.tolist(), [False, True, True, False])
        calls = int(costs[:, 1].sum() + costs[chosen, 2].sum())
        self.assertLessEqual(calls, 111)

    def test_rank_reconstruction_keeps_outside_topk_rank(self):
        scores = np.asarray(
            [
                [[0.1, 0.9, 0.0], [0.8, 0.1, 0.0]],
                [[0.2, 0.1, 0.7], [0.1, 0.8, 0.2]],
            ]
        )
        chosen = np.asarray([False, True])
        gt = np.asarray([1, -1])
        original = np.asarray([4, 19])
        ranks = rank_from_path(scores, chosen, gt, original)
        self.assertEqual(ranks.tolist(), [1, 19])


if __name__ == "__main__":
    unittest.main()
