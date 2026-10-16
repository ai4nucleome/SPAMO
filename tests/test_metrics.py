import unittest

import numpy as np

from spamo.metrics import compute


class MetricTests(unittest.TestCase):
    def test_perfect_partition(self):
        truth = np.array([1, 1, 2, 2, 3, 3])
        pred = np.array([3, 3, 1, 1, 2, 2])                            # label permutation
        for name, value in compute(pred, truth).items():
            self.assertAlmostEqual(value, 1.0, msg=name)

    def test_known_values(self):
        truth = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0, 0, 1, 1, 1, 1])
        m = compute(pred, truth)
        # pairs: tp=3 (0,1),(0,2)? -> computed by hand: tp=1+3=4? verify through the contingency table
        # table rows pred, cols truth: [[2,0],[1,3]] -> tp=1+0+0+3=4, fp=C(2,2)+C(4,2)-4=3, fn=C(3,2)*2-4=2
        self.assertAlmostEqual(m['Jaccard'], 4 / (4 + 3 + 2))
        self.assertAlmostEqual(m['F1'], 2 * (4 / 7) * (4 / 6) / (4 / 7 + 4 / 6))
        self.assertAlmostEqual(m['Avg6'], np.mean([m[k] for k in ['NMI', 'AMI', 'FMI', 'ARI', 'F1', 'Jaccard']]))


if __name__ == '__main__':
    unittest.main()
