import unittest

import numpy as np
import torch

from spamo.graphs import feature_graph, spatial_graph, normalize_affinity
from helpers import make_features


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.data, _ = make_features()

    def _check_operator(self, graph, n):
        dense = graph.dense().numpy()
        self.assertEqual(dense.shape, (n, n))
        np.testing.assert_allclose(dense, dense.T, atol=1e-6)          # symmetric
        self.assertTrue((dense.diagonal() > 0).all())                  # unit self loop survives normalisation
        self.assertTrue((dense >= 0).all())
        self.assertFalse(graph.values.requires_grad)                   # base graphs are fixed, not parameters

    def test_feature_graph(self):
        g = feature_graph(self.data['x1'], k=5)
        self._check_operator(g, len(self.data['x1']))
        degree = (g.dense().numpy() > 0).sum(1) - 1
        self.assertTrue((degree >= 5).all())                           # symmetric union: degree >= k

    def test_feature_graph_is_deterministic(self):
        a, b = feature_graph(self.data['x1'], 5), feature_graph(self.data['x1'], 5)
        self.assertTrue(torch.equal(a.indices, b.indices) and torch.equal(a.values, b.values))

    def test_spatial_graph(self):
        g = spatial_graph(self.data['coordinates'], 3)
        self._check_operator(g, len(self.data['x1']))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            feature_graph(np.zeros((10, 4), 'float32'), 3)             # zero rows
        with self.assertRaises(ValueError):
            spatial_graph(np.zeros((10, 2)), 3)                        # duplicate coordinates

    def test_normalize_requires_symmetry(self):
        import scipy.sparse as sp
        with self.assertRaises(ValueError):
            normalize_affinity(sp.csr_matrix(np.array([[0, 1.], [0, 0]])))


if __name__ == '__main__':
    unittest.main()
