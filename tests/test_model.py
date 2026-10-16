import unittest

import torch

from spamo import SPAMO
from spamo.graphs import feature_graph, spatial_graph
from helpers import make_features


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        d, _ = make_features()
        self.x = [torch.from_numpy(d['x1']), torch.from_numpy(d['x2'])]
        self.spatial = spatial_graph(d['coordinates'], 3)
        self.feature = [feature_graph(t.numpy(), 5) for t in self.x]
        self.model = SPAMO(self.x[0].shape[1], self.x[1].shape[1], latent_dim=16)

    def test_forward_shapes(self):
        out = self.model(self.x[0], self.x[1], self.spatial, self.feature[0], self.spatial, self.feature[1])
        n = len(self.x[0])
        self.assertEqual(out['emb_latent_combined'].shape, (n, 16))
        self.assertEqual(out['emb_recon_omics1'].shape, self.x[0].shape)
        self.assertEqual(out['emb_recon_omics2'].shape, self.x[1].shape)

    def test_two_global_graph_logits_start_at_half(self):
        logits = [p for name, p in self.model.named_parameters() if 'adj_fusion' in name]
        self.assertEqual(len(logits), 2)                               # one scalar per modality
        for p in logits:
            self.assertEqual(p.numel(), 1)
            self.assertAlmostEqual(float(p.sigmoid()), 0.5)

    def test_loss_decreases_and_logits_receive_gradient(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01)
        losses = []
        for _ in range(30):
            out = self.model(self.x[0], self.x[1], self.spatial, self.feature[0], self.spatial, self.feature[1])
            loss = sum(torch.nn.functional.mse_loss(out[f'emb_recon_omics{m}'], v) for m, v in enumerate(self.x, 1))
            opt.zero_grad(); loss.backward()
            if not losses:
                grads = [p.grad for n, p in self.model.named_parameters() if 'adj_fusion' in n]
                self.assertTrue(all(g is not None and g.abs() > 0 for g in grads))
            opt.step(); losses.append(float(loss))
        self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
    unittest.main()
