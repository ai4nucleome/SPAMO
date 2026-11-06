"""Three-modality model: three graph logits, three reconstructions, CLI prepare/train."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from helpers import make_features
from spamo import SPAMOMulti
from spamo.cli import main
from spamo.graphs import feature_graph, spatial_graph


class ThreeModalityTests(unittest.TestCase):
    def test_forward_and_three_logits(self):
        data, _ = make_features(n=80, d1=6, d2=5)
        rng = np.random.default_rng(1)
        x3 = torch.from_numpy(rng.normal(size=(80, 4)).astype('float32'))
        xs = [torch.from_numpy(data['x1']), torch.from_numpy(data['x2']), x3]
        spatial = [spatial_graph(data['coordinates'], 3)] * 3
        feature = [feature_graph(t.numpy(), 5) for t in xs]
        model = SPAMOMulti([t.shape[1] for t in xs], latent_dim=8)
        logits = [p for n, p in model.named_parameters() if 'adj_fusion' in n]
        self.assertEqual(len(logits), 3)
        out = model(xs, spatial, feature)
        self.assertEqual(out['emb_latent_combined'].shape, (80, 8))
        for m, x in enumerate(xs, 1):
            self.assertEqual(out[f'emb_recon_omics{m}'].shape, x.shape)
        loss = sum(torch.nn.functional.mse_loss(out[f'emb_recon_omics{m}'], x) for m, x in enumerate(xs, 1))
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in logits))

    def test_cli_prepares_three_graphs(self):
        data, _ = make_features(n=60, d1=30, d2=8)
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            np.savez(t/'features.npz', x1=data['x1'], x2=data['x2'],
                     x3=rng.normal(size=(60, 7)).astype('float32'), barcodes=data['barcodes'],
                     coordinates=data['coordinates'])
            main(['prepare', '--features', str(t/'features.npz'), '--preset', 'sim', '--seed', '3',
                  '--out', str(t/'cache')])
            ready = json.loads((t/'cache'/'READY.json').read_text())
            self.assertEqual(ready['modalities'], 3)
            self.assertTrue((t/'cache'/'feature3.npz').exists() and (t/'cache'/'spatial3.npz').exists())
            main(['train', '--cache', str(t/'cache'), '--preset', 'sim', '--seed', '3',
                  '--out', str(t/'run'), '--epochs', '2', '--device', 'cpu'])
            done = json.loads((t/'run'/'DONE.json').read_text())
            self.assertEqual(len(done['mix_spatial']), 3)
            emb = np.load(t/'run'/'embedding.npz')['embedding']
            self.assertEqual(emb.shape, (60, 64))


if __name__ == '__main__':
    unittest.main()
