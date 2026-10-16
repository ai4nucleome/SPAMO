import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spamo.cli import main
from helpers import make_features

PRESET_KW = ['--preset', 'hln']


class PipelineTest(unittest.TestCase):
    """prepare -> train -> cluster on a tiny CPU problem through the public CLI."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        data, domain = make_features(n=200)
        data['x1'] = np.hstack([data['x1']] * 3)[:, :30]
        np.savez(t/'features.npz', **data)
        pd.DataFrame(dict(barcode=data['barcodes'], label=domain + 1)).to_csv(t/'ref.csv', index=False)
        cls.t, cls.domain = t, domain

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_pipeline(self, name, seed=7):
        cache, run = self.t/name/'cache', self.t/name/'run'
        main(['prepare', '--features', str(self.t/'features.npz'), *PRESET_KW, '--seed', str(seed), '--out', str(cache)])
        main(['train', '--cache', str(cache), *PRESET_KW, '--seed', str(seed), '--out', str(run), '--epochs', '5'])
        return cache, run

    def test_end_to_end_and_determinism(self):
        cache, run = self.run_pipeline('a')
        ready = json.loads((cache/'READY.json').read_text())
        self.assertEqual(ready['feature_k'], 20)
        self.assertFalse(ready['label_access'])
        done = json.loads((run/'DONE.json').read_text())
        self.assertEqual(done['epochs'], 5)
        self.assertEqual(len(done['mix_spatial']), 2)
        _, run2 = self.run_pipeline('b')
        e1 = np.load(run/'embedding.npz')['embedding']
        e2 = np.load(run2/'embedding.npz')['embedding']
        self.assertEqual(e1.shape, (200, 64))
        np.testing.assert_array_equal(e1, e2)                          # same seed -> identical embedding
        np.testing.assert_allclose(np.linalg.norm(e1, axis=1), 1, atol=1e-5)

    def test_cluster_scores_reference(self):
        _, run = self.run_pipeline('c', seed=3)
        main(['cluster', '--run', str(run), *PRESET_KW, '--reference', f'toy={self.t/"ref.csv"}'])
        metrics = pd.read_csv(run/'metrics.csv')
        self.assertEqual(list(metrics['method']), ['kmeans'])         # no Rscript -> k-means sensitivity only
        for col in ['NMI', 'AMI', 'FMI', 'ARI', 'F1', 'Jaccard', 'Avg6']:
            self.assertIn(col, metrics.columns)
        self.assertEqual(metrics['n_spots'].iloc[0], 200)

    def test_refuses_nonempty_output_and_seed_mismatch(self):
        cache, run = self.run_pipeline('d', seed=5)
        with self.assertRaises(FileExistsError):
            main(['train', '--cache', str(cache), *PRESET_KW, '--seed', '5', '--out', str(run), '--epochs', '1'])
        with self.assertRaises(ValueError):
            main(['train', '--cache', str(cache), *PRESET_KW, '--seed', '6', '--out', str(self.t/'d'/'run2'),
                  '--epochs', '1'])

    def test_feature_k_mismatch_is_rejected(self):
        cache, _ = self.run_pipeline('e', seed=9)
        from spamo.data import load_cache
        with self.assertRaises(ValueError):
            load_cache(cache, 'cpu', feature_k=6)


if __name__ == '__main__':
    unittest.main()
