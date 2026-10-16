import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import anndata
    import scanpy  # noqa: F401
    import skmisc  # noqa: F401
    HAVE_RAW = True
except ImportError:
    HAVE_RAW = False


@unittest.skipUnless(HAVE_RAW, 'needs the optional "raw" extras (anndata, scanpy, scikit-misc)')
class PreprocessTest(unittest.TestCase):
    def test_hln_style_h5ad_to_features(self):
        from spamo.preprocess import preprocess_h5ad
        rng = np.random.default_rng(0)
        n = 150
        names = [f'cell{i}' for i in range(n)]
        coords = rng.uniform(0, 30, (n, 2))
        rna = anndata.AnnData(rng.poisson(1.5, (n, 300)).astype('float32'))
        adt = anndata.AnnData(rng.poisson(20, (n, 15)).astype('float32'))
        for a in (rna, adt):
            a.obs_names = names
            a.obsm['spatial'] = coords
        with tempfile.TemporaryDirectory() as tmp:
            rna.write_h5ad(Path(tmp)/'rna.h5ad'); adt.write_h5ad(Path(tmp)/'adt.h5ad')
            out = Path(tmp)/'features.npz'
            preprocess_h5ad(Path(tmp)/'rna.h5ad', Path(tmp)/'adt.h5ad', 'hln', out, seed=1)
            with np.load(out) as d:
                self.assertEqual(d['x1'].shape, (n, 14))               # ADT dims - 1 components for both modalities
                self.assertEqual(d['x2'].shape, (n, 14))
                self.assertEqual(list(d['barcodes']), names)
                self.assertTrue(np.isfinite(d['x1']).all() and np.isfinite(d['x2']).all())


if __name__ == '__main__':
    unittest.main()
