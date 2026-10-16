"""Tiny synthetic RNA+ADT-like data for CPU tests (no labels, no scanpy needed)."""
import numpy as np


def make_features(n=160, d1=12, d2=8, seed=0):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.array([(i % side, i // side) for i in range(n)], float) + rng.normal(0, 0.05, (n, 2))
    domain = (grid[:, 0] > side / 2).astype(int) + 2 * (grid[:, 1] > side / 2).astype(int)
    centers1, centers2 = rng.normal(size=(4, d1)), rng.normal(size=(4, d2))
    x1 = centers1[domain] + rng.normal(0, .5, (n, d1))
    x2 = centers2[domain] + rng.normal(0, .5, (n, d2))
    barcodes = np.array([f'spot{i:04d}-1' for i in range(n)])
    return dict(x1=x1.astype('float32'), x2=x2.astype('float32'), barcodes=barcodes,
                coordinates=grid), domain
