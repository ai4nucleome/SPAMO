#!/usr/bin/env python
"""Compare two sets of SPAMO runs seed by seed: embedding difference, cluster agreement and metric deltas.

    python scripts/compare_runs.py --new runs/hln --old /path/to/old/seed_{seed} --seeds 2024 2025
``--new`` is a benchmark output directory (``seed_*/run``); ``--old`` is a template for the reference run
directory containing ``embedding.npz``, ``labels_*.csv`` and ``metrics.csv`` (or ``metrics6.csv``) (``{seed}`` is substituted).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

METRICS = ['NMI', 'AMI', 'FMI', 'ARI', 'F1', 'Jaccard']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--old', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026, 2027, 2028])
    p.add_argument('--csv', type=Path)
    a = p.parse_args()
    rows = []
    for s in a.seeds:
        new, old = a.new/f'seed_{s}'/'run', Path(a.old.format(seed=s))
        e1, e0 = np.load(new/'embedding.npz'), np.load(old/'embedding.npz')
        assert (e1['barcodes'] == e0['barcodes']).all(), 'barcode order differs'
        diff = float(np.abs(e1['embedding'] - e0['embedding']).max())
        m1 = pd.read_csv(new/'metrics.csv').set_index('method')
        m0 = pd.read_csv(old/('metrics6.csv' if (old/'metrics6.csv').exists() else 'metrics.csv')).set_index('method')
        for method in m1.index:
            if method not in m0.index:
                continue
            row = dict(seed=s, method=method, embedding_max_abs_diff=diff,
                       **{f'd{m}': float(m1.loc[method, m] - m0.loc[method, m]) for m in METRICS if m in m0.columns})
            l1 = pd.read_csv(new/f'labels_{method}.csv'); l0 = pd.read_csv(old/f'labels_{method}.csv')
            row['label_agreement_ARI'] = adjusted_rand_score(l1['prediction'], l0['prediction'])
            row['ARI_new'], row['ARI_old'] = float(m1.loc[method, 'ARI']), float(m0.loc[method, 'ARI'])
            rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(df.round(6).to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == '__main__':
    main()
