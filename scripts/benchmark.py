#!/usr/bin/env python
"""Run SPAMO end to end (preprocess -> train -> PCA20 + mclust -> metrics) for several seeds and summarise.

Example (paper protocol, 5 seeds, one job per GPU):
    python scripts/benchmark.py --dataset hln --data-root data/raw --out runs/hln \
        --rscript $(which Rscript) --gpus 0,1,2,3,4
"""
import argparse
import itertools
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILES = {'hln': ('HLN/adata_RNA.h5ad', 'HLN/adata_ADT.h5ad'),
         'mb': ('Mouse_Brain/adata_RNA.h5ad', 'Mouse_Brain/adata_peaks_normalized.h5ad')}
METRICS = ['NMI', 'AMI', 'FMI', 'ARI', 'F1', 'Jaccard', 'Avg', 'Avg6']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=sorted(FILES), required=True)
    p.add_argument('--data-root', type=Path, default=ROOT/'data'/'raw')
    p.add_argument('--features-template', help='use preprocessed features instead of h5ad, e.g. work/hln/{seed}/features.npz')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026, 2027, 2028])
    p.add_argument('--gpus', default='0', help='comma separated GPU ids (one job per GPU at a time) or "cpu"')
    p.add_argument('--rscript', default=None)
    p.add_argument('--python', default=sys.executable)
    a = p.parse_args()
    rna, mod2 = (a.data_root/f for f in FILES[a.dataset])
    ref = ROOT/'data'/'references'/f'{a.dataset}.csv'
    devices = ['cpu'] if a.gpus == 'cpu' else a.gpus.split(',')
    jobs = queue.Queue()
    for s in a.seeds:
        jobs.put(s)

    def worker(device):
        while True:
            try:
                seed = jobs.get_nowait()
            except queue.Empty:
                return
            out = a.out/f'seed_{seed}'
            if (out/'run'/'metrics.csv').exists():
                continue
            inputs = (['--features', a.features_template.format(seed=seed)] if a.features_template
                      else ['--rna', str(rna), '--mod2', str(mod2)])
            cmd = [a.python, '-m', 'spamo', 'run', *inputs, '--preset', a.dataset,
                   '--seed', str(seed), '--out', str(out), '--device', 'cpu' if device == 'cpu' else 'cuda:0',
                   '--reference', f'ref={ref}']
            if a.rscript:
                cmd += ['--rscript', a.rscript]
            env = None if device == 'cpu' else {**__import__('os').environ, 'CUDA_VISIBLE_DEVICES': device}
            out.mkdir(parents=True, exist_ok=True)
            with open(out/'log.txt', 'w') as log:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f'seed {seed} on {device}: exit {code}', flush=True)

    threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
    [t.start() for t in threads]
    [t.join() for t in threads]
    rows = []
    for s in a.seeds:
        path = a.out/f'seed_{s}'/'run'/'metrics.csv'
        if path.exists():
            rows.append(pd.read_csv(path).assign(seed=s))
    if not rows:
        sys.exit('no finished runs')
    df = pd.concat(rows)
    df.to_csv(a.out/'per_seed.csv', index=False)
    summary = []
    for method, g in df.groupby('method'):
        summary.append(dict(method=method, n=len(g), **{f'{m}_mean': g[m].mean() for m in METRICS},
                            **{f'{m}_sd': g[m].std(ddof=1) for m in METRICS}))
    pd.DataFrame(summary).to_csv(a.out/'summary.csv', index=False)
    print(pd.DataFrame(summary).set_index('method')[[f'{m}_mean' for m in METRICS]].round(4).to_string())


if __name__ == '__main__':
    main()
