"""Cluster a trained embedding (PCA-20, then mclust and k-means) and score it against optional references."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .data import sha, write_json
from .metrics import compute


def evaluate(run, clusters, rscript=None, references=(), exclude_labels=()):
    """Cluster ``run/embedding.npz`` and write labels, ``metrics.csv`` and ``evaluation.json``.

    mclust (R, models EEE -> VVV -> EII) is the primary clustering and needs ``rscript``; k-means is always
    computed as a sensitivity analysis.  ``references`` is a sequence of ``(name, csv)`` with unique
    ``barcode`` and ``label`` columns whose barcode set equals the embedding's.  If ``exclude_labels`` is given,
    every reference is additionally scored on the spots whose label is not in that list (subset ``filtered``).
    """
    run = Path(run)
    done = json.loads((run/'DONE.json').read_text())
    if sha(run/'embedding.npz') != done['embedding_sha256']:
        raise ValueError('Embedding checksum mismatch.')
    data = np.load(run/'embedding.npz', allow_pickle=False)
    barcodes = data['barcodes'].astype(str)
    np.random.seed(done['seed'])
    x = PCA(n_components=min(20, len(barcodes)-1, data['embedding'].shape[1])).fit_transform(data['embedding'])
    predictions = dict(kmeans=KMeans(n_clusters=clusters, n_init=10, random_state=0).fit_predict(x)+1)
    selected = None
    if rscript:
        np.savetxt(run/'pca20.csv', x, delimiter=',', fmt='%.17g')
        subprocess.run([str(rscript), str(Path(__file__).with_name('cluster.R')), str(run/'pca20.csv'),
                        str(run/'mclust_raw.txt'), str(run/'mclust_model.txt'), str(clusters)], check=True,
                       env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        selected = (run/'mclust_model.txt').read_text().strip()
        if selected not in ['EEE', 'VVV', 'EII']:
            raise RuntimeError('mclust failed; no automatic switch to a different primary evaluator.')
        predictions['mclust'] = np.loadtxt(run/'mclust_raw.txt', dtype=int)
    rows, receipts, truths = [], {}, {}
    for name, path in references:
        if name in truths:
            raise ValueError(f'Duplicate reference name: {name}')
        ref = pd.read_csv(path, dtype={'barcode': str, 'label': str}).set_index('barcode', verify_integrity=True)
        if set(ref.index) != set(barcodes):
            raise ValueError(f'Reference barcode set differs: {name}')
        truths[name] = ref.loc[barcodes, 'label'].to_numpy()
        receipts[name] = dict(path=str(Path(path).resolve()), sha256=sha(path))
    for method, prediction in predictions.items():
        if len(prediction) != len(barcodes) or len(np.unique(prediction)) != clusters:
            raise ValueError('Invalid clustering output.')
        pd.DataFrame(dict(barcode=barcodes, prediction=prediction)).to_csv(run/f'labels_{method}.csv', index=False)
        for name, truth in truths.items():
            subsets = [('all', np.ones(len(barcodes), bool))]
            if len(exclude_labels):
                subsets.append(('filtered', ~np.isin(truth, list(exclude_labels))))
            for subset, mask in subsets:
                if not mask.any():
                    raise ValueError(f'Reference has no usable spots: {name}')
                rows.append(dict(method=method, reference=name, subset=subset, n_spots=int(mask.sum()),
                                 **compute(prediction[mask], truth[mask])))
    pd.DataFrame(rows).to_csv(run/'metrics.csv', index=False)
    write_json(run/'evaluation.json', dict(rows=rows, references=receipts, mclust_model=selected,
        exclude_labels=list(exclude_labels), embedding_sha256=done['embedding_sha256'],
        primary='mclust' if rscript else None,
        kmeans_role='sensitivity; not a replacement for mclust benchmark scores'))
