"""Clustering metrics reported in the paper: NMI, AMI, FMI, ARI, pairwise F1 and Jaccard.

``Avg`` is the mean of NMI, AMI, ARI, F1 and Jaccard (the definition of the original SPAMO code);
``Avg6`` additionally includes FMI.
"""
import numpy as np
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score,
                             normalized_mutual_info_score)

METRICS = ['NMI', 'AMI', 'FMI', 'ARI', 'F1', 'Jaccard', 'Avg', 'Avg6']


def pairwise_f_jaccard(pred, truth):
    """Pair-counting F1 and Jaccard index between two partitions."""
    _, t = np.unique(truth, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    table = np.zeros((p.max() + 1, t.max() + 1), dtype=np.int64)
    np.add.at(table, (p, t), 1)
    comb2 = lambda v: v * (v - 1) // 2
    tp = int(comb2(table).sum())
    fp = int(comb2(table.sum(1)).sum() - tp)
    fn = int(comb2(table.sum(0)).sum() - tp)
    jaccard = tp / (tp + fp + fn)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall), jaccard


def compute(pred, truth):
    """Return all metrics of a predicted partition against a reference partition."""
    f1, jac = pairwise_f_jaccard(pred, truth)
    out = dict(NMI=normalized_mutual_info_score(truth, pred), AMI=adjusted_mutual_info_score(truth, pred),
               FMI=fowlkes_mallows_score(truth, pred), ARI=adjusted_rand_score(truth, pred), F1=f1, Jaccard=jac)
    out['Avg'] = (out['NMI'] + out['AMI'] + out['ARI'] + out['F1'] + out['Jaccard']) / 5
    out['Avg6'] = (out['NMI'] + out['AMI'] + out['FMI'] + out['ARI'] + out['F1'] + out['Jaccard']) / 6
    return {k: float(v) for k, v in out.items()}
