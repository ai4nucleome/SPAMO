import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def read_labels(path: str) -> np.ndarray:
    return np.array([int(x.strip()) for x in Path(path).read_text().splitlines() if x.strip()])


def pairwise_f_jaccard(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    _, truth_inv = np.unique(truth, return_inverse=True)
    _, pred_inv = np.unique(pred, return_inverse=True)
    contingency = np.zeros((pred_inv.max() + 1, truth_inv.max() + 1), dtype=np.int64)
    np.add.at(contingency, (pred_inv, truth_inv), 1)

    def comb2(values: np.ndarray) -> np.ndarray:
        return values * (values - 1) // 2

    tp = int(comb2(contingency).sum())
    fp = int(comb2(contingency.sum(axis=1)).sum() - tp)
    fn = int(comb2(contingency.sum(axis=0)).sum() - tp)

    jaccard = tp / (tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure, jaccard


def main(args: argparse.Namespace) -> None:
    truth = read_labels(args.GT_path)
    pred = read_labels(args.our_path)
    if truth.shape != pred.shape:
        raise ValueError(f"Label length mismatch: GT={len(truth)} prediction={len(pred)}")

    f_measure, jaccard = pairwise_f_jaccard(pred, truth)
    mutual_info = mutual_info_score(truth, pred)
    nmi = normalized_mutual_info_score(truth, pred)
    ami = adjusted_mutual_info_score(truth, pred)
    v_measure = v_measure_score(truth, pred)
    homogeneity = homogeneity_score(truth, pred)
    completeness = completeness_score(truth, pred)
    ari = adjusted_rand_score(truth, pred)
    fmi = fowlkes_mallows_score(truth, pred)
    avg = (nmi + ami + ari + f_measure + jaccard) / 5

    output = "\n".join(
        [
            f"Our     jaccard: {jaccard:.6f}",
            f"Our     F_measure: {f_measure:.6f}",
            f"Our     Mutual Information: {mutual_info:.6f}",
            f"Our     NMI: {nmi:.6f}",
            f"Our     AMI: {ami:.6f}",
            f"Our     V-measure: {v_measure:.6f}",
            f"Our     Homogeneity: {homogeneity:.6f}",
            f"Our     Completeness: {completeness:.6f}",
            f"Our     (ARI): {ari:.6f}",
            f"Our     (FMI): {fmi:.6f}",
            f"Our     Avg.: {avg:.6f}",
            "",
        ]
    )
    Path(args.save_path).write_text(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate clustering metrics.")
    parser.add_argument("--GT_path", required=True, type=str, help="Path to ground-truth labels")
    parser.add_argument("--our_path", required=True, type=str, help="Path to predicted labels")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save metrics")
    main(parser.parse_args())
