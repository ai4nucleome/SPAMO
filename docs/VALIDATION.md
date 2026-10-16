# Validation of the refactored package

Goal: show that this repository (`src/spamo`) reproduces the accuracy of the code that produced the paper results
on the two benchmark datasets (HLN, Mouse Brain), seeds 2024–2028, feature-graph k = 20.

* **Reference:** the paper runs (`rev_reproduction/runs/main/{hln,mb}/spamo`), i.e. the experiment layer built on the
  development package `spamoRev`.
* **New:** `scripts/benchmark.py` of this repository (`spamo run`: preprocessing → graphs → training → PCA-20 + mclust
  → six metrics), PyTorch 2.11 / CUDA 12.8, one job per GPU, FP32, deterministic-algorithms flag on.
* Evaluation protocol is identical in both: embedding → PCA(20) → mclust (EEE → VVV → EII, 10 / 14 clusters), k-means as
  sensitivity analysis, six metrics against `data/references/{hln,mb}.csv`.

## 1. Accuracy (mclust, mean ± SD over 5 seeds)

| Dataset | Version | NMI | AMI | FMI | ARI | F1 | Jaccard | Avg6 |
|---|---|---|---|---|---|---|---|---|
| HLN | reference (paper runs) | .420±.011 | .416±.011 | .492±.016 | .334±.014 | .491±.016 | .326±.014 | .413±.013 |
| HLN | **new, end-to-end from h5ad** | .420±.010 | .416±.010 | .492±.016 | .334±.014 | .491±.016 | .326±.014 | .413±.013 |
| HLN | new, reference's preprocessed features | .413±.008 | .409±.008 | .472±.024 | .319±.019 | .471±.025 | .308±.021 | .399±.017 |
| MB | reference (paper runs) | .513±.004 | .511±.004 | .501±.004 | .439±.005 | .501±.004 | .334±.004 | .466±.004 |
| MB | **new, end-to-end from h5ad** | .506±.016 | .504±.016 | .497±.013 | .433±.017 | .496±.013 | .330±.012 | .461±.014 |
| MB | new, reference's preprocessed features | .512±.004 | .511±.004 | .501±.005 | .438±.006 | .501±.004 | .334±.004 | .466±.004 |

Per-seed mclust ARI:

| Dataset | Version | 2024 | 2025 | 2026 | 2027 | 2028 |
|---|---|---|---|---|---|---|
| HLN | reference | .3389 | .3417 | .3137 | .3262 | .3484 |
| HLN | new, end-to-end | .3389 | .3421 | .3139 | .3262 | .3484 |
| HLN | new, same features | .3389 | .2890 | .3137 | .3262 | .3252 |
| MB | reference | .4360 | .4320 | .4384 | .4418 | .4460 |
| MB | new, end-to-end | .4360 | .4033 | .4385 | .4420 | .4458 |
| MB | new, same features | .4350 | .4320 | .4376 | .4420 | .4458 |

HLN: the end-to-end run agrees with the reference on every seed to ≤ 4·10⁻⁴ ARI (mean ARI .334 vs .334).
MB: four of five seeds agree to ≤ 3·10⁻⁴; seed 2025 differs by −0.029 (ARI .403 vs .432), which moves the mean ARI from
.439 to .433, within one SD of the reference. Full tables: `validation/per_seed_metrics.csv`,
`validation/compare_*.csv`.

## 2. Where the remaining differences come from

The few seeds that differ do so through the *mclust partition*, not through the representation:

1. **The embeddings agree to ~10⁻⁴–10⁻³** (max |Δ| of the L2-normalised 64-d embedding, `compare_*.csv`), but a
   few seeds sit on a boundary between two mclust solutions, so a perturbation of that size flips some assignments.
2. **GPU runs are not bit-reproducible even for identical code and inputs.** HLN, seed 2025, same preprocessed features,
   same code, repeated three times: mclust ARI = .2885, .2864 (same GPU) and .3417 (another GPU); embedding max |Δ| ≈ 2·10⁻⁴.
   The run-to-run spread of this seed (≈ 0.05 ARI) is as large as the differences in the table above.
3. **Preprocessing is not bit-reproducible either** (PCA/SVD with multithreaded BLAS): re-running the preprocessing changes
   the features at the 10⁻¹²–10⁻⁶ level (`x1`/`x2` of HLN seed 2025: max |Δ| 1.8·10⁻¹² / 2.9·10⁻⁶), which is
   amplified by training. The "end-to-end" and "same features" rows therefore differ on different seeds.

## 3. Exact code equivalence (CPU)

On CPU the computation is deterministic, which allows a strict test (HLN, seed 2025, identical preprocessed features,
k = 20, 200 epochs):

| Comparison | max \|Δ embedding\| |
|---|---|
| new package vs. itself (two runs) | **0.0** (bitwise) |
| new package vs. development package `spamoRev` (same training loop) | **0.0** (bitwise) |
| new package vs. the model implementation used by the paper's experiment layer (`variants.py`, arm `native`) | 5.9·10⁻⁴ |

So the refactoring is numerically a no-op with respect to the development package; the paper experiments used a
mathematically equivalent but separately written model (fusion variants share one code path), which differs from the
package at floating-point level (≈ 10⁻⁴–10⁻³ after 200–400 epochs), hence the occasional mclust flips above. A CPU run
of MB (seed 2025) gives the same mclust ARI (.4417) with both implementations. CPU and GPU runs differ by ~0.2–0.4 in
max |Δ| (the optimisation is chaotic), yet the benchmark accuracy is the same, as Section 1 shows.

## Reproduce

```bash
python scripts/benchmark.py --dataset hln --data-root data/raw --out runs/hln --rscript $(which Rscript) --gpus 0,1,2,3,4
python scripts/benchmark.py --dataset mb  --data-root data/raw --out runs/mb  --rscript $(which Rscript) --gpus 0,1,2,3,4
python scripts/compare_runs.py --new runs/hln --old '/path/to/reference/hln/spamo/seed_{seed}'
```
