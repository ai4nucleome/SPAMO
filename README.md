# SPAMO

**Spatial Multi-Omics Integration via Dual-Graph Encoding and Neighborhood-Constrained Cross-Modal Fusion**

SPAMO integrates two spatial omics modalities measured on the same spots (RNA + ADT, RNA + ATAC) into one embedding for
spatial-domain identification. Each modality is encoded by a two-layer GCN over **two fixed graphs** (spatial and
feature-similarity), mixed by one learned global scalar per modality; the modalities are fused by cross-modal attention
restricted to spatial neighborhoods and trained with a reconstruction loss only.

## Method in one table

| Component | Setting |
|---|---|
| Spatial graph | Euclidean KNN on coordinates (MB k = 6, HLN k = 3), binary weights |
| Feature graph | global cosine KNN on the preprocessed features, **k = 20**, weights `exp(-(1 - cos))` |
| Base graphs | built once, symmetrised (max), unit self-loops, degree-normalised once; **fixed** during training |
| Graph mixing | operator `σ(α)·S + (1 − σ(α))·F`, one scalar `α` per modality shared by both GCN layers, `σ(α) = 0.5` at start |
| Fusion | neighborhood cross-attention (spatial mask), gated residual fusion of four concatenated blocks |
| Loss | weighted reconstruction MSE (MB 1:10, HLN 5:1); no graph update, Frobenius, EMA or contrastive terms |
| Training | FP32, deterministic algorithms; MB 400 epochs AdamW (wd 0.05), HLN 200 epochs Adam (wd 0.005), LR 0.01 → 1e-4 cosine |
| Clustering | embedding → PCA(20) → mclust (EEE → VVV → EII); k-means as sensitivity analysis |

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/ai4nucleome/SPAMO && cd SPAMO
pip install -e .            # core: torch, numpy, scipy, scikit-learn, pandas
pip install -e '.[raw]'     # additionally scanpy / anndata / scikit-misc to start from .h5ad files
```

or `conda env create -f environment.yml`. mclust (the primary clustering) needs **R** with the `mclust` package
(`install.packages("mclust")`); without `--rscript` only the k-means sensitivity result is produced.

## Quick start

Smoke test on a tiny synthetic dataset (CPU, ~15 s):

```bash
python examples/make_tiny_demo.py examples/tiny_demo
spamo run --rna examples/tiny_demo/adata_RNA.h5ad --mod2 examples/tiny_demo/adata_ADT.h5ad \
    --preset hln --clusters 4 --out runs/tiny_demo --device cpu \
    --reference demo=examples/tiny_demo/reference.csv --rscript $(which Rscript)
```

### Your own data

Inputs are two `.h5ad` files with the expression matrix in `.X`, coordinates in `.obsm["spatial"]` and shared `obs_names`.
Pick the preset closest to your data: `hln` (RNA + ADT: CLR + PCA for ADT) or `mb` (RNA + ATAC: LSI for ATAC).

```bash
spamo run --rna RNA.h5ad --mod2 ADT.h5ad --preset hln --clusters 8 --seed 2024 \
    --device cuda:0 --rscript $(which Rscript) --out runs/my_sample
```

The command is equivalent to the three stages, which can also be run separately:

```bash
spamo prepare --rna RNA.h5ad --mod2 ADT.h5ad --preset hln --seed 2024 --out work/cache      # preprocessing + fixed graphs
spamo train   --cache work/cache --preset hln --seed 2024 --device cuda:0 --out work/run    # needs an empty --out
spamo cluster --run work/run --preset hln --clusters 8 --rscript $(which Rscript)           # labels (+ metrics if --reference)
```

Preprocessed features can be supplied instead of `.h5ad` (`--features features.npz` with arrays `x1`, `x2`, `barcodes`,
`coordinates`, optional `coordinates2`). Pass references as `--reference NAME=CSV` (columns `barcode,label`; the barcode set
must equal the embedding's). Use `--epochs 3` for smoke runs; select the GPU with `CUDA_VISIBLE_DEVICES`.

### Outputs (`<out>/run/`)

`embedding.npz/.csv` (L2-normalised fused embedding), `labels_mclust.csv`, `labels_kmeans.csv`, `metrics.csv`
(NMI, AMI, FMI, ARI, F1, Jaccard, Avg, Avg6), `config.json`, `losses.json`, `initial_model.pt`, `checkpoint.pt`,
`DONE.json` (final spatial mixing weights, checksums).

### Python API

```python
import torch
from spamo import SPAMO
from spamo.graphs import feature_graph, spatial_graph

spatial = spatial_graph(coords, k=6)                       # fixed sparse operators
feat1, feat2 = feature_graph(x1, k=20), feature_graph(x2, k=20)
model = SPAMO(x1.shape[1], x2.shape[1], latent_dim=64)
out = model(x1, x2, spatial, feat1, spatial, feat2)        # dict with emb_latent_combined, emb_recon_omics{1,2}, ...
```

## Reproducing the paper benchmark

Place the data as described in [`data/README.md`](data/README.md), then (5 seeds, one job per GPU):

```bash
python scripts/benchmark.py --dataset hln --data-root data/raw --out runs/hln --rscript $(which Rscript) --gpus 0,1,2,3,4
python scripts/benchmark.py --dataset mb  --data-root data/raw --out runs/mb  --rscript $(which Rscript) --gpus 0,1,2,3,4
```

This writes `per_seed.csv` and `summary.csv` (mean/SD per method). Reference labels are in `data/references/`.
Accuracy of this repository against the original paper runs is documented in [`docs/VALIDATION.md`](docs/VALIDATION.md).
Note that mclust partitions of a few seeds are sensitive to GPU floating-point noise (see that document).

## Repository layout

```
src/spamo/       package: cli, preprocess, graphs, model, training, evaluation, metrics, cluster.R, presets/
tests/           unit tests (python -m unittest discover -s tests)
scripts/         benchmark.py (multi-seed runs + summary), compare_runs.py (compare two sets of runs)
examples/        tiny synthetic demo
data/            data instructions and benchmark reference labels (raw data not included)
docs/            validation report
```

## Tests

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests -v
```

## Data availability

Benchmark datasets: Google Drive or Zenodo record [20160187](https://zenodo.org/records/20160187). Large data files and
generated results are intentionally excluded from git (see [`data/README.md`](data/README.md)).

## Citation and licence

If you use this code or data, please cite the corresponding paper and data resources.

```bibtex
@misc{spamo,
  title = {SPAMO: Spatial Multi-Omics Integration via Dual-Graph Encoding and Neighborhood-Constrained Cross-Modal Fusion},
  note = {Data and scripts available at https://zenodo.org/records/20160187}
}
```

Third-party attributions are in [NOTICE.md](NOTICE.md). The preprocessing is derived from SpatialGlue (AGPL-3.0); please
follow the licence terms of the original codebases, datasets and dependencies.
