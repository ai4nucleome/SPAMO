# Data

Raw `.h5ad` files are **not** stored in this repository. Download them from Google Drive or
[Zenodo record 20160187](https://zenodo.org/records/20160187) and place them under `data/raw/`:

| Dataset | Modalities | Files |
|---|---|---|
| HLN (10x human lymph node) | RNA + ADT | `data/raw/HLN/adata_RNA.h5ad`, `data/raw/HLN/adata_ADT.h5ad` |
| MB (mouse brain, spatial epigenome-transcriptome) | RNA + ATAC peaks | `data/raw/Mouse_Brain/adata_RNA.h5ad`, `data/raw/Mouse_Brain/adata_peaks_normalized.h5ad` |
| Simulation | RNA + ADT + ATAC | `data/raw/Simulation/adata_RNA.h5ad`, `adata_ADT.h5ad`, `adata_ATAC.h5ad` |

Each file needs the expression matrix in `.X`, spatial coordinates in `.obsm["spatial"]` and `obs_names`
shared by the two modalities (MB additionally needs `.obsm["X_lsi"]` on the ATAC object, or raw peak counts).

`references/hln.csv`, `references/mb.csv` and `references/sim.csv` hold the benchmark labels used in the paper
(columns `barcode,label`; 3484 spots for HLN, 9196 for MB after the low-quality-spot filter, 1296 for Simulation).
