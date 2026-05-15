# Data Directory

Place dataset files here before running `run_best.sh`.

Expected layout:

```text
Data/
├── HLN/
│   ├── adata_RNA.h5ad
│   ├── adata_ADT.h5ad
│   └── GT_labels.txt
├── Mouse_Brain/
│   ├── adata_RNA.h5ad
│   ├── adata_peaks_normalized.h5ad
│   └── MB_cluster.txt
└── Simulation/
    ├── adata_RNA.h5ad
    ├── adata_ADT.h5ad
    ├── adata_ATAC.h5ad
    └── GT_1.txt
```

The `.h5ad` data files are intentionally not included in the clean repository.
