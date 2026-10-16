"""Create a tiny synthetic RNA + ADT dataset (4 spatial domains) for a quick CPU smoke run.

    python examples/make_tiny_demo.py examples/tiny_demo
    spamo run --rna examples/tiny_demo/adata_RNA.h5ad --mod2 examples/tiny_demo/adata_ADT.h5ad \
        --preset hln --clusters 4 --out runs/tiny_demo --device cpu \
        --reference demo=examples/tiny_demo/reference.csv
"""
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    side = 25
    grid = np.array([(i, j) for i in range(side) for j in range(side)], float)
    grid += rng.normal(0, 0.05, grid.shape)
    n = len(grid)
    domain = (grid[:, 0] > side / 2).astype(int) + 2 * (grid[:, 1] > side / 2).astype(int)
    genes = rng.gamma(2.0, 1.0, size=(4, 400))
    genes[:, :60] *= rng.uniform(1, 4, size=(4, 60))
    proteins = rng.gamma(4.0, 5.0, size=(4, 20))
    rna = anndata.AnnData(rng.poisson(genes[domain]).astype('float32'))
    adt = anndata.AnnData(rng.poisson(proteins[domain]).astype('float32'))
    names = [f'spot{i:04d}-1' for i in range(n)]
    rna.var_names = [f'gene{i}' for i in range(rna.n_vars)]
    adt.var_names = [f'prot{i}' for i in range(adt.n_vars)]
    for a in (rna, adt):
        a.obs_names = names
        a.obsm['spatial'] = grid
    rna.write_h5ad(out/'adata_RNA.h5ad')
    adt.write_h5ad(out/'adata_ADT.h5ad')
    pd.DataFrame(dict(barcode=names, label=domain + 1)).to_csv(out/'reference.csv', index=False)
    print(f'wrote {n} spots to {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'examples/tiny_demo')
