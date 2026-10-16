"""Raw h5ad -> aligned, label-free feature matrices (RNA PCA + ATAC LSI / ADT CLR-PCA).

The RNA/ADT/ATAC preprocessing follows the original SPAMO implementation, which in turn adapts
SpatialGlue (Yahui Long et al., https://github.com/JinmiaoChenLab/SpatialGlue, AGPL-3.0).
"""
import numpy as np


def preprocess_h5ad(rna_path, mod2_path, name, out, seed):
    """RNA PCA + ATAC LSI (MB) or RNA PCA + CLR/scale/PCA ADT (HLN)."""
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import Normalizer
    from sklearn.utils.extmath import randomized_svd

    np.random.seed(seed)
    rna, other = sc.read_h5ad(rna_path), sc.read_h5ad(mod2_path)
    if not rna.obs_names.is_unique or not other.obs_names.is_unique:
        raise ValueError('Input obs_names must be unique.')
    shared = set(rna.obs_names.intersection(other.obs_names))
    order = [s for s in rna.obs_names if s in shared]
    if not order:
        raise ValueError('Modalities have no shared spots.')
    rna, other = rna[order].copy(), other[order].copy()
    rna.var_names_make_unique()
    other.var_names_make_unique()
    sc.pp.filter_genes(rna, min_cells=10)
    if name == 'mb':
        sc.pp.filter_cells(rna, min_genes=200)
        other = other[rna.obs_names].copy()
    sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=min(3000,rna.n_vars))
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    high = rna[:,rna.var['highly_variable']]

    def pca(adata, components):
        matrix = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        return PCA(n_components=components).fit_transform(matrix)

    if name == 'mb':
        x1 = pca(high, max(1,min(50,rna.n_obs-1,high.n_vars)))
        if 'X_lsi' not in other.obsm:
            sc.pp.highly_variable_genes(other, flavor='seurat_v3', n_top_genes=min(3000,other.n_vars))
            matrix = other.X
            cols, rows = np.asarray(matrix.sum(0)).ravel(), np.asarray(matrix.sum(1)).ravel()
            if (cols <= 0).any() or (rows <= 0).any():
                raise ValueError('ATAC TF-IDF requires nonzero rows and columns.')
            idf = matrix.shape[0]/cols
            if sp.issparse(matrix):
                tfidf = matrix.multiply(1/rows[:,None]).multiply(idf)
            else:
                tfidf = matrix/rows[:,None]*idf
            normalized = np.log1p(Normalizer(norm='l1').fit_transform(tfidf)*1e4)
            lsi = randomized_svd(normalized, max(2,min(51,other.n_obs-1,other.n_vars-1)))[0]
            lsi -= lsi.mean(1,keepdims=True)
            lsi /= lsi.std(1,ddof=1,keepdims=True)
            other.obsm['X_lsi'] = lsi[:,1:]
        x2 = other.obsm['X_lsi'].copy()
    else:
        components = max(1,min(other.n_vars-1,rna.n_obs-1))
        x1 = pca(high,components)
        matrix = other.X.toarray() if sp.issparse(other.X) else np.asarray(other.X)
        def clr(row):
            scale = np.exp(np.log1p(row[row>0]).sum()/len(row))
            return np.log1p(row/scale)
        other.X = np.apply_along_axis(clr,1,matrix)
        sc.pp.scale(other)
        x2 = pca(other,components)
    if 'spatial' not in rna.obsm or 'spatial' not in other.obsm:
        raise ValueError('Both inputs need obsm["spatial"].')
    np.savez_compressed(out, x1=np.asarray(x1,dtype='float32'), x2=np.asarray(x2,dtype='float32'),
        barcodes=rna.obs_names.to_numpy().astype(str), coordinates=rna.obsm['spatial'],
        coordinates2=other.obsm['spatial'])
