import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from .preprocess import pca
import matplotlib.pyplot as plt

'''
---------------------
author: Yahui Long https://github.com/JinmiaoChenLab/SpatialGlue
e-mail: chen_jinmiao@bii.a-star.edu.sg
AGPL-3.0 LICENSE
---------------------
'''

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    
    # Convert numpy array to R matrix properly
    data_matrix = adata.obsm[used_obsm]
    if hasattr(data_matrix, 'toarray'):
        data_matrix = data_matrix.toarray()
    data_matrix = np.asarray(data_matrix)
    
    # Use R code directly to avoid dimnames issues
    robjects.r(f'''
    library(mclust)
    set.seed({random_seed})
    data_matrix <- matrix(c({",".join(map(str, data_matrix.flatten()))}), 
                          nrow={data_matrix.shape[0]}, ncol={data_matrix.shape[1]}, 
                          byrow=TRUE)
    res <- Mclust(data_matrix, G={num_cluster}, modelNames="{modelNames}")
    mclust_res <- res$classification
    ''')
    mclust_res = np.array(robjects.r['mclust_res'])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']
       
def spatial_smoothing(adata, cluster_key: str, spatial_key: str = 'spatial',
                      n_neighbors: int = 6, n_iter: int = 1) -> None:
    """
    空间平滑后处理：对每个 spot 执行邻域多数投票，减少孤立噪声点。

    受空间转录组领域的 label smoothing 经典技巧启发（如 STAGATE、BayesSpace）。

    Parameters
    ----------
    adata        : AnnData，需含 obsm[spatial_key]（物理坐标）
    cluster_key  : obs 中的聚类标签列名
    n_neighbors  : 构建 KNN 邻域的邻居数（默认 6）
    n_iter       : 迭代平滑次数（默认 1）
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    coords = adata.obsm[spatial_key]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors(coords)   # [N, k+1]，第 0 列是自身

    labels = adata.obs[cluster_key].values.copy()
    for _ in range(n_iter):
        new_labels = labels.copy()
        for i in range(len(labels)):
            neighbor_idx = indices[i, 1:]   # 排除自身
            neighbor_labels = labels[neighbor_idx]
            # 多数投票：取邻居中出现最多的标签
            counts = np.bincount(neighbor_labels.astype(int))
            majority = np.argmax(counts)
            # 仅在邻居一致性高时更新（超过半数同意才改）
            if np.max(counts) > n_neighbors // 2:
                new_labels[i] = majority
        labels = new_labels

    adata.obs[cluster_key + '_smooth'] = labels.astype(str)
    adata.obs[cluster_key + '_smooth'] = adata.obs[cluster_key + '_smooth'].astype('category')


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

