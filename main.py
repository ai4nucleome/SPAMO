import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import time
from spamo.preprocess import fix_seed
from spamo.preprocess import clr_normalize_each_cell, pca
from spamo.preprocess import construct_neighbor_graph, lsi
from spamo.trainer import Train
from spamo.trainer_3m import Train_3M
from spamo.utils import clustering, spatial_smoothing
try:
    from spamo.preprocess_3m import construct_neighbor_graph as construct_neighbor_graph_3M
except ImportError:
    construct_neighbor_graph_3M = None

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.data_type in ['10x', 'SPOTS', 'Stereo-CITE-seq']:
        adata_omics1 = sc.read_h5ad(args.file_fold + '/adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + '/adata_ADT.h5ad')
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        adata_omics1 = sc.read_h5ad(args.file_fold + '/adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + '/adata_peaks_normalized.h5ad')
    elif args.data_type == 'Simulation':
        adata_omics1 = sc.read_h5ad(args.file_fold + '/adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + '/adata_ADT.h5ad')
        adata_omics3 = sc.read_h5ad(args.file_fold + '/adata_ATAC.h5ad')

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    if args.data_type == 'Simulation':
        adata_omics3.var_names_make_unique()

    random_seed = args.random_seed
    fix_seed(random_seed)

    # Preprocess
    if args.data_type == '10x':
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=200)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'SPOTS':
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Stereo-CITE-seq':
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=80)
        sc.pp.filter_genes(adata_omics2, min_cells=50)
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Simulation':
        n_protein = adata_omics2.n_vars
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_protein)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_protein)
        sc.pp.highly_variable_genes(adata_omics3, flavor="seurat_v3", n_top_genes=3000)
        lsi(adata_omics3, use_highly_variable=False, n_components=n_protein + 1)
        adata_omics3.obsm['feat'] = adata_omics3.obsm['X_lsi'].copy()
        data = construct_neighbor_graph_3M(adata_omics1, adata_omics2, adata_omics3)
    else:
        assert 0

    if args.data_type == 'Simulation':
        model = Train_3M(
            data, datatype=args.data_type, device=device, Arg=args,
            dgi_weight=args.dgi_weight,
            spatial_weight=args.spatial_weight,
            epochs_override=args.epochs_override,
            dropout=args.dropout,
            use_cross_attn=args.use_cross_attn,
            optimizer_type=args.optimizer_type,
            lr_scheduler_type=args.lr_scheduler_type,
            ordered_ablation_mode=args.ordered_ablation_mode,
        )
    else:
        model = Train(
            data,
            datatype=args.data_type,
            device=device,
            dim_output=args.dim_output,
            Arg=args,
            dgi_weight=args.dgi_weight,
            spatial_weight=args.spatial_weight,
            epochs_override=args.epochs_override,
            dropout=args.dropout,
            use_cross_attn=args.use_cross_attn,
            random_seed=random_seed,
            optimizer_type=args.optimizer_type,
            lr_scheduler_type=args.lr_scheduler_type,
            ordered_ablation_mode=args.ordered_ablation_mode,
        )

    start_time = time.time()
    output = model.train()
    end_time = time.time()
    print("Training time: ", end_time - start_time)

    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['emb_combined'] = output['emb_combined'].copy()

    tool = 'mclust'
    clustering(adata, key='emb_combined', add_key='emb_combined', n_clusters=args.n_clusters, method=tool, use_pca=True)

    label = adata.obs['emb_combined']

    if args.data_type == 'Simulation':
        ids = label.index.astype(str).str[:4]
        int_list = [int(num_str) for num_str in ids]
        list = [-1 for i in range(len(int_list))]
        for i in range(len(int_list)):
            list[int_list[i]] = label[i]
        spot_size = 60
    else:
        list = label.tolist()
        spot_size = 20

    output_file = args.txt_out_path
    with open(output_file, 'w') as f:
        for num in list:
            f.write(f"{num}\n")

    if args.skip_plot:
        return

    if args.data_type == 'Stereo-CITE-seq':
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
    elif args.data_type == 'SPOTS':
        import numpy as np
        adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

    import matplotlib.pyplot as plt
    fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
    sc.pp.neighbors(adata, use_rep='emb_combined', n_neighbors=500)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color='emb_combined', ax=ax_list[0], title='SpaMO', s=spot_size, show=False)
    sc.pl.embedding(adata, basis='spatial', color='emb_combined', ax=ax_list[1], title='SpaMO', s=spot_size, show=False)

    plt.tight_layout(w_pad=0.3)
    plt.savefig(args.vis_out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SpaMO: Spatial Multi-Omics Integration')
    parser.add_argument('--file_fold', type=str, help='Path to data folder')
    parser.add_argument('--data_type', type=str,
                        choices=['10x', 'Spatial-epigenome-transcriptome', 'SPOTS', 'Stereo-CITE-seq', 'Simulation'],
                        help='data_type')
    parser.add_argument('--n_clusters', type=int, help='n_clusters for clustering')
    parser.add_argument('--dim_output', type=int, default=64, help='Latent space dimension')
    parser.add_argument('--init_k', type=int, default=10, help='init k')
    parser.add_argument('--KNN_k', type=int, default=20, help='KNN_k')
    parser.add_argument('--alpha', type=float, default=0.9, help='EMA coefficient')
    parser.add_argument('--cl_weight', type=float, default=1, help='Clustering loss weight')
    parser.add_argument('--RNA_weight', type=float, default=5, help='RNA reconstruction weight')
    parser.add_argument('--ADT_weight', type=float, default=5, help='ADT/ATAC reconstruction weight')
    parser.add_argument('--tau', type=float, default=2, help='Temperature for prototype contrastive loss')
    parser.add_argument('--vis_out_path', type=str, default='results/HLN.png', help='vis_out_path')
    parser.add_argument('--txt_out_path', type=str, default='results/HLN.txt', help='txt_out_path')
    parser.add_argument('--skip_plot', action='store_true',
                        help='Skip UMAP/spatial plotting after writing clustering labels')
    parser.add_argument('--dgi_weight', type=float, default=0.1, help='DGI self-supervised loss weight')
    parser.add_argument('--spatial_weight', type=float, default=0.01, help='Spatial smoothness regularization weight')
    parser.add_argument('--epochs_override', type=int, default=0, help='Override training epochs (0=use dataset default)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_cross_attn', action='store_true', default=True,
                        help='Use cross-modal attention in fusion (default: True)')
    parser.add_argument('--no_cross_attn', dest='use_cross_attn', action='store_false',
                        help='Disable cross-modal attention in fusion')
    parser.add_argument('--random_seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--lr_scheduler_type', type=str, default='none', choices=['none', 'cosine', 'plateau'],
                        help='LR scheduler type')
    parser.add_argument('--ordered_ablation_mode', type=str, default='full',
                        choices=[
                            'full',
                            'early_interaction',
                            'late_interaction',
                            'no_ordered_design',
                            'fusion_before_graph_calibration',
                            'regularization_before_fusion',
                        ],
                        help='Ordered-framework ablation variant')
    args = parser.parse_args()
    main(args)
