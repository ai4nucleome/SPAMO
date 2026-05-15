"""
SpaMO-3M Trainer: Training loop for triple-modality SpaMO.
Mirrors trainer.py but handles 3 modalities with DGI + SpatialSmooth losses.
"""

import torch
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from .model_3m import SpaMO_3M
from .preprocess_3m import adjacent_matrix_preprocessing
from .preprocess import fix_seed


class Parametered_Graph(nn.Module):
    def __init__(self, adj, device):
        super().__init__()
        self.adj = adj
        self.device = device
        n = self.adj.shape[0]
        self.paramed_adj_omics = nn.Parameter(torch.FloatTensor(n, n))
        self.paramed_adj_omics.data.copy_(self.adj)

    def forward(self, A=None):
        if A is None:
            adj = (self.paramed_adj_omics + self.paramed_adj_omics.t()) / 2
        else:
            adj = (A + A.t()) / 2
        adj = nn.ReLU(inplace=True)(adj)
        normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj.to(self.device)

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx @ r_mat_inv
        return mx


class Train_3M:
    def __init__(self,
                 data,
                 datatype='Triplet',
                 device=torch.device('cpu'),
                 random_seed=2025,
                 learning_rate=0.001,
                 weight_decay=2e-2,
                 epochs=200,
                 dim_input=3000,
                 dim_output=64,
                 weight_factors=[1, 3, 3],
                 Arg=None,
                 dgi_weight: float = 0.1,
                 spatial_weight: float = 0.01,
                 dropout: float = 0.1,
                 use_cross_attn: bool = True,
                 optimizer_type: str = 'sgd',
                 lr_scheduler_type: str = 'none',
                 epochs_override: int = 0,
                 ordered_ablation_mode: str = 'full',
                 ):
        self.arg = Arg
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.dgi_weight = dgi_weight
        self.spatial_weight = spatial_weight
        self.dropout = dropout
        self.use_cross_attn = use_cross_attn
        self.optimizer_type = optimizer_type
        self.lr_scheduler_type = lr_scheduler_type
        self.ordered_ablation_mode = ordered_ablation_mode
        self.EMA_coeffi = 0.9

        if epochs_override > 0:
            self.epochs = epochs_override

        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adata_omics3 = self.data['adata_omics3']

        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, self.adata_omics3)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device).to_dense()
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device).to_dense()
        self.adj_spatial_omics3 = self.adj['adj_spatial_omics3'].to(self.device).to_dense()
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device).to_dense()
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device).to_dense()
        self.adj_feature_omics3 = self.adj['adj_feature_omics3'].to(self.device).to_dense()

        self.paramed_adj_omics1 = Parametered_Graph(self.adj_feature_omics1, self.device).to(self.device)
        self.paramed_adj_omics2 = Parametered_Graph(self.adj_feature_omics2, self.device).to(self.device)
        self.paramed_adj_omics3 = Parametered_Graph(self.adj_feature_omics3, self.device).to(self.device)

        self.adj_feature_omics1_copy = copy.deepcopy(self.adj_feature_omics1)
        self.adj_feature_omics2_copy = copy.deepcopy(self.adj_feature_omics2)
        self.adj_feature_omics3_copy = copy.deepcopy(self.adj_feature_omics3)

        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.features_omics3 = torch.FloatTensor(self.adata_omics3.obsm['feat'].copy()).to(self.device)

        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_input3 = self.features_omics3.shape[1]

    def train(self):
        self.model = SpaMO_3M(
            self.dim_input1, self.dim_output,
            self.dim_input2, self.dim_output,
            self.dim_input3, self.dim_output,
            dropout=self.dropout,
            use_cross_attn=self.use_cross_attn,
            ordered_ablation_mode=self.ordered_ablation_mode,
        ).to(self.device)

        all_params = (list(self.model.parameters()) +
                      list(self.paramed_adj_omics1.parameters()) +
                      list(self.paramed_adj_omics2.parameters()) +
                      list(self.paramed_adj_omics3.parameters()))

        fix_seed(self.random_seed)

        if self.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(all_params, lr=self.learning_rate,
                                               weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(all_params, lr=self.learning_rate,
                                             momentum=0.9, weight_decay=self.weight_decay)

        if self.lr_scheduler_type == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=self.learning_rate * 0.01)
        elif self.lr_scheduler_type == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5)
        else:
            scheduler = None

        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            results = self.model(
                self.features_omics1, self.features_omics2, self.features_omics3,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
                self.adj_spatial_omics3, self.adj_feature_omics3,
            )

            loss_recon = (
                self.weight_factors[0] * F.mse_loss(self.features_omics1, results['emb_recon_omics1']) +
                self.weight_factors[1] * F.mse_loss(self.features_omics2, results['emb_recon_omics2']) +
                self.weight_factors[2] * F.mse_loss(self.features_omics3, results['emb_recon_omics3'])
            )

            updated_adj_omics1 = self.paramed_adj_omics1()
            updated_adj_omics2 = self.paramed_adj_omics2()
            updated_adj_omics3 = self.paramed_adj_omics3()
            loss_fro = (
                torch.norm(updated_adj_omics1 - self.adj_feature_omics1_copy.detach(), p='fro') +
                torch.norm(updated_adj_omics2 - self.adj_feature_omics2_copy.detach(), p='fro') +
                torch.norm(updated_adj_omics3 - self.adj_feature_omics3_copy.detach(), p='fro')
            ) / 3

            loss_spatial = results['spatial_loss'].float() * self.spatial_weight
            dgi_loss = results['dgi_loss'].float() * self.dgi_weight

            loss = loss_recon + loss_fro + loss_spatial + dgi_loss

            print(f"[ep {epoch:03d}] recon={loss_recon.item():.4f}  "
                  f"fro={loss_fro.item():.4f}  "
                  f"spatial={loss_spatial.item():.4f}  "
                  f"dgi={dgi_loss.item():.4f}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

            self.adj_feature_omics1 = self.paramed_adj_omics1()
            self.adj_feature_omics2 = self.paramed_adj_omics2()
            self.adj_feature_omics3 = self.paramed_adj_omics3()

            self.adj_feature_omics1_copy = (
                self.EMA_coeffi * self.adj_feature_omics1_copy +
                (1 - self.EMA_coeffi) * updated_adj_omics1.detach().clone()
            )
            self.adj_feature_omics2_copy = (
                self.EMA_coeffi * self.adj_feature_omics2_copy +
                (1 - self.EMA_coeffi) * updated_adj_omics2.detach().clone()
            )
            self.adj_feature_omics3_copy = (
                self.EMA_coeffi * self.adj_feature_omics3_copy +
                (1 - self.EMA_coeffi) * updated_adj_omics3.detach().clone()
            )

        print("Model training finished!\n")

        start_time = time.time()
        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1, self.features_omics2, self.features_omics3,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
                self.adj_spatial_omics3, self.adj_feature_omics3,
            )
        end_time = time.time()
        print("Infer time: ", end_time - start_time)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_omics3 = F.normalize(results['emb_latent_omics3'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        return {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'emb_latent_omics3': emb_omics3.detach().cpu().numpy(),
            'emb_combined': emb_combined.detach().cpu().numpy(),
        }
