"""
SpaMO: Spatial Multi-Omics Integration Model
=============================================

Architecture:
1. RobustEncoder: 2-layer GCN + AdaptiveAdjFusion (spatial-feature graph blend) + LayerNorm
2. RobustFusionModule: MLP fusion with optional lightweight cross-modal attention
3. RobustDecoder: single GCN layer reconstruction
4. DGILoss: Deep Graph Infomax self-supervised loss (COSMOS/SpatialEx inspired)
5. SpatialSmoothRegularizer: spatial neighborhood smoothness regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


# ─── Graph Convolution ──────────────────────────────────────────────────────

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


# ─── Adaptive Adjacency Fusion ──────────────────────────────────────────────

class AdaptiveAdjFusion(nn.Module):
    """Learns optimal blend of spatial and feature adjacency matrices."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, adj_spatial, adj_feature):
        w = torch.sigmoid(self.alpha)
        return w * adj_spatial + (1 - w) * adj_feature


# ─── Cross-Modal Attention ──────────────────────────────────────────────────

class LightCrossModalAttention(nn.Module):
    """Single-head cross-modal attention for inter-modality information exchange."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, query, key_value):
        Q = self.W_q(query)
        K = self.W_k(key_value)
        V = self.W_v(key_value)
        scores = torch.mm(Q, K.t()) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return self.W_o(torch.mm(attn, V))


# ─── Spatial Smooth Regularizer ─────────────────────────────────────────────

class SpatialSmoothRegularizer(nn.Module):
    """Encourages spatially proximal spots to have similar embeddings."""
    def forward(self, emb, adj_spatial):
        emb_norm = F.normalize(emb, p=2, dim=1)
        degree = adj_spatial.sum(dim=1, keepdim=True).clamp(min=1)
        neighbor_emb = torch.mm(adj_spatial, emb_norm) / degree
        similarity = (emb_norm * neighbor_emb).sum(dim=1).mean()
        return 1 - similarity


# ─── DGI Self-Supervised Loss ───────────────────────────────────────────────

class DGILoss(nn.Module):
    """
    Deep Graph Infomax: maximizes mutual information between node embeddings
    and a global graph summary, using corruption-based contrastive learning.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.discriminator = nn.Bilinear(d_model, d_model, 1)
        self.summary_proj = nn.Linear(d_model, d_model)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        N = emb.shape[0]
        summary = torch.sigmoid(self.summary_proj(emb.mean(0, keepdim=True))).expand(N, -1)
        corrupt_emb = emb[torch.randperm(N, device=emb.device)]

        pos_logits = self.discriminator(emb, summary).squeeze(-1)
        neg_logits = self.discriminator(corrupt_emb, summary).squeeze(-1)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones(N, device=emb.device))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros(N, device=emb.device))
        return (pos_loss + neg_loss) * 0.5


# ─── Encoder ────────────────────────────────────────────────────────────────

class RobustEncoder(nn.Module):
    """2-layer GCN with adaptive spatial-feature graph fusion and LayerNorm."""
    def __init__(self, in_feat, out_feat, dropout=0.1, linear=False):
        super().__init__()
        act1 = (lambda x: x) if linear else F.relu
        self.gc1 = GraphConvolution(in_feat, out_feat, dropout=0.0 if linear else dropout, act=act1)
        self.gc2 = GraphConvolution(out_feat, out_feat, dropout=0.0, act=lambda x: x)
        self.adj_fusion = AdaptiveAdjFusion()
        self.norm = nn.LayerNorm(out_feat)

    def forward(self, feat, adj_spatial, adj_feature):
        adj = self.adj_fusion(adj_spatial, adj_feature)
        h = self.gc1(feat, adj)
        h = self.gc2(h, adj)
        h = self.norm(h)
        return h, adj


# ─── Decoder ────────────────────────────────────────────────────────────────

class RobustDecoder(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0.1):
        super().__init__()
        self.gc = GraphConvolution(in_feat, out_feat, dropout, act=lambda x: x)

    def forward(self, feat, adj):
        return self.gc(feat, adj)


# ─── Fusion Module ──────────────────────────────────────────────────────────

class RobustFusionModule(nn.Module):
    """MLP fusion with optional cross-modal attention enhancement."""
    def __init__(self, d_model, dropout=0.1, use_cross_attn=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        if use_cross_attn:
            self.cross_attn_1to2 = LightCrossModalAttention(d_model, dropout)
            self.cross_attn_2to1 = LightCrossModalAttention(d_model, dropout)
            self.gate = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.Sigmoid())
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
            )
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
            )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, emb1, emb2):
        if self.use_cross_attn:
            emb1_cross = self.cross_attn_1to2(emb1, emb2)
            emb2_cross = self.cross_attn_2to1(emb2, emb1)
            emb1_enhanced = self.norm1(emb1 + 0.2 * emb1_cross)
            emb2_enhanced = self.norm2(emb2 + 0.2 * emb2_cross)
            concat = torch.cat([emb1, emb2, emb1_enhanced, emb2_enhanced], dim=-1)
            gate = self.gate(concat)
            fused = self.fusion_mlp(concat)
            fused = gate * fused + (1 - gate) * ((emb1 + emb2) / 2)
            return fused, emb1_enhanced, emb2_enhanced
        else:
            concat = torch.cat([emb1, emb2], dim=-1)
            fused = self.fusion_mlp(concat)
            return fused, emb1, emb2


# ─── SpaMO: Full Model ─────────────────────────────────────────────────────

class SpaMO(Module):
    """
    SpaMO: Spatial Multi-Omics integration model.

    Components:
    - RobustEncoder (per modality): 2-layer GCN + AdaptiveAdjFusion + LayerNorm
    - RobustFusionModule: gated MLP fusion with cross-modal attention
    - RobustDecoder (per modality): GCN reconstruction
    - DGILoss: Deep Graph Infomax self-supervised learning
    - SpatialSmoothRegularizer: spatial topology preservation
    """
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1,
                 dim_in_feat_omics2, dim_out_feat_omics2,
                 dropout=0.1, use_cross_attn=True, linear=False,
                 ordered_ablation_mode='full'):
        super().__init__()

        self.ordered_ablation_mode = ordered_ablation_mode
        d_model = dim_out_feat_omics1

        self.encoder_omics1 = RobustEncoder(dim_in_feat_omics1, dim_out_feat_omics1, dropout, linear=linear)
        self.encoder_omics2 = RobustEncoder(dim_in_feat_omics2, dim_out_feat_omics2, dropout, linear=linear)
        self.fusion_module = RobustFusionModule(dim_out_feat_omics1, dropout, use_cross_attn)

        self.input_proj_omics1 = nn.Sequential(nn.Linear(dim_in_feat_omics1, d_model), nn.LayerNorm(d_model))
        self.input_proj_omics2 = nn.Sequential(nn.Linear(dim_in_feat_omics2, d_model), nn.LayerNorm(d_model))
        self.simple_fusion = RobustFusionModule(d_model, dropout, use_cross_attn=False)
        self.early_encoder_omics1 = RobustEncoder(d_model, d_model, dropout, linear=linear)
        self.early_encoder_omics2 = RobustEncoder(d_model, d_model, dropout, linear=linear)
        self.fused_encoder = RobustEncoder(d_model, d_model, dropout, linear=linear)
        self.late_attn_to_omics1 = LightCrossModalAttention(d_model, dropout)
        self.late_attn_to_omics2 = LightCrossModalAttention(d_model, dropout)
        self.late_fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )
        self.parallel_fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )

        self.decoder_omics1 = RobustDecoder(dim_out_feat_omics1, dim_in_feat_omics1, dropout)
        self.decoder_omics2 = RobustDecoder(dim_out_feat_omics2, dim_in_feat_omics2, dropout)
        self.spatial_smooth = SpatialSmoothRegularizer()
        self.dgi_loss_module = DGILoss(dim_out_feat_omics1)

    def _encode_ordered_variant(self, features_omics1, features_omics2,
                                adj_spatial_omics1, adj_feature_omics1,
                                adj_spatial_omics2, adj_feature_omics2):
        mode = self.ordered_ablation_mode

        if mode in ('full', 'regularization_before_fusion'):
            emb1, _ = self.encoder_omics1(features_omics1, adj_spatial_omics1, adj_feature_omics1)
            emb2, _ = self.encoder_omics2(features_omics2, adj_spatial_omics2, adj_feature_omics2)
            emb_combined, _, _ = self.fusion_module(emb1, emb2)
            return emb1, emb2, emb_combined

        proj1 = self.input_proj_omics1(features_omics1)
        proj2 = self.input_proj_omics2(features_omics2)

        if mode == 'early_interaction':
            _, proj1_inter, proj2_inter = self.fusion_module(proj1, proj2)
            emb1, _ = self.early_encoder_omics1(proj1_inter, adj_spatial_omics1, adj_feature_omics1)
            emb2, _ = self.early_encoder_omics2(proj2_inter, adj_spatial_omics2, adj_feature_omics2)
            emb_combined, _, _ = self.simple_fusion(emb1, emb2)
            return emb1, emb2, emb_combined

        if mode == 'late_interaction':
            emb1, _ = self.encoder_omics1(features_omics1, adj_spatial_omics1, adj_feature_omics1)
            emb2, _ = self.encoder_omics2(features_omics2, adj_spatial_omics2, adj_feature_omics2)
            fused_pre, _, _ = self.simple_fusion(emb1, emb2)
            fused_from_omics1 = self.late_attn_to_omics1(fused_pre, emb1)
            fused_from_omics2 = self.late_attn_to_omics2(fused_pre, emb2)
            emb_combined = self.late_fusion_mlp(torch.cat([fused_pre, fused_from_omics1, fused_from_omics2], dim=-1))
            return emb1, emb2, emb_combined

        if mode == 'no_ordered_design':
            emb1, _ = self.encoder_omics1(features_omics1, adj_spatial_omics1, adj_feature_omics1)
            emb2, _ = self.encoder_omics2(features_omics2, adj_spatial_omics2, adj_feature_omics2)
            _, proj1_inter, proj2_inter = self.fusion_module(proj1, proj2)
            emb_combined = self.parallel_fusion_mlp(torch.cat([emb1, emb2, proj1_inter, proj2_inter], dim=-1))
            return emb1, emb2, emb_combined

        if mode == 'fusion_before_graph_calibration':
            fused_raw, _, _ = self.simple_fusion(proj1, proj2)
            adj_spatial = (adj_spatial_omics1 + adj_spatial_omics2) / 2
            adj_feature = (adj_feature_omics1 + adj_feature_omics2) / 2
            emb_combined, _ = self.fused_encoder(fused_raw, adj_spatial, adj_feature)
            return emb_combined, emb_combined, emb_combined

        raise ValueError(f"Unknown ordered_ablation_mode: {mode}")

    def forward(self, features_omics1, features_omics2,
                adj_spatial_omics1, adj_feature_omics1,
                adj_spatial_omics2, adj_feature_omics2):

        emb1, emb2, emb_combined = self._encode_ordered_variant(
            features_omics1, features_omics2,
            adj_spatial_omics1, adj_feature_omics1,
            adj_spatial_omics2, adj_feature_omics2,
        )

        recon1 = self.decoder_omics1(emb_combined, adj_spatial_omics1)
        recon2 = self.decoder_omics2(emb_combined, adj_spatial_omics2)

        if self.ordered_ablation_mode == 'regularization_before_fusion':
            spatial_loss = (
                self.spatial_smooth(emb1, adj_spatial_omics1) +
                self.spatial_smooth(emb2, adj_spatial_omics2)
            ) / 2
            dgi_loss = (self.dgi_loss_module(emb1) + self.dgi_loss_module(emb2)) / 2
        else:
            spatial_loss = self.spatial_smooth(emb_combined, adj_spatial_omics1)
            dgi_loss = self.dgi_loss_module(emb_combined)

        return {
            'emb_latent_omics1': emb1,
            'emb_latent_omics2': emb2,
            'emb_latent_combined': emb_combined,
            'emb_recon_omics1': recon1,
            'emb_recon_omics2': recon2,
            'spatial_loss': spatial_loss,
            'dgi_loss': dgi_loss,
        }
