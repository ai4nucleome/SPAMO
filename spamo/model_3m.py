"""
SpaMO-3M: Spatial Multi-Omics Integration Model (Triple Modality)
=================================================================

Extends the 2-modality SpaMO to 3 modalities (e.g., RNA + ADT + ATAC).
Reuses: GraphConvolution, AdaptiveAdjFusion, RobustEncoder, RobustDecoder,
        DGILoss, SpatialSmoothRegularizer from model.py.

New:  RobustFusionModule3M — cross-modal attention where Q = single modality,
      K/V = concatenation of the other two modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import math

from .model import (
    GraphConvolution,
    AdaptiveAdjFusion,
    LightCrossModalAttention,
    SpatialSmoothRegularizer,
    DGILoss,
    RobustEncoder,
    RobustDecoder,
)


# ─── 3-Modality Cross-Modal Attention ──────────────────────────────────────

class CrossModalAttention3M(nn.Module):
    """
    Cross-modal attention for triple modality.
    Q = single modality embedding (d_model),
    K/V = concatenation of the other two modalities (2 * d_model), projected to d_model.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model * 2, d_model)
        self.W_v = nn.Linear(d_model * 2, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, query, kv_other1, kv_other2):
        """
        query:      (N, d_model)  — the single modality
        kv_other1:  (N, d_model)  — other modality 1
        kv_other2:  (N, d_model)  — other modality 2
        """
        kv_concat = torch.cat([kv_other1, kv_other2], dim=-1)  # (N, 2*d_model)
        Q = self.W_q(query)         # (N, d_model)
        K = self.W_k(kv_concat)     # (N, d_model)
        V = self.W_v(kv_concat)     # (N, d_model)
        scores = torch.mm(Q, K.t()) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return self.W_o(torch.mm(attn, V))


# ─── 3-Modality Fusion Module ──────────────────────────────────────────────

class RobustFusionModule3M(nn.Module):
    """
    MLP fusion with optional cross-modal attention for 3 modalities.
    Each modality attends to the concatenation of the other two.
    """
    def __init__(self, d_model, dropout=0.1, use_cross_attn=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        if use_cross_attn:
            self.xattn_1 = CrossModalAttention3M(d_model, dropout)
            self.xattn_2 = CrossModalAttention3M(d_model, dropout)
            self.xattn_3 = CrossModalAttention3M(d_model, dropout)
            self.gate = nn.Sequential(nn.Linear(d_model * 6, d_model), nn.Sigmoid())
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model * 6, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
            )
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model),
            )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, emb1, emb2, emb3):
        if self.use_cross_attn:
            emb1_cross = self.xattn_1(emb1, emb2, emb3)
            emb2_cross = self.xattn_2(emb2, emb1, emb3)
            emb3_cross = self.xattn_3(emb3, emb1, emb2)

            emb1_enh = self.norm1(emb1 + 0.2 * emb1_cross)
            emb2_enh = self.norm2(emb2 + 0.2 * emb2_cross)
            emb3_enh = self.norm3(emb3 + 0.2 * emb3_cross)

            concat = torch.cat([emb1, emb2, emb3, emb1_enh, emb2_enh, emb3_enh], dim=-1)
            gate = self.gate(concat)
            fused = self.fusion_mlp(concat)
            fused = gate * fused + (1 - gate) * ((emb1 + emb2 + emb3) / 3)
            return fused, emb1_enh, emb2_enh, emb3_enh
        else:
            concat = torch.cat([emb1, emb2, emb3], dim=-1)
            fused = self.fusion_mlp(concat)
            return fused, emb1, emb2, emb3


# ─── SpaMO_3M: Full Triple-Modality Model ─────────────────────────────────

class SpaMO_3M(Module):
    """
    SpaMO for triple-modality spatial multi-omics.

    Components (per modality):
    - RobustEncoder: 2-layer GCN + AdaptiveAdjFusion + LayerNorm
    - RobustDecoder: single GCN reconstruction

    Shared:
    - RobustFusionModule3M: gated MLP + 3-way cross-modal attention
    - DGILoss: Deep Graph Infomax self-supervised learning
    - SpatialSmoothRegularizer: spatial topology preservation
    """
    def __init__(self, dim_in1, dim_out1, dim_in2, dim_out2,
                 dim_in3, dim_out3, dropout=0.1, use_cross_attn=True,
                 ordered_ablation_mode='full'):
        super().__init__()

        self.ordered_ablation_mode = ordered_ablation_mode
        d_model = dim_out1

        self.encoder_omics1 = RobustEncoder(dim_in1, dim_out1, dropout)
        self.encoder_omics2 = RobustEncoder(dim_in2, dim_out2, dropout)
        self.encoder_omics3 = RobustEncoder(dim_in3, dim_out3, dropout)

        self.fusion_module = RobustFusionModule3M(dim_out1, dropout, use_cross_attn)
        self.simple_fusion = RobustFusionModule3M(d_model, dropout, use_cross_attn=False)

        self.input_proj_omics1 = nn.Sequential(nn.Linear(dim_in1, d_model), nn.LayerNorm(d_model))
        self.input_proj_omics2 = nn.Sequential(nn.Linear(dim_in2, d_model), nn.LayerNorm(d_model))
        self.input_proj_omics3 = nn.Sequential(nn.Linear(dim_in3, d_model), nn.LayerNorm(d_model))
        self.early_encoder_omics1 = RobustEncoder(d_model, d_model, dropout)
        self.early_encoder_omics2 = RobustEncoder(d_model, d_model, dropout)
        self.early_encoder_omics3 = RobustEncoder(d_model, d_model, dropout)
        self.fused_encoder = RobustEncoder(d_model, d_model, dropout)
        self.late_attn_12 = CrossModalAttention3M(d_model, dropout)
        self.late_attn_13 = CrossModalAttention3M(d_model, dropout)
        self.late_attn_23 = CrossModalAttention3M(d_model, dropout)
        self.late_fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )
        self.parallel_fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )

        self.decoder_omics1 = RobustDecoder(dim_out1, dim_in1, dropout)
        self.decoder_omics2 = RobustDecoder(dim_out2, dim_in2, dropout)
        self.decoder_omics3 = RobustDecoder(dim_out3, dim_in3, dropout)

        self.spatial_smooth = SpatialSmoothRegularizer()
        self.dgi_loss_module = DGILoss(dim_out1)

    def _encode_ordered_variant(self, feat1, feat2, feat3,
                                adj_sp1, adj_ft1, adj_sp2, adj_ft2, adj_sp3, adj_ft3):
        mode = self.ordered_ablation_mode

        if mode in ('full', 'regularization_before_fusion'):
            emb1, _ = self.encoder_omics1(feat1, adj_sp1, adj_ft1)
            emb2, _ = self.encoder_omics2(feat2, adj_sp2, adj_ft2)
            emb3, _ = self.encoder_omics3(feat3, adj_sp3, adj_ft3)
            emb_combined, _, _, _ = self.fusion_module(emb1, emb2, emb3)
            return emb1, emb2, emb3, emb_combined

        proj1 = self.input_proj_omics1(feat1)
        proj2 = self.input_proj_omics2(feat2)
        proj3 = self.input_proj_omics3(feat3)

        if mode == 'early_interaction':
            _, proj1_inter, proj2_inter, proj3_inter = self.fusion_module(proj1, proj2, proj3)
            emb1, _ = self.early_encoder_omics1(proj1_inter, adj_sp1, adj_ft1)
            emb2, _ = self.early_encoder_omics2(proj2_inter, adj_sp2, adj_ft2)
            emb3, _ = self.early_encoder_omics3(proj3_inter, adj_sp3, adj_ft3)
            emb_combined, _, _, _ = self.simple_fusion(emb1, emb2, emb3)
            return emb1, emb2, emb3, emb_combined

        if mode == 'late_interaction':
            emb1, _ = self.encoder_omics1(feat1, adj_sp1, adj_ft1)
            emb2, _ = self.encoder_omics2(feat2, adj_sp2, adj_ft2)
            emb3, _ = self.encoder_omics3(feat3, adj_sp3, adj_ft3)
            fused_pre, _, _, _ = self.simple_fusion(emb1, emb2, emb3)
            late12 = self.late_attn_12(fused_pre, emb1, emb2)
            late13 = self.late_attn_13(fused_pre, emb1, emb3)
            late23 = self.late_attn_23(fused_pre, emb2, emb3)
            emb_combined = self.late_fusion_mlp(torch.cat([fused_pre, late12, late13, late23], dim=-1))
            return emb1, emb2, emb3, emb_combined

        if mode == 'no_ordered_design':
            emb1, _ = self.encoder_omics1(feat1, adj_sp1, adj_ft1)
            emb2, _ = self.encoder_omics2(feat2, adj_sp2, adj_ft2)
            emb3, _ = self.encoder_omics3(feat3, adj_sp3, adj_ft3)
            _, proj1_inter, proj2_inter, proj3_inter = self.fusion_module(proj1, proj2, proj3)
            emb_combined = self.parallel_fusion_mlp(
                torch.cat([emb1, emb2, emb3, proj1_inter, proj2_inter, proj3_inter], dim=-1)
            )
            return emb1, emb2, emb3, emb_combined

        if mode == 'fusion_before_graph_calibration':
            fused_raw, _, _, _ = self.simple_fusion(proj1, proj2, proj3)
            adj_spatial = (adj_sp1 + adj_sp2 + adj_sp3) / 3
            adj_feature = (adj_ft1 + adj_ft2 + adj_ft3) / 3
            emb_combined, _ = self.fused_encoder(fused_raw, adj_spatial, adj_feature)
            return emb_combined, emb_combined, emb_combined, emb_combined

        raise ValueError(f"Unknown ordered_ablation_mode: {mode}")

    def forward(self, feat1, feat2, feat3,
                adj_sp1, adj_ft1, adj_sp2, adj_ft2, adj_sp3, adj_ft3):

        emb1, emb2, emb3, emb_combined = self._encode_ordered_variant(
            feat1, feat2, feat3,
            adj_sp1, adj_ft1, adj_sp2, adj_ft2, adj_sp3, adj_ft3,
        )

        recon1 = self.decoder_omics1(emb_combined, adj_sp1)
        recon2 = self.decoder_omics2(emb_combined, adj_sp2)
        recon3 = self.decoder_omics3(emb_combined, adj_sp3)

        if self.ordered_ablation_mode == 'regularization_before_fusion':
            spatial_loss = (
                self.spatial_smooth(emb1, adj_sp1) +
                self.spatial_smooth(emb2, adj_sp2) +
                self.spatial_smooth(emb3, adj_sp3)
            ) / 3
            dgi_loss = (
                self.dgi_loss_module(emb1) +
                self.dgi_loss_module(emb2) +
                self.dgi_loss_module(emb3)
            ) / 3
        else:
            spatial_loss = self.spatial_smooth(emb_combined, adj_sp1)
            dgi_loss = self.dgi_loss_module(emb_combined)

        return {
            'emb_latent_omics1': emb1,
            'emb_latent_omics2': emb2,
            'emb_latent_omics3': emb3,
            'emb_latent_combined': emb_combined,
            'emb_recon_omics1': recon1,
            'emb_recon_omics2': recon2,
            'emb_recon_omics3': recon3,
            'spatial_loss': spatial_loss,
            'dgi_loss': dgi_loss,
        }
