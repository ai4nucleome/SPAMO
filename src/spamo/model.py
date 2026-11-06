"""SPAMO model: fixed dual graphs, one global graph-mixing logit per modality (initialised at 0.5)."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter, Module
from .graphs import MixedGraph, EdgeMask

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
        output = adj.mm(support)
        output = self.act(output)
        return output

class AdaptiveAdjFusion(nn.Module):
    """One global logit per modality, shared across both GCN layers."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, spatial, feature):
        return MixedGraph(spatial, feature, self.alpha.sigmoid())


class LightCrossModalAttention(nn.Module):
    """Single-head attention over co-located and spatial neighboring spots."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, query, key_value, attn_mask):
        q, k, v = self.W_q(query), self.W_k(key_value), self.W_v(key_value)
        row, col, n = attn_mask.row, attn_mask.col, attn_mask.n
        scores = (q[row] * k[col]).sum(-1) / self.scale
        maxima = scores.detach().new_full((n,), -torch.inf)
        maxima.scatter_reduce_(0, row, scores.detach(), reduce='amax', include_self=True)
        numerator = (scores - maxima[row]).exp()
        denominator = numerator.new_zeros(n).index_add(0, row, numerator)
        weights = self.dropout(numerator / denominator[row])
        output = v.new_zeros((n, v.shape[1])).index_add(0, row, weights[:, None] * v[col])
        return self.W_o(output)


class RobustEncoder(nn.Module):
    """2-layer GCN with adaptive spatial-feature graph fusion and LayerNorm."""
    def __init__(self, in_feat, out_feat, dropout=0.1):
        super().__init__()
        act1 = F.relu

        self.gc1 = GraphConvolution(
            in_feat,
            out_feat,
            dropout=dropout,
            act=act1,
        )
        self.gc2 = GraphConvolution(
            out_feat,
            out_feat,
            dropout=0.0,
            act=lambda x: x,
        )
        self.adj_fusion = AdaptiveAdjFusion()
        self.norm = nn.LayerNorm(out_feat)

    def forward(self, feat, adj_spatial, adj_feature):
        adj = self.adj_fusion(adj_spatial, adj_feature)
        h = self.gc1(feat, adj)
        h = self.gc2(h, adj)
        h = self.norm(h)
        return h, adj

class RobustDecoder(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0.1):
        super().__init__()
        self.gc = GraphConvolution(in_feat, out_feat, dropout, act=lambda x: x)

    def forward(self, feat, adj):
        return self.gc(feat, adj)

class RobustFusionModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.cross_attn_1to2 = LightCrossModalAttention(d_model, dropout)
        self.cross_attn_2to1 = LightCrossModalAttention(d_model, dropout)
        self.gate = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, emb1, emb2, attn_mask):
        cross1 = self.cross_attn_1to2(emb1, emb2, attn_mask)
        cross2 = self.cross_attn_2to1(emb2, emb1, attn_mask)
        enhanced1 = self.norm1(emb1 + cross1)
        enhanced2 = self.norm2(emb2 + cross2)
        u = torch.cat([emb1, emb2, enhanced1, enhanced2], dim=-1)
        gate = self.gate(u)
        fused = gate * self.fusion_mlp(u) + (1 - gate) * ((emb1 + emb2) / 2)
        return fused, enhanced1, enhanced2


class SPAMO(nn.Module):
    """Two fixed graphs per modality; exactly two learned global graph logits."""
    def __init__(self, input_dim1, input_dim2, latent_dim=64, dropout=0.1):
        super().__init__()
        self.encoder_omics1 = RobustEncoder(input_dim1, latent_dim, dropout)
        self.encoder_omics2 = RobustEncoder(input_dim2, latent_dim, dropout)
        self.fusion_module = RobustFusionModule(latent_dim, dropout)
        self.decoder_omics1 = RobustDecoder(latent_dim, input_dim1, dropout)
        self.decoder_omics2 = RobustDecoder(latent_dim, input_dim2, dropout)
        self._mask_key = None

    def forward(self, x1, x2, spatial1, feature1, spatial2, feature2):
        key = (id(spatial1), id(spatial2))
        if self._mask_key != key:
            self._mask_value = EdgeMask([spatial1, spatial2])
            self._mask_key = key
        h1, _ = self.encoder_omics1(x1, spatial1, feature1)
        h2, _ = self.encoder_omics2(x2, spatial2, feature2)
        fused, _, _ = self.fusion_module(h1, h2, self._mask_value)
        recon1 = self.decoder_omics1(fused, spatial1)
        recon2 = self.decoder_omics2(fused, spatial2)
        return dict(emb_latent_omics1=h1, emb_latent_omics2=h2,
                    emb_latent_combined=fused, emb_recon_omics1=recon1,
                    emb_recon_omics2=recon2)


class _CrossAttention(nn.Module):
    """Neighborhood attention with a configurable key/value width (several other modalities concatenated)."""

    def __init__(self, d_model, kv_dim, dropout=0.1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(kv_dim, d_model)
        self.W_v = nn.Linear(kv_dim, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, query, key_value, mask):
        q, k, v = self.W_q(query), self.W_k(key_value), self.W_v(key_value)
        row, col, n = mask.row, mask.col, mask.n
        scores = (q[row] * k[col]).sum(-1) / self.scale
        maxima = scores.detach().new_full((n,), -torch.inf)
        maxima.scatter_reduce_(0, row, scores.detach(), reduce='amax', include_self=True)
        numerator = (scores - maxima[row]).exp()
        denominator = numerator.new_zeros(n).index_add(0, row, numerator)
        weights = self.dropout(numerator / denominator[row])
        out = v.new_zeros((n, v.shape[1])).index_add(0, row, weights[:, None] * v[col])
        return self.W_o(out)


class MultiFusion(nn.Module):
    """Gated residual fusion for M modalities. Each modality attends to the concatenation of the others."""

    def __init__(self, d, n_modalities, dropout=0.1):
        super().__init__()
        self.M = n_modalities
        kv = (n_modalities - 1) * d
        self.attn = nn.ModuleList([_CrossAttention(d, kv, dropout) for _ in range(n_modalities)])
        self.gate = nn.Sequential(nn.Linear(d * 2 * n_modalities, d), nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(nn.Linear(d * 2 * n_modalities, d * 2), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d * 2, d), nn.LayerNorm(d))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_modalities)])

    def forward(self, hs, mask):
        enhanced = []
        for m in range(self.M):
            others = torch.cat([h for j, h in enumerate(hs) if j != m], dim=-1)
            enhanced.append(self.norms[m](hs[m] + self.attn[m](hs[m], others, mask)))
        u = torch.cat(list(hs) + enhanced, dim=-1)
        gate = self.gate(u)
        return gate * self.fusion_mlp(u) + (1 - gate) * (sum(hs) / self.M)


class SPAMOMulti(nn.Module):
    """Fixed dual graphs for M modalities (the three-modality simulation benchmark uses M=3).

    One global graph-mixing logit per modality, neighborhood cross-attention, reconstruction only.
    The two-modality model stays :class:`SPAMO`; this class is the generalisation used when a third modality is given.
    """

    def __init__(self, input_dims, latent_dim=64, dropout=0.1):
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError('SPAMOMulti needs at least two modalities.')
        self.encoders = nn.ModuleList([RobustEncoder(dim, latent_dim, dropout) for dim in input_dims])
        self.fusion_module = MultiFusion(latent_dim, len(input_dims), dropout)
        self.decoders = nn.ModuleList([RobustDecoder(latent_dim, dim, dropout) for dim in input_dims])
        self._mask_key = None

    def forward(self, xs, spatials, features):
        key = tuple(id(s) for s in spatials)
        if self._mask_key != key:
            self._mask = EdgeMask(spatials)
            self._mask_key = key
        hs = [enc(x, s, f)[0] for enc, x, s, f in zip(self.encoders, xs, spatials, features)]
        fused = self.fusion_module(hs, self._mask)
        out = dict(emb_latent_combined=fused)
        for m, (dec, spatial, h) in enumerate(zip(self.decoders, spatials, hs), 1):
            out[f'emb_latent_omics{m}'] = h
            out[f'emb_recon_omics{m}'] = dec(fused, spatial)
        return out


SpaMO = SPAMO  # backwards-compatible alias
