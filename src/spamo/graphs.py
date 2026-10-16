"""Fixed sparse graphs and propagation; graph values are not parameters."""
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

class EdgeMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, x, n):
        ctx.save_for_backward(indices, values, x)
        ctx.n = n
        matrix = torch.sparse_coo_tensor(indices, values, (n, n), is_coalesced=True)
        return torch.sparse.mm(matrix, x)

    @staticmethod
    def backward(ctx, grad_output):
        indices, values, x = ctx.saved_tensors
        grad_values = None
        grad_x = None
        if ctx.needs_input_grad[1]:
            grad_values = (grad_output[indices[0]] * x[indices[1]]).sum(1)
        if ctx.needs_input_grad[2]:
            matrix = torch.sparse_coo_tensor(indices, values, (ctx.n, ctx.n), is_coalesced=True)
            grad_x = torch.sparse.mm(matrix.transpose(0, 1), grad_output)
        return None, grad_values, grad_x, None

class EdgeGraph:
    def __init__(self, indices, values, n):
        self.indices, self.values, self.n = indices, values, n
        self.shape = (n, n)
        self.device = values.device

    def mm(self, x):
        return EdgeMultiply.apply(self.indices, self.values, x, self.n)

    def dense(self):
        return torch.sparse_coo_tensor(self.indices, self.values, self.shape).to_dense()

class MixedGraph:
    def __init__(self, spatial, feature, w):
        self.spatial, self.feature, self.w = spatial, feature, w
        self.shape, self.device = spatial.shape, spatial.device

    def mm(self, x):
        return self.w * self.spatial.mm(x) + (1 - self.w) * self.feature.mm(x)

class EdgeMask:
    def __init__(self, graphs):
        n = graphs[0].shape[0]
        codes = []
        for g in graphs:
            i = g.indices
            codes.extend([i[0] * n + i[1], i[1] * n + i[0]])
        diag = torch.arange(n, device=graphs[0].device)
        codes.append(diag * n + diag)
        code = torch.unique(torch.cat(codes), sorted=True)
        self.row, self.col, self.n = code // n, code % n, n

def graph_from_scipy(matrix, device='cpu'):
    coo = matrix.tocoo(copy=True)
    order = (coo.row.astype('int64') * coo.shape[0] + coo.col).argsort()
    indices = torch.from_numpy(np.stack([coo.row[order], coo.col[order]]).astype('int64')).to(device)
    values = torch.from_numpy(coo.data[order].astype('float32')).to(device)
    return EdgeGraph(indices, values, coo.shape[0])


def normalize_affinity(raw):
    """Add one unit self loop, then D^-1/2 A D^-1/2 exactly once."""
    raw = raw.tocsr(copy=True)
    if raw.shape[0] != raw.shape[1] or not np.isfinite(raw.data).all() or (raw.data < 0).any():
        raise ValueError('Graph must be square with finite nonnegative affinities.')
    if (raw != raw.T).nnz or np.count_nonzero(raw.diagonal()):
        raise ValueError('Expected symmetric raw affinities without self loops.')
    raw.setdiag(1.)
    graph = graph_from_scipy(raw)
    row, col = graph.indices
    degree = graph.values.new_zeros(graph.n).index_add(0, row, graph.values)
    inv = degree.rsqrt()
    return EdgeGraph(graph.indices, graph.values * inv[row] * inv[col], graph.n)


@torch.no_grad()
def feature_graph(features, k=20):
    """Global cosine KNN, exp[-(1-cos)], maximum union; k excludes self."""
    x = torch.as_tensor(features, dtype=torch.float32, device='cpu').detach()
    n = len(x)
    norms = x.norm(dim=1, keepdim=True)
    if x.ndim != 2 or not 0 < k < n or not torch.isfinite(x).all() or (norms <= 1e-8).any():
        raise ValueError('Need finite nonzero feature rows and 0 < k < n.')
    unit = x / norms
    neighbors, affinities = [], []
    for first in range(0, n, 256):
        stop = min(first + 256, n)
        sim = unit[first:stop] @ unit.T
        sim[torch.arange(stop-first), torch.arange(first, stop)] = -torch.inf
        values, ids = sim.topk(k, dim=1, sorted=True)
        tied = (sim == values[:, -1:]).sum(1) > 1
        if tied.any():
            ids[tied] = sim[tied].argsort(dim=1, descending=True, stable=True)[:, :k]
            values[tied] = sim[tied].gather(1, ids[tied])
        neighbors.append(ids.numpy())
        affinities.append(torch.exp(-(1-values.clamp(-1, 1))).numpy())
    ids, weights = np.concatenate(neighbors), np.concatenate(affinities)
    raw = sp.csr_matrix((weights.ravel(), (np.repeat(np.arange(n), k), ids.ravel())), shape=(n, n))
    raw = raw.maximum(raw.T).tocsr()
    raw.setdiag(0)
    raw.eliminate_zeros()
    return normalize_affinity(raw)


def spatial_graph(coordinates, k):
    """Euclidean KNN determines support; spatial edges have binary raw weights."""
    coords = np.asarray(coordinates)
    n = len(coords)
    if coords.ndim != 2 or not np.isfinite(coords).all() or not 0 < k < n:
        raise ValueError('Need finite spatial coordinates and 0 < k < n.')
    if len(np.unique(coords, axis=0)) != n:
        raise ValueError('Duplicate coordinates: resolve spatial positions before constructing KNN.')
    ids = NearestNeighbors(n_neighbors=k+1).fit(coords).kneighbors(coords, return_distance=False)
    if not np.array_equal(ids[:, 0], np.arange(n)):
        raise ValueError('Spatial self-neighbor was not first.')
    raw = sp.csr_matrix((np.ones(n*k), (np.repeat(ids[:, 0], k), ids[:, 1:].ravel())), shape=(n, n))
    raw = raw.maximum(raw.T)
    # Match the validated spatial builder: float64 SciPy normalization, then FP32.
    raw.setdiag(1.)
    degree = np.asarray(raw.sum(1)).ravel()
    return graph_from_scipy((sp.diags(degree**-.5) @ raw @ sp.diags(degree**-.5)).astype('float32'))


def save_graph(graph, path):
    np.savez_compressed(path, indices=graph.indices.cpu().numpy(),
                        values=graph.values.detach().cpu().numpy(), n=graph.n)


def load_graph(path, device='cpu'):
    with np.load(path, allow_pickle=False) as saved:
        if 'indices' in saved and 'values' in saved and 'n' in saved:
            graph = EdgeGraph(torch.from_numpy(saved['indices']).to(device),
                              torch.from_numpy(saved['values']).to(device), int(saved['n']))
        else:
            graph = graph_from_scipy(sp.load_npz(path), device)
    row, col = graph.indices
    if graph.indices.shape[0] != 2 or graph.values.numel() != row.numel():
        raise ValueError(f'Invalid graph shape: {path}')
    code = row * graph.n + col
    if not (torch.all(code[1:] > code[:-1]) and torch.isfinite(graph.values).all()
            and (graph.values > 0).all() and row.min() >= 0 and col.min() >= 0
            and row.max() < graph.n and col.max() < graph.n):
        raise ValueError(f'Graph must have sorted unique indices and positive finite values: {path}')
    if graph.values.requires_grad:
        raise ValueError('Base graph weights must be fixed.')
    return graph
