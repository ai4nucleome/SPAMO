"""Label-free graph cache preparation and loading (no labels are ever read here)."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .graphs import feature_graph, spatial_graph, save_graph, load_graph


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def preset(name):
    return json.loads((Path(__file__).parent/'presets'/f'{name}.json').read_text())


def prepare_cache(features, out, settings, seed):
    features, out = Path(features), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Cache destination must be empty: {out}')
    with np.load(features, allow_pickle=False) as data:
        x = [np.asarray(data[f'x{m}'], dtype='float32') for m in [1,2]]
        barcodes = data['barcodes'].astype(str)
        coords = [data['coordinates'], data['coordinates2'] if 'coordinates2' in data else data['coordinates']]
        n = len(barcodes)
        if len(set(barcodes)) != n or any(t.ndim != 2 or len(t) != n or not np.isfinite(t).all() for t in x):
            raise ValueError('Features must be aligned finite matrices with unique barcodes.')
        if any(len(c) != n for c in coords):
            raise ValueError('Coordinates and feature row counts differ.')
        shutil.copyfile(features, out/'features.npz')
        for m in [1,2]:
            save_graph(spatial_graph(coords[m-1], settings['spatial_k']), out/f'spatial{m}.npz')
            save_graph(feature_graph(x[m-1], settings['feature_k']), out/f'feature{m}.npz')
    names = ['features.npz','spatial1.npz','spatial2.npz','feature1.npz','feature2.npz']
    write_json(out/'READY.json', dict(seed=seed, n=n, feature_k=settings['feature_k'],
        spatial_k=settings['spatial_k'], label_access=False, graph_fixed=True,
        feature_metric='cosine', feature_weight='exp(-(1-similarity))', normalized_once=True,
        files={name: sha(out/name) for name in names},
        barcode_sha256=hashlib.sha256(('\n'.join(barcodes)+'\n').encode()).hexdigest()))


def load_cache(path, device='cpu', feature_k=None):
    """Load features and fixed graphs; verifies checksums and, if given, the feature-graph k."""
    path = Path(path)
    receipt = json.loads((path/'READY.json').read_text())
    if feature_k is not None and receipt.get('feature_k') != feature_k:
        raise ValueError(f"Cache feature k={receipt.get('feature_k')} differs from requested k={feature_k}.")
    for name, expected in receipt['files'].items():
        if sha(path/name) != expected:
            raise ValueError(f'Cache checksum mismatch: {name}')
    with np.load(path/'features.npz', allow_pickle=False) as data:
        barcodes = data['barcodes'].astype(str)
        x = [torch.from_numpy(data[f'x{m}'].astype('float32')).to(device) for m in [1,2]]
    barcode_sha = hashlib.sha256(('\n'.join(barcodes)+'\n').encode()).hexdigest()
    if barcode_sha != receipt['barcode_sha256'] or len(set(barcodes)) != len(barcodes):
        raise ValueError('Invalid barcode alignment.')
    if any(t.ndim != 2 or len(t) != len(barcodes) or not torch.isfinite(t).all() for t in x):
        raise ValueError('Invalid cached features.')
    spatial = [load_graph(path/f'spatial{m}.npz', device) for m in [1,2]]
    feature = [load_graph(path/f'feature{m}.npz', device) for m in [1,2]]
    if any(g.n != len(barcodes) for g in spatial + feature):
        raise ValueError('Graph and feature dimensions differ.')
    return x, spatial, feature, barcodes, receipt
