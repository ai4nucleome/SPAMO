"""Train the final recipe: weighted reconstruction, fixed base graphs."""
import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_cache, sha, write_json
from .model import SPAMO


def setup(seed, device, threads=2):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already fixed earlier in this process (repeated calls)
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.8,6*1024**3/total), device)


def train(cache, out, settings, seed=2064, device='cpu', epochs=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Training output must be empty: {out}')
    setup(seed, device)
    x, spatial, feature, barcodes, receipt = load_cache(cache, device, settings['feature_k'])
    if receipt.get('seed') != seed:
        raise ValueError('Use the cache prepared with the requested seed.')
    for key in ('spatial_k', 'feature_k'):
        if receipt.get(key, settings[key]) != settings[key]:
            raise ValueError(f'Cache {key} differs from the requested preset.')
    cfg = dict(settings)
    if epochs is not None:
        cfg['epochs'] = epochs
    if cfg['epochs'] <= 0:
        raise ValueError('epochs must be positive.')
    model = SPAMO(x[0].shape[1], x[1].shape[1], cfg['latent_dim'], cfg['dropout'])
    torch.save(model.state_dict(), out/'initial_model.pt')
    write_json(out/'config.json', dict(recipe='spamo_fixed_dual_global_half', seed=seed, settings=cfg,
        device=str(device), cache=str(Path(cache).resolve()), cache_sha256=sha(Path(cache)/'READY.json'),
        graph_mode='fixed_dual', alpha_mode='global_per_modality', alpha_parameters=2,
        alpha_raw_initial=0., alpha_effective_initial=.5, alpha_shared_layers=True,
        objective='weighted_reconstruction_MSE', fro=False, ema=False,
        precision='FP32', tf32=False, amp=False, label_access=False,
        source_sha256={str(p.relative_to(Path(__file__).parent)):sha(p)
                       for p in Path(__file__).parent.rglob('*') if p.suffix in ['.py','.R','.json']},
        torch_version=torch.__version__, cuda_version=torch.version.cuda))
    model.to(device)
    optimizer = getattr(torch.optim,cfg['optimizer'])(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['lr_min'])
    start = time.monotonic()
    history = []
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        result = model(x[0],x[1],spatial[0],feature[0],spatial[1],feature[1])
        loss = sum(w*F.mse_loss(result[f'emb_recon_omics{m}'],values)
                   for m,values,w in zip([1,2],x,cfg['weights']))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        row = dict(epoch=epoch,recon=float(loss.detach()),seconds=time.monotonic()-start,
                   mix_spatial=[float(e.adj_fusion.alpha.sigmoid().detach())
                                for e in [model.encoder_omics1,model.encoder_omics2]])
        history.append(row)
        if epoch == 1 or epoch%10 == 0 or epoch == cfg['epochs']:
            write_json(out/'progress.json',row)
            print(json.dumps(row),flush=True)
    model.eval()
    with torch.no_grad():
        result = model(x[0],x[1],spatial[0],feature[0],spatial[1],feature[1])
        embedding = F.normalize(result['emb_latent_combined'],p=2,eps=1e-12,dim=1).cpu().numpy()
    np.savez_compressed(out/'embedding.npz',embedding=embedding,barcodes=barcodes)
    with (out/'embedding.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['barcode']+[f'emb_{i+1}' for i in range(embedding.shape[1])])
        writer.writerows(zip(barcodes,*embedding.T))
    write_json(out/'losses.json',history)
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),out/'checkpoint.pt')
    for name,expected in receipt['files'].items():
        if sha(Path(cache)/name) != expected:
            raise RuntimeError(f'Input graph/features changed: {name}')
    write_json(out/'DONE.json',dict(seed=seed,epochs=cfg['epochs'],seconds=time.monotonic()-start,
        n_spots=len(barcodes),embedding_sha256=sha(out/'embedding.npz'),
        graph_unchanged=True,mix_spatial=history[-1]['mix_spatial']))
    return out
