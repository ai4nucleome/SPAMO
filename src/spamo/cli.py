"""Command line interface: ``spamo prepare | train | cluster | run``."""
import argparse
import os
from pathlib import Path
import tempfile


def _add_reference_args(parser):
    parser.add_argument('--clusters', type=int, help='number of clusters (default: preset value)')
    parser.add_argument('--rscript', help='Rscript with the mclust package; omit to obtain k-means only')
    parser.add_argument('--reference', action='append', default=[], metavar='NAME=CSV',
                        help='reference labels (CSV with barcode,label); repeatable')
    parser.add_argument('--exclude-labels', nargs='*', default=[], metavar='LABEL',
                        help='additionally score on spots whose reference label is not in this list')


def build_parser():
    parser = argparse.ArgumentParser(prog='spamo', description='SPAMO: fixed dual graphs + global graph mixing')
    sub = parser.add_subparsers(dest='command', required=True)

    prepare = sub.add_parser('prepare', help='Build the label-free fixed graph cache')
    inputs = prepare.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--features', help='NPZ: x1, x2, barcodes, coordinates; optional coordinates2')
    inputs.add_argument('--rna', help='RNA h5ad; requires --mod2')
    prepare.add_argument('--mod2', help='ATAC h5ad for mb; ADT h5ad for hln')
    prepare.add_argument('--preset', choices=['mb', 'hln'], required=True)
    prepare.add_argument('--seed', type=int, default=2024)
    prepare.add_argument('--out', required=True)

    train = sub.add_parser('train', help='Train the model on a prepared cache')
    train.add_argument('--cache', required=True)
    train.add_argument('--preset', choices=['mb', 'hln'], required=True)
    train.add_argument('--seed', type=int, default=2024)
    train.add_argument('--out', required=True)
    train.add_argument('--device', default='cpu', help='cpu or cuda:0; select the card with CUDA_VISIBLE_DEVICES')
    train.add_argument('--epochs', type=int, help='override the preset epoch budget (smoke runs)')

    cluster = sub.add_parser('cluster', help='Cluster an embedding and score optional references')
    cluster.add_argument('--run', required=True)
    cluster.add_argument('--preset', choices=['mb', 'hln'], required=True)
    _add_reference_args(cluster)

    run = sub.add_parser('run', help='prepare + train + cluster in one command (h5ad input)')
    run_inputs = run.add_mutually_exclusive_group(required=True)
    run_inputs.add_argument('--rna', help='RNA h5ad; requires --mod2')
    run_inputs.add_argument('--features', help='preprocessed NPZ instead of h5ad (see `prepare --features`)')
    run.add_argument('--mod2')
    run.add_argument('--preset', choices=['mb', 'hln'], required=True)
    run.add_argument('--seed', type=int, default=2024)
    run.add_argument('--out', required=True, help='output directory (cache/ and run/ are created inside)')
    run.add_argument('--device', default='cpu')
    run.add_argument('--epochs', type=int)
    _add_reference_args(run)
    return parser


def _parse_references(parser, items):
    references = []
    for item in items:
        name, _, path = item.partition('=')
        if not name or not path:
            parser.error('--reference must be NAME=CSV')
        references.append((name, path))
    return references


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # Determinism / resource defaults must be set before torch is imported.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ.setdefault(variable, '2')
    from .data import preset, prepare_cache
    settings = preset(args.preset)
    if getattr(args, 'clusters', None):
        settings['clusters'] = args.clusters

    def prepare_from_h5ad(rna, mod2, out):
        import torch
        from .preprocess import preprocess_h5ad
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='spamo-prepare-') as directory:
            features = Path(directory)/'features.npz'
            preprocess_h5ad(rna, mod2, args.preset, features, args.seed)
            prepare_cache(features, out, settings, args.seed)

    if args.command == 'prepare':
        if args.rna:
            if not args.mod2:
                parser.error('--rna requires --mod2')
            prepare_from_h5ad(args.rna, args.mod2, args.out)
        else:
            if args.mod2:
                parser.error('--mod2 requires --rna')
            import torch
            torch.set_num_threads(2)
            prepare_cache(args.features, args.out, settings, args.seed)
    elif args.command == 'train':
        from .training import train
        train(args.cache, args.out, settings, args.seed, args.device, args.epochs)
    elif args.command == 'cluster':
        from .evaluation import evaluate
        evaluate(args.run, settings['clusters'], args.rscript, _parse_references(parser, args.reference),
                 args.exclude_labels)
    else:
        from .evaluation import evaluate
        from .training import train
        out = Path(args.out)
        if args.rna:
            if not args.mod2:
                parser.error('--rna requires --mod2')
            prepare_from_h5ad(args.rna, args.mod2, out/'cache')
        else:
            prepare_cache(args.features, out/'cache', settings, args.seed)
        train(out/'cache', out/'run', settings, args.seed, args.device, args.epochs)
        evaluate(out/'run', settings['clusters'], args.rscript, _parse_references(parser, args.reference),
                 args.exclude_labels)
