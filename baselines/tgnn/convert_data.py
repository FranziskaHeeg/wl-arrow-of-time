"""Exports the temporal graphs of the empirical tasks as interaction CSV files for the TGNN baselines.

Each temporal graph becomes one dataset with id ``{dataset}_slice{s}_c{c}_{i}``. Nodes are indexed from 1,
timestamps are shifted to start at 0 and all labels are 0.

* ``--format tgat`` (TGAT, CAW): writes ``ml_{id}.csv`` with all-zero edge (``ml_{id}.npy``) and node
  (``ml_{id}_node.npy``) features of dimension ``--node_dim``, read from ``./processed/``.
* ``--format tgn`` (TGN, PINT): writes ``{id}.csv``, to be placed in ``./data/`` and converted by the
  model's own ``utils/preprocess_data.py``.

    python baselines/tgnn/convert_data.py --format tgat --out baselines/tgnn/tgat/processed
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from egnn.datasets import EMPIRICAL_DATASETS, PROCESSED_DIR


def convert(g, out_file: Path, fmt: str, node_dim: int):
    tedges = sorted(g.temporal_edges, key=lambda x: x[2])
    min_time = tedges[0][2]

    lines = [",u,i,ts,label,idx\n" if fmt == 'tgat' else "u,i,ts,label,idx\n"]
    for i, (u, v, t) in enumerate(tedges, start=1):
        row = f"{g.mapping.to_idx(u) + 1},{g.mapping.to_idx(v) + 1},{t + np.abs(min_time)},0.0,{i}\n"
        lines.append(f"{i - 1},{row}" if fmt == 'tgat' else row)

    with open(out_file.with_suffix('.csv'), 'w') as f:
        f.writelines(lines)

    if fmt == 'tgat':
        np.save(out_file.with_suffix('.npy'), np.zeros((len(tedges) + 1, node_dim)))
        np.save(out_file.parent / f'{out_file.name}_node.npy', np.zeros((g.n + 1, node_dim)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--format', choices=['tgat', 'tgn'], required=True)
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--datasets', nargs='+', default=list(EMPIRICAL_DATASETS), choices=list(EMPIRICAL_DATASETS))
    parser.add_argument('--node_dim', type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for file in sorted((PROCESSED_DIR / 'temporal' / dataset).rglob('*.pickle')):
            stem = file.stem.replace('-', '_')
            out_file = out_dir / (f'ml_{stem}' if args.format == 'tgat' else stem)
            if out_file.with_suffix('.csv').exists():
                continue
            with open(file, 'rb') as f:
                convert(pickle.load(f), out_file, args.format, args.node_dim)
        print(f'{dataset} done', flush=True)


if __name__ == '__main__':
    main()
