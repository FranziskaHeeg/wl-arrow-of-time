"""GAT on the time-concatenated static graphs of the empirical tasks (row "GAT" in Table 1).

    python baselines/gat/run_gat_empirical.py --dataset ants_1_1 --slice 0
"""

import argparse

import torch

from egnn.datasets import EMPIRICAL_DATASETS, RESULTS_DIR, num_slices, task_files
from egnn.utils import save_results
from gat import fit, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=list(EMPIRICAL_DATASETS))
    parser.add_argument('--slice', type=int, nargs='*', help='slices to run (default: all)')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None, help='default: results/.../GAT or GAT-timestamps')
    parser.add_argument('--use_timestamps', action='store_true',
                        help='use the normalised timestamps as edge features (edge_dim=1)')
    parser.add_argument('--quiet', action='store_true', help='do not print the per-epoch training output')
    args = parser.parse_args()
    args.out = args.out or str(RESULTS_DIR / 'empirical' / ('GAT-timestamps' if args.use_timestamps else 'GAT'))

    slices = args.slice if args.slice else range(num_slices(args.dataset))
    for s in slices:
        accs = []
        for run in range(args.runs):
            torch.manual_seed(args.seed + run)
            dataset = load_dataset(task_files(args.dataset, s, 'concat'), use_timestamps=args.use_timestamps)
            accs.append(fit(dataset, epochs=args.epochs, lr=args.lr, use_timestamps=args.use_timestamps,
                            verbose=not args.quiet)['accuracy'])
            print(f'{args.dataset} slice {s} run {run}: {accs[-1]:.3f}', flush=True)
        save_results(f'{args.out}/{args.dataset}_slice{s}.json', accs, dataset=args.dataset, slice=s, params=vars(args))


if __name__ == '__main__':
    main()
