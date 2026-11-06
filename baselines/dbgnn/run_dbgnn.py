"""DBGNN on the empirical classification tasks (row "DBGNN" in Table 1).

    python baselines/dbgnn/run_dbgnn.py --dataset ants_1_1
"""

import argparse

import torch

from dbgnn import run_experiment
from egnn.datasets import EMPIRICAL_DATASETS, RESULTS_DIR, num_slices
from egnn.utils import save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=list(EMPIRICAL_DATASETS))
    parser.add_argument('--slice', type=int, nargs='*', help='slices to run (default: all)')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--layers', type=int, nargs='+', default=[32, 16, 8])
    parser.add_argument('--lin_layers', type=int, nargs='+', default=[8, 4])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(RESULTS_DIR / 'empirical' / 'DBGNN'))
    parser.add_argument('--quiet', action='store_true', help='do not print the per-epoch training output')
    args = parser.parse_args()

    slices = args.slice if args.slice else range(num_slices(args.dataset))
    for s in slices:
        accs = []
        for run in range(args.runs):
            torch.manual_seed(args.seed + run)
            acc, _ = run_experiment(args.dataset, s, lr=args.lr, layers=args.layers, lin_layers=args.lin_layers,
                                    epochs=args.epochs, verbose=not args.quiet)
            accs.append(acc)
            print(f'{args.dataset} slice {s} run {run}: {acc:.3f}', flush=True)
        save_results(f'{args.out}/{args.dataset}_slice{s}.json', accs, dataset=args.dataset, slice=s, params=vars(args))


if __name__ == '__main__':
    main()
