"""EGNN on the empirical classification tasks (RQ1/RQ2, row "Ours" in Table 1).

For each slice, classes 1/3 (preserved time-respecting paths) are distinguished from classes 2/4
(shuffled) on the augmented event graphs; the hyperparameters are those of Table 4.

    python scripts/run_empirical.py --dataset ants_1_1 --slice 0 --runs 10
"""

import argparse

import torch

from egnn.datasets import EMPIRICAL_DATASETS, RESULTS_DIR, num_slices, task_files
from egnn.training import fit, load_dataset
from egnn.utils import save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=list(EMPIRICAL_DATASETS))
    parser.add_argument('--slice', type=int, nargs='*', help='slices to run (default: all)')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--layers', type=int, nargs='+', default=[32, 16, 8])
    parser.add_argument('--lin_layers', type=int, nargs='+', default=[8, 4])
    parser.add_argument('--pooling', default='add', choices=['add', 'mean', 'max'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(RESULTS_DIR / 'empirical' / 'EGNN'))
    parser.add_argument('--quiet', action='store_true', help='do not print the per-epoch training output')
    args = parser.parse_args()

    slices = args.slice if args.slice else range(num_slices(args.dataset))
    for s in slices:
        dataset = load_dataset(task_files(args.dataset, s, 'augmented'))
        accs = []
        for run in range(args.runs):
            torch.manual_seed(args.seed + run)
            acc, _, _ = fit(dataset, hidden_dims=args.layers, dense_dims=args.lin_layers, lr=args.lr,
                            weight_decay=args.weight_decay, epochs=args.epochs, batch_size=args.batch_size,
                            pooling=args.pooling, verbose=not args.quiet)
            accs.append(acc)
            print(f'{args.dataset} slice {s} run {run}: {acc:.3f}', flush=True)

        save_results(f'{args.out}/{args.dataset}_slice{s}.json', accs, dataset=args.dataset, slice=s, params=vars(args))


if __name__ == '__main__':
    main()
