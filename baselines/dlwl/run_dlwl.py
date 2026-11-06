"""DL-WL temporal graph kernel on the empirical classification tasks (row "DL-WL" in Table 1).

Requires the compiled tgkernel binary, see tgkernel/PATCHES.md. With --delta_restricted, the line graph expansion
is restricted to the maximum waiting time delta of the dataset (Table 3).

    python baselines/dlwl/run_dlwl.py --dataset ants_1_1
"""

import argparse

from dlwl import evaluate_holdout, get_grams
from egnn.datasets import EMPIRICAL_DATASETS, RESULTS_DIR, num_slices
from egnn.utils import save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=list(EMPIRICAL_DATASETS))
    parser.add_argument('--slice', type=int, nargs='*', help='slices to run (default: all)')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--hmax', type=int, default=5, help='maximum number of WL iterations')
    parser.add_argument('--delta_restricted', action='store_true',
                        help='bound the waiting time by the delta of the dataset (default: unbounded)')
    parser.add_argument('--out', default=None, help='default: results/empirical/DL-WL or DL-WL-delta')
    args = parser.parse_args()
    delta = EMPIRICAL_DATASETS[args.dataset] if args.delta_restricted else None
    args.out = args.out or str(RESULTS_DIR / 'empirical' / ('DL-WL-delta' if args.delta_restricted else 'DL-WL'))

    slices = args.slice if args.slice else range(num_slices(args.dataset))
    for s in slices:
        grams, y = get_grams(args.dataset, s, kernel='DL-WL', hmax=args.hmax, delta=delta)
        accs, params = [], []
        for run in range(args.runs):
            acc, p = evaluate_holdout(grams, y, seed=run)
            accs.append(acc)
            params.append(p)
            print(f'{args.dataset} slice {s} run {run}: {acc:.3f} (h={p["h"]}, C={p["C"]})', flush=True)
        save_results(f'{args.out}/{args.dataset}_slice{s}.json', accs, dataset=args.dataset, slice=s,
                     selected_params=params, delta=delta, params=vars(args))


if __name__ == '__main__':
    main()
