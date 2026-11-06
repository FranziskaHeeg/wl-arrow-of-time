"""GAT on the time-concatenated static graphs of the synthetic cluster model (Figure 7).

Without --sigma1/--sigma2, all pairs sigma1 <= sigma2 of the grid in Figure 7 are run.

    python baselines/gat/run_gat_synthetic.py --sigma1 0 --sigma2 0.9
"""

import argparse
import itertools

import torch

from egnn.datasets import RESULTS_DIR, SIGMAS, sigma_str, synthetic_files
from egnn.utils import save_results
from gat import fit, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sigma1', type=float)
    parser.add_argument('--sigma2', type=float)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None, help='default: results/.../GAT or GAT-timestamps')
    parser.add_argument('--use_timestamps', action='store_true',
                        help='use the normalised timestamps as edge features (edge_dim=1)')
    parser.add_argument('--quiet', action='store_true', help='do not print the per-epoch training output')
    args = parser.parse_args()
    args.out = args.out or str(RESULTS_DIR / 'synthetic' / ('GAT-timestamps' if args.use_timestamps else 'GAT'))

    if args.sigma1 is not None and args.sigma2 is not None:
        pairs = [(args.sigma1, args.sigma2)]
    else:
        pairs = list(itertools.combinations_with_replacement(SIGMAS, 2))

    for sigma1, sigma2 in pairs:
        accs = []
        for run in range(args.runs):
            torch.manual_seed(args.seed + run)
            files = synthetic_files(sigma1, 1, 'concat') + synthetic_files(sigma2, 0, 'concat')
            dataset = load_dataset(files, use_timestamps=args.use_timestamps)
            accs.append(fit(dataset, epochs=args.epochs, lr=args.lr, use_timestamps=args.use_timestamps,
                            verbose=not args.quiet)['accuracy'])
            print(f'sigma {sigma1} vs {sigma2} run {run}: {accs[-1]:.3f}', flush=True)
        save_results(f'{args.out}/sigma{sigma_str(sigma1)}_vs_sigma{sigma_str(sigma2)}.json', accs,
                     sigma1=sigma1, sigma2=sigma2, params=vars(args))


if __name__ == '__main__':
    main()
