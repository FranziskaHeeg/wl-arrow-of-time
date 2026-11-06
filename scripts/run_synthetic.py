"""EGNN on the synthetic cluster model (RQ3, Figure 3 and Figure 6).

Distinguishes temporal graphs generated with sigma1 from those generated with sigma2, using the
compressed augmented event graphs (or, with --representation augmented, the uncompressed ones). Without
--sigma1/--sigma2, all pairs sigma1 <= sigma2 of the grid in Figure 3 are run.

    python scripts/run_synthetic.py --sigma1 0 --sigma2 0.9
"""

import argparse
import itertools

import torch

from egnn.datasets import RESULTS_DIR, SIGMAS, sigma_str, synthetic_files
from egnn.training import fit, load_dataset
from egnn.utils import save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sigma1', type=float)
    parser.add_argument('--sigma2', type=float)
    parser.add_argument('--representation', default='compressed', choices=['compressed', 'augmented'])
    parser.add_argument('--runs', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=149)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--layers', type=int, nargs='+', default=[32, 16, 8])
    parser.add_argument('--lin_layers', type=int, nargs='+', default=[8, 4])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(RESULTS_DIR / 'synthetic' / 'EGNN'))
    parser.add_argument('--quiet', action='store_true', help='do not print the per-epoch training output')
    args = parser.parse_args()

    if args.sigma1 is not None and args.sigma2 is not None:
        pairs = [(args.sigma1, args.sigma2)]
    else:
        pairs = list(itertools.combinations_with_replacement(SIGMAS, 2))

    for sigma1, sigma2 in pairs:
        files = synthetic_files(sigma1, 1, args.representation) + synthetic_files(sigma2, 0, args.representation)
        dataset = load_dataset(files)
        accs = []
        for run in range(args.runs):
            torch.manual_seed(args.seed + run)
            acc, _, _ = fit(dataset, hidden_dims=args.layers, dense_dims=args.lin_layers, lr=args.lr,
                            weight_decay=args.weight_decay, epochs=args.epochs, batch_size=args.batch_size,
                            verbose=not args.quiet)
            accs.append(acc)
            print(f'sigma {sigma1} vs {sigma2} run {run}: {acc:.3f}', flush=True)

        save_results(f'{args.out}/sigma{sigma_str(sigma1)}_vs_sigma{sigma_str(sigma2)}.json', accs,
                     sigma1=sigma1, sigma2=sigma2, params=vars(args))


if __name__ == '__main__':
    main()
