"""Ablation study on the synthetic cluster model, sigma1 = 0 vs. sigma2 = 0.9 (Table 10, Appendix G).

Variants:

* ``full``: the model of Section 5 on the compressed augmented event graph,
* ``no_node_labels``: node types are replaced by a one-hot encoding of the node index and all nodes are pooled,
* ``one_direction``: messages are only passed along edge directions (GraphConv instead of Dir-GNN),
* ``no_edge_weights``: the edge weights of the compressed augmented event graph are set to one.

    python scripts/run_ablation.py --variant no_node_labels
"""

import argparse

import torch

from egnn.datasets import RESULTS_DIR, synthetic_files
from egnn.training import fit, load_dataset
from egnn.utils import save_results

VARIANTS = ['full', 'no_node_labels', 'one_direction', 'no_edge_weights']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', nargs='+', default=VARIANTS, choices=VARIANTS)
    parser.add_argument('--sigma1', type=float, default=0.9)
    parser.add_argument('--sigma2', type=float, default=0.0)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=299)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--layers', type=int, nargs='+', default=[32, 16, 8])
    parser.add_argument('--lin_layers', type=int, nargs='+', default=[8, 4])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(RESULTS_DIR / 'ablation'))
    parser.add_argument('--quiet', action='store_true', help='do not print the per-epoch training output')
    args = parser.parse_args()

    for variant in args.variant:
        dataset = load_dataset(synthetic_files(args.sigma1, 1) + synthetic_files(args.sigma2, 0))
        max_nodes = max(data.num_nodes for data in dataset)
        if variant == 'no_node_labels':
            for data in dataset:
                data.x = torch.eye(data.num_nodes, max_nodes)
        if variant == 'no_edge_weights':
            for data in dataset:
                data.edge_weight = torch.ones(data.edge_index.shape[1])

        accs = []
        for run in range(args.runs):
            torch.manual_seed(args.seed + run)
            acc, _, _ = fit(dataset, hidden_dims=args.layers, dense_dims=args.lin_layers, lr=args.lr,
                            weight_decay=args.weight_decay, epochs=args.epochs, batch_size=args.batch_size,
                            test_batch_size=args.test_batch_size,
                            consider_node_labels=variant != 'no_node_labels',
                            pool_all=variant == 'no_node_labels',
                            dirGNN=variant != 'one_direction', verbose=not args.quiet)
            accs.append(acc)
            print(f'{variant} run {run}: {acc:.3f}', flush=True)

        save_results(f'{args.out}/{variant}.json', accs, variant=variant, params=vars(args))


if __name__ == '__main__':
    main()
