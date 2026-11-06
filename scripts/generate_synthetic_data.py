"""Generates the synthetic temporal graphs of the cluster model (RQ3, Appendix E).

A single static graph with two 3-regular communities of 10 nodes, linked by two edges, is drawn once.
For each sigma, 250 temporal graphs are generated from 500 second-order random walks of length two
(1000 timestamped edges each). Writes the temporal graphs, their compressed augmented event graphs
(delta = 1, Appendix B) and their time-concatenated static graphs (for the GAT baseline).

    python scripts/generate_synthetic_data.py
"""

import argparse
import pickle
import random

import numpy as np
import torch

from egnn.datasets import PROCESSED_DIR, SIGMAS, sigma_str
from egnn.event_graphs import augmented_event_graph_compressed, time_concatenated_graph
from egnn.synthetic import natcom_model_fo, temporal_graph_from_natcom


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sigmas', type=float, nargs='+', default=SIGMAS)
    parser.add_argument('--graphs', type=int, default=250, help='temporal graphs per sigma')
    parser.add_argument('--walks', type=int, default=500, help='random walks per temporal graph')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n', type=int, default=10, help='nodes per community')
    parser.add_argument('--inter_com', type=int, default=1, help='number of rewired edge pairs between communities')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = PROCESSED_DIR / 'synthetic'
    g, inter_com_edges = natcom_model_fo(k=args.k, n=args.n, inter_com=args.inter_com)
    dump((g, inter_com_edges), root / 'static_graph.pkl')

    for sigma in args.sigmas:
        for i in range(args.graphs):
            name = f'sigma{sigma_str(sigma)}/net{i}.pkl'
            # steps=1 on the second-order graph yields walks of two timestamped edges
            temp_net = temporal_graph_from_natcom(g, inter_com_edges, sigma=sigma, runs=args.walks, steps=1)
            dump(temp_net, root / 'temporal' / name)
            dump(augmented_event_graph_compressed(temp_net, delta=1), root / 'compressed' / name)
            dump(time_concatenated_graph(temp_net), root / 'concat' / name)
        print(f'sigma {sigma} done', flush=True)


if __name__ == '__main__':
    main()
