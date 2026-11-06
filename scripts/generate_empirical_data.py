"""Builds the classification tasks on empirical temporal graphs (RQ1/RQ2, Appendix E).

Each dataset is cut into windows of 500 timestamped edges. For each window, 100 graphs are generated
per class:

* class 1: Method A (random shift) applied to the window,
* class 2: Method C (timestamp shuffling) applied to class 1,
* class 3: Method B (component reordering) applied to class 1,
* class 4: Method B applied to class 2.

Writes the temporal graphs, their augmented event graphs and their time-concatenated static graphs
(see `egnn/datasets.py` for the layout).

    python scripts/generate_empirical_data.py --datasets ants_1_1 ants_1_2
"""

import argparse
import json
import pickle
import random

import numpy as np
import pathpyG as pp

from egnn.datasets import EMPIRICAL_DATASETS, PROCESSED_DIR, RAW_DIR
from egnn.event_graphs import augmented_event_graph, time_concatenated_graph
from egnn.shuffling import get_slices, shift_randomly, shuffle, shuffle_components


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def generate(dataset: str, delta: int, graphs_per_class: int, window: int):
    g = pp.io.read_csv_temporal_graph(str(RAW_DIR / f'{dataset}.tedges'), header=False)
    slices = get_slices(g, window)
    print(f'{dataset}: {len(slices)} slices')

    for s, slice_graph in enumerate(slices):
        folder = PROCESSED_DIR / 'temporal' / dataset / f'slice{s}'
        cc_class1, cc_class2, cc_sizes_class1, cc_sizes_class2 = [], [], [], []

        for i in range(graphs_per_class):
            g1 = shift_randomly(slice_graph, delta=delta)
            g3, n3, sizes3 = shuffle_components(g1, delta=delta)
            g2 = shuffle(g1)  # in place: g1 is not used afterwards
            g4, n4, sizes4 = shuffle_components(g2, delta=delta)

            cc_class1.append(n3)
            cc_class2.append(n4)
            cc_sizes_class1.append((np.mean(sizes3), np.std(sizes3)))
            cc_sizes_class2.append((np.mean(sizes4), np.std(sizes4)))

            for c, graph in ((1, g1), (2, g2), (3, g3), (4, g4)):
                name = f'class{c}/{dataset}_slice{s}_c{c}_{i}.pickle'
                dump(graph, folder / name)
                dump(augmented_event_graph(graph, delta), PROCESSED_DIR / 'augmented' / dataset / f'slice{s}' / name)
                dump(time_concatenated_graph(graph), PROCESSED_DIR / 'concat' / dataset / f'slice{s}' / name)

        meta = {
            'delta': delta,
            'num_nodes': slice_graph.n,
            'num_edges': slice_graph.m,
            'Number of CC in class1/3': cc_class1,
            'Number of CC in class2/4': cc_class2,
            'Mean/std Number of CC in class1/3': (np.mean(cc_class1), np.std(cc_class1)),
            'Mean/std Number of CC in class2/4': (np.mean(cc_class2), np.std(cc_class2)),
            'Mean/std Nodes in CC in class1/3': cc_sizes_class1,
            'Mean/std Nodes in CC in class2/4': cc_sizes_class2,
        }
        with open(folder / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=4, default=float)
        print(f'{dataset} slice {s} done', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=list(EMPIRICAL_DATASETS), choices=list(EMPIRICAL_DATASETS))
    parser.add_argument('--graphs_per_class', type=int, default=100)
    parser.add_argument('--window', type=int, default=500, help='timestamped edges per slice')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    for dataset in args.datasets:
        generate(dataset, EMPIRICAL_DATASETS[dataset], args.graphs_per_class, args.window)


if __name__ == '__main__':
    main()
