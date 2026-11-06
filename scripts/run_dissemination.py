"""EGNN on the temporal dissemination benchmarks of Oettershagen et al. (RQ4, Table 2 and Table 11).

The raw TU datasets are downloaded to ~/.pathpyG/tu_datasets on first use.

    python scripts/run_dissemination.py --experiment table2
    python scripts/run_dissemination.py --experiment table11
"""

import argparse
import json
from pathlib import Path

import numpy as np

from egnn.datasets import RESULTS_DIR
from egnn.node_labels import causal_consistency_baseline, load_dataset, run_experiment

TABLE2_DATASETS = ['highschool_ct1', 'highschool_ct2', 'infectious_ct1', 'mit_ct1', 'tumblr_ct1']
TABLE11_MODES = ['structure', 'static', 'temporal', 'transition']


def run(name, feature_mode, representation, seeds, epochs, delta, results, verbose=True):
    readout = 'both' if representation == 'event_graph' else 'first_order'
    dataset, meta = load_dataset(name, delta=delta, feature_mode=feature_mode, representation=representation, verbose=False)
    accs = []
    for seed in seeds:
        r = run_experiment(name, delta=delta, feature_mode=feature_mode, representation=representation,
                           epochs=epochs, seed=seed, dataset=dataset, meta=meta, readout=readout, verbose=verbose)
        r.pop('history')
        results.append(r)
        accs.append(r['test_acc'])
    print(f'{name:16s} | {representation:11s} | {feature_mode:10s} | test {np.mean(accs):.3f} ± {np.std(accs):.3f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=['table2', 'table11'], default='table2')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--delta', type=int, default=1)
    parser.add_argument('--out', default=str(RESULTS_DIR / 'dissemination'))
    parser.add_argument('--quiet', action='store_true', help='do not print the per-epoch training output')
    args = parser.parse_args()

    results = []
    if args.experiment == 'table2':
        # node type only vs. node type + temporal node labels, on the augmented event graph
        for name in TABLE2_DATASETS:
            for mode in ['structure', 'transition']:
                run(name, mode, 'event_graph', args.seeds, args.epochs, args.delta, results, not args.quiet)
        extra = {}
    else:
        # nested feature ablation on highschool_ct1, augmented event graph vs. time-aggregated graph
        for representation in ['event_graph', 'static']:
            for mode in TABLE11_MODES:
                run('highschool_ct1', mode, representation, args.seeds, args.epochs, args.delta, results, not args.quiet)
        extra = {'causal_consistency_baseline': causal_consistency_baseline('highschool_ct1')}

    path = Path(args.out) / f'{args.experiment}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'runs': results, **extra}, indent=1, default=float))


if __name__ == '__main__':
    main()
