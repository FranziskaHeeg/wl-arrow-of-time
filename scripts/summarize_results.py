"""Aggregates the results written by the run scripts.

* Table 1: mean ± std accuracy per model and empirical dataset, over all runs of all slices. For the
  TGNN baselines, the classifier configuration with the best mean accuracy per dataset is reported.
* Figures 3, 6 and 7: heatmaps of the mean and standard deviation of the accuracy for all sigma pairs.

    python scripts/summarize_results.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from egnn.datasets import EMPIRICAL_DATASETS, RESULTS_DIR, SIGMAS

TGNN_MODELS = ['tgat', 'tgn', 'caw', 'pint']


def dataset_of(name: str) -> str:
    return re.sub(r'_slice\d+$', '', name)


def table1():
    rows = {}
    empirical = RESULTS_DIR / 'empirical'
    if not empirical.exists():
        return rows
    for model_dir in sorted(p for p in empirical.iterdir() if p.is_dir()):
        accs = defaultdict(list)
        if model_dir.name in TGNN_MODELS:
            # (dataset, config) -> accuracies; pick the best config per dataset
            per_config = defaultdict(list)
            for f in model_dir.rglob('exp_*.json'):
                r = json.loads(f.read_text())
                config = json.dumps({k: v for k, v in r['params'].items() if k != 'seed'}, sort_keys=True)
                per_config[(dataset_of(r['name']), config)].append(r['metrics']['accuracy'])
            for dataset in EMPIRICAL_DATASETS:
                candidates = [v for (d, _), v in per_config.items() if d == dataset]
                if candidates:
                    accs[dataset] = max(candidates, key=np.mean)
        else:
            for f in model_dir.glob('*.json'):
                r = json.loads(f.read_text())
                accs[r['dataset']] += r['accuracies']
        rows[model_dir.name] = {d: (np.mean(a), np.std(a), len(a)) for d, a in accs.items()}
    return rows


def heatmaps(model: str):
    folder = RESULTS_DIR / 'synthetic' / model
    if not folder.exists():
        return
    idx = {s: i for i, s in enumerate(SIGMAS)}
    mean = np.full((len(SIGMAS), len(SIGMAS)), np.nan)
    std = np.full_like(mean, np.nan)
    for f in folder.glob('*.json'):
        r = json.loads(f.read_text())
        i, j = idx[round(r['sigma1'], 1)], idx[round(r['sigma2'], 1)]
        mean[i, j] = mean[j, i] = r['mean']
        std[i, j] = std[j, i] = r['std']

    for values, label, suffix in ((mean, 'mean accuracy', 'mean'), (std, 'standard deviation', 'std')):
        fig, ax = plt.subplots(figsize=(4, 3.4))
        im = ax.imshow(values.T, origin='lower', cmap='viridis',
                       extent=(SIGMAS[0] - 0.05, SIGMAS[-1] + 0.05, SIGMAS[0] - 0.05, SIGMAS[-1] + 0.05))
        ax.set(xlabel=r'$\sigma_1$', ylabel=r'$\sigma_2$', title=model)
        fig.colorbar(im, ax=ax, label=label)
        fig.tight_layout()
        out = RESULTS_DIR / 'figures' / f'synthetic_{model}_{suffix}.pdf'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        print(f'saved {out}')


def main():
    rows = table1()
    if rows:
        datasets = list(EMPIRICAL_DATASETS)
        print('model'.ljust(10) + ''.join(d[:22].rjust(24) for d in datasets))
        for model, res in rows.items():
            cells = [f'{res[d][0]:.3f} ± {res[d][1]:.3f} (n={res[d][2]})' if d in res else '-' for d in datasets]
            print(model.ljust(10) + ''.join(c.rjust(24) for c in cells))

    for model in ('EGNN', 'GAT', 'GAT-timestamps'):
        heatmaps(model)

    ablation = RESULTS_DIR / 'ablation'
    if ablation.exists():
        print('\nablation (Table 10)')
        for f in sorted(ablation.glob('*.json')):
            r = json.loads(f.read_text())
            print(f'  {f.stem:16s} {r["mean"]:.2f} ± {r["std"]:.2f}')


if __name__ == '__main__':
    main()
