# Weisfeiler and Leman Follow the Arrow of Time

Code for the NeurIPS 2026 paper *Weisfeiler and Leman Follow the Arrow of Time: Expressive Power of Message
Passing in Temporal Event Graphs* (Franziska Heeg, Jonas Sauer, Petra Mutzel, Ingo Scholtes).

The repository contains

* the construction of temporal event graphs, augmented event graphs (Definition 8) and compressed augmented
  event graphs (Definition 10),
* the message passing architecture of Section 5 (Dir-GNN on the augmented event graph),
* the scripts that build the classification tasks and reproduce all experiments of Section 6 and the appendix,
* the code used to run the baselines (GAT, TGAT, TGN, CAW, PINT, DBGNN, DL-WL kernel).

## Installation

The experiments were run with Python 3.10, PyTorch 2.4.1 (CUDA 12.1) and PyTorch Geometric 2.5.1 (Table 9).

```bash
conda create -n egnn python=3.10
conda activate egnn
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.1
pip install "pathpyG @ git+https://github.com/pathpy/pathpyG.git@v0.2.0-alpha"
pip install -e .
```

The synthetic data generator uses `pathpyG.processes`, which was removed from pathpyG after `v0.2.0-alpha`;
please install exactly this tag. All commands below are run from the repository root.

## Layout

```
egnn/                     the method
  event_graphs.py         temporal / augmented / compressed event graphs, time-concatenated graph
  model.py                EGNN (Dir-GNN on the augmented event graph)
  training.py             data loading and training loop
  shuffling.py            timestamp randomisation (Methods A, B, C of Appendix E)
  synthetic.py            synthetic cluster model (Scholtes et al., 2014)
  node_labels.py          EGNN with temporal node labels for the dissemination benchmarks
  datasets.py             dataset names, waiting times and directory layout
scripts/                  data generation and experiments
baselines/
  gat/                    GAT on the time-concatenated graph
  dbgnn/                  De Bruijn GNN
  dlwl/                   DL-WL temporal graph kernel (with the vendored reference implementation tgkernel)
  tgnn/                   TGN, CAW (forks of the reference implementations; TGAT, PINT follow) + embedding classifier
data/raw/                 empirical temporal graphs (one timestamped edge per line)
```

## Data

`data/raw/` contains the five empirical temporal graphs of Table 3: `ants_1_1` and `ants_1_2` [Blonder and
Dornhaus, 2011], `sociopatterns_workplace` [Génois and Barrat, 2018], `sociopatterns_hospital` [Vanhems et al.,
2013] and `eu_email_dept2` [Paranjape et al., 2017]. The temporal dissemination benchmarks (RQ4) are downloaded
from [graphlearning.io](https://www.graphlearning.io/) on first use.

Generate the classification tasks of RQ1/RQ2 (windows of 500 timestamped edges, 4 × 100 graphs per window) and
the synthetic graphs of RQ3 (250 graphs per σ):

```bash
python scripts/generate_empirical_data.py
python scripts/generate_synthetic_data.py
```

Both write to `data/processed/`. The empirical data needs about 35 GB, most of it for the time-concatenated
graphs of the GAT baseline.

## Reproducing the results

| Result | Command |
|---|---|
| Table 1, "Ours" | `python scripts/run_empirical.py --dataset <name>` |
| Table 1, GAT | `python baselines/gat/run_gat_empirical.py --dataset <name>` |
| Table 1, DBGNN | `python baselines/dbgnn/run_dbgnn.py --dataset <name>` |
| Table 1, DL-WL | `python baselines/dlwl/run_dlwl.py --dataset <name>` (build tgkernel first, see `baselines/dlwl/tgkernel/PATCHES.md`) |
| Table 1, TGAT / TGN / CAW / PINT | see [`baselines/README.md`](baselines/README.md) |
| Figure 3, Figure 6 | `python scripts/run_synthetic.py` |
| Figure 7 | `python baselines/gat/run_gat_synthetic.py` |
| Table 10 (ablation) | `python scripts/run_ablation.py` |
| Table 2 | `python scripts/run_dissemination.py --experiment table2` |
| Table 11 | `python scripts/run_dissemination.py --experiment table11` |

`<name>` is one of `ants_1_1`, `ants_1_2`, `sociopatterns_workplace`, `sociopatterns_hospital`,
`eu_email_dept2`. Every script accepts `--help`; the defaults are the settings used for the paper. Results are
written as JSON files to `results/`, and

```bash
python scripts/summarize_results.py
```

prints Table 1 and the ablation and plots the heatmaps of Figures 3, 6 and 7 to `results/figures/`.

The synthetic experiments run all 190 pairs (σ1, σ2) in one process; pass `--sigma1` and `--sigma2` to run a
single pair, e.g. as one job of a cluster array.

## Acknowledgements

The temporal graph neural network baselines in `baselines/tgnn/` are based on the reference implementations
published by the original authors. We thank them for making their code available:

* **TGAT**: Xu et al., *Inductive Representation Learning on Temporal Graphs*, ICLR 2020.
  https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
* **TGN**: Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs*, 2020.
  https://github.com/twitter-research/tgn (Apache License 2.0, see `baselines/tgnn/tgn/LICENSE`)
* **CAW**: Wang et al., *Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks*,
  ICLR 2021. https://github.com/snap-stanford/CAW (MIT License, see `baselines/tgnn/caw/LICENSE`)
* **PINT**: Souza et al., *Provably Expressive Temporal Graph Networks*, NeurIPS 2022.
  https://github.com/AaltoPML/PINT

Our code in `baselines/tgnn/{tgn,caw}` is a fork of these repositories; the changes are described in
[`baselines/README.md`](baselines/README.md). Our forks of TGAT and PINT, whose repositories do not specify a
license, will be added in a later version; `baselines/README.md` lists the changes we made to them. The DL-WL baseline uses the reference implementation `tgkernel` of
Oettershagen et al., *Temporal Graph Kernels for Classifying Dissemination Processes*, SDM 2020
(https://gitlab.com/tgpublic/tgkernel), vendored in `baselines/dlwl/tgkernel/` with the changes listed in
`PATCHES.md`. The DBGNN baseline and the temporal graph data structures use
[pathpyG](https://github.com/pathpy/pathpyG).

## License

The code of our method (`egnn/`, `scripts/` and the baseline scripts written by us) is released under the
[MIT License](LICENSE). The forked third-party code in `baselines/tgnn/` and `baselines/dlwl/tgkernel/` remains
subject to the licenses and terms of the respective original repositories. Every file we changed
in these directories starts with a notice that names the original repository and lists our changes.

## Citation

```bibtex
@inproceedings{heeg2026weisfeiler,
  title     = {Weisfeiler and Leman Follow the Arrow of Time: Expressive Power of Message Passing in Temporal Event Graphs},
  author    = {Heeg, Franziska and Sauer, Jonas and Mutzel, Petra and Scholtes, Ingo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
