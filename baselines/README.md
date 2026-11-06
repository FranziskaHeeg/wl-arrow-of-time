# Baselines

All baselines solve the same task as our model: separating classes 1/3 from classes 2/4 of each window of the
empirical temporal graphs. Generate the data first (`python scripts/generate_empirical_data.py`) and run all
commands from the repository root.

## GAT (`gat/`)

GAT on the time-concatenated static graph with the timestamps as edge attributes and a randomly permuted one-hot
encoding of the nodes as node features. `run_gat_empirical.py` produces Table 1, `run_gat_synthetic.py` Figure 7.
`GATConv` is used without `edge_dim`, in which case PyTorch Geometric does not use the edge attributes.

## DBGNN (`dbgnn/`)

The De Bruijn GNN of Qarkaxhija et al. (2022), using the implementation in pathpyG, with first- and
second-order message passing, the bipartite fusion layer, mean pooling and an MLP head. Node features are
log-degrees and log-strengths in the first- and second-order De Bruijn graphs.

## DL-WL kernel (`dlwl/`)

The directed line graph WL kernel of Oettershagen et al. (2020), computed with their reference implementation
(vendored in `dlwl/tgkernel/`, with the changes listed in `dlwl/tgkernel/PATCHES.md`), and a C-SVM on the
precomputed kernel. The number of WL iterations h ∈ {0, …, 5} and C ∈ {10^-3, …, 10^3} are chosen by 5-fold
cross-validation on the training split. Build the binary once:

```bash
export EIGEN3_INCLUDE_DIR=/path/to/eigen3
cd baselines/dlwl/tgkernel && mkdir -p release && cd release
cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

## TGAT, TGN, CAW, PINT (`tgnn/`)

`tgnn/{tgat,tgn,caw,pint}` are forks of the reference implementations

* TGAT: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
* TGN: https://github.com/twitter-research/tgn
* CAW: https://github.com/snap-stanford/CAW
* PINT: https://github.com/AaltoPML/PINT

with one addition each: after the self-supervised link prediction training, the temporal embedding of the source
node of every event is saved to `embeddings/`. Smaller changes: PINT uses a 50/25/25 temporal split and 16-dim
zero node features, TGN uses 16-dim zero node features (`utils/preprocess_data.py`), and TGN and PINT do not save
the trained model. Each model has its own dependencies, see the README and environment files in its
directory.

The pipeline has two steps per model.

1. Export every temporal graph as an interaction file, train the model self-supervised on it with the settings
   in the script and save the embeddings:

   ```bash
   bash baselines/tgnn/tgat/run_pipeline.sh
   bash baselines/tgnn/tgn/run_pipeline.sh
   bash baselines/tgnn/caw/run_pipeline.sh
   bash baselines/tgnn/pint/run_pipeline.sh
   ```

   This trains one model per temporal graph (14,400 graphs) and is best parallelised over the loop in the script.

2. Train an MLP on the embeddings of each graph (`tgnn/classify_embeddings.py`), sweeping how the temporal
   embeddings are aggregated per node (all / last / mean), the graph readout (mean / max / concatenation), the
   learning rate (0.01 / 0.001) and the hidden layers (128 / 128-64), with 3 repetitions each:

   ```bash
   bash baselines/tgnn/sweep_classification.sh tgat 100
   bash baselines/tgnn/sweep_classification.sh tgn 100
   bash baselines/tgnn/sweep_classification.sh caw 50
   bash baselines/tgnn/sweep_classification.sh pint 50
   ```

   `python scripts/summarize_results.py` then reports, per model and dataset, the configuration with the best
   mean accuracy (Tables 5–8).
