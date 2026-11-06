#!/bin/bash
# TGN: export the empirical tasks, train the model self-supervised on every temporal graph and save its
# temporal node embeddings to embeddings/. Run from the release root:  bash baselines/tgnn/tgn/run_pipeline.sh
#
# The flags below (--use_memory --memory_updater rnn --embedding_module time) are the configuration used
# for the results in the paper.
set -euo pipefail
ROOT="$(pwd)"
cd "$(dirname "$0")"
mkdir -p data embeddings

python "$ROOT/baselines/tgnn/convert_data.py" --format tgn --out data

for csv in data/*.csv; do
  id="$(basename "$csv" .csv)"
  [[ "$id" == ml_* ]] && continue
  [[ -f "embeddings/TNG_${id}_embedding.npy" ]] && continue
  [[ -f "data/ml_${id}.csv" ]] || python utils/preprocess_data.py --data "$id"
  python train_self_supervised.py --use_memory --memory_updater rnn --embedding_module time \
    --prefix jodie_rnn --n_runs 1 --data "$id" --memory_dim 16
done
