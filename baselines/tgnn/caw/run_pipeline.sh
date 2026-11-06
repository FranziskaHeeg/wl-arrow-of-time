#!/bin/bash
# Added by Franziska Heeg, Jonas Sauer, Petra Mutzel, Ingo Scholtes (2026) for "Weisfeiler and Leman Follow the Arrow of Time" (NeurIPS 2026);
# not part of the original repository. Released under the MIT License (see LICENSE at the repository root).
# CAW: export the empirical tasks, train CAW-N self-supervised on every temporal graph and save its
# temporal node embeddings to embeddings/. Run from the release root:  bash baselines/tgnn/caw/run_pipeline.sh
set -euo pipefail
ROOT="$(pwd)"
cd "$(dirname "$0")"
mkdir -p processed embeddings results log saved_checkpoints best_models

python "$ROOT/baselines/tgnn/convert_data.py" --format tgat --out processed

for csv in processed/ml_*.csv; do
  id="$(basename "$csv" .csv)"; id="${id#ml_}"
  [[ -f "embeddings/CAWN_${id}_embedding.npy" ]] && continue
  python main.py -d "$id" --pos_dim 64 --bs 32 --n_degree 8 1 --mode i --bias 1e-5 --pos_enc lp \
    --walk_pool sum --seed 0 --embedding true
done
