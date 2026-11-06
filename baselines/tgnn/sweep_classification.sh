#!/bin/bash
# Hyperparameter sweep of the embedding classifier for one TGNN (Tables 5-8), 3 repetitions per configuration.
# Run from the release root after the model's run_pipeline.sh:
#
#   bash baselines/tgnn/sweep_classification.sh tgat 100
#   bash baselines/tgnn/sweep_classification.sh tgn  100
#   bash baselines/tgnn/sweep_classification.sh caw  50
#   bash baselines/tgnn/sweep_classification.sh pint 50
#
# The second argument is the number of training epochs of the classifier (the values above were used for
# the results in the paper). Results are written to results/empirical/<model>/<slice>/.
set -euo pipefail
MODEL="$1"
EPOCHS="${2:-100}"
EMB="baselines/tgnn/${MODEL}/embeddings"

slices=()
for i in 0 1 2; do slices+=("ants_1_1_slice$i" "ants_1_2_slice$i"); done
for i in $(seq 0 9); do slices+=("sociopatterns_workplace_slice$i" "sociopatterns_hospital_slice$i" "eu_email_dept2_slice$i"); done

for NAME in "${slices[@]}"; do
  cfg=0
  for MODE in last mean all; do
    for POOL in mean max all; do
      for LR in 0.01 0.001; do
        for LAYERS in "128" "128 64"; do
          for rep in 0 1 2; do
            python baselines/tgnn/classify_embeddings.py --embeddings_dir "$EMB" --model_name "$MODEL" \
              --name "$NAME" --mode "$MODE" --pooling "$POOL" --epochs "$EPOCHS" --lr "$LR" \
              --layer_sizes $LAYERS --seed $((1000 + cfg * 10 + rep))
          done
          cfg=$((cfg + 1))
        done
      done
    done
  done
done
