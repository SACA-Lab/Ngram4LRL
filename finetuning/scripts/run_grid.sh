#!/usr/bin/env bash
# Iterate the (model, language, training-size, seed) grid.
#
# Slice via environment variables:
#   DATASET  masakhanews | afrisenti | sib200   (default: masakhanews)
#   MODELS   space-separated model config names (default: xlm-r afroxlmr)
#   LANGS    space-separated ISO 639-3 codes    (default: all of $DATASET's languages)
#   NS       space-separated sizes              (default: 100 250 500 750 full)
#   SEEDS    space-separated integers           (default: 42 123 456 789 1024)
#
# Behaviour:
#   - Idempotent: any run whose log already exists in results/[<DATASET>/]logs/
#     is skipped.
#   - At N=full only the first seed is used (training data is deterministic).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-masakhanews}"
MODELS="${MODELS:-xlm-r afroxlmr}"
LANGS="${LANGS:-$(python -c "
import yaml
cfg = yaml.safe_load(open('configs/base.yaml'))
if '$DATASET' == 'masakhanews':
    langs = cfg['languages']
else:
    langs = yaml.safe_load(open('configs/${DATASET}.yaml'))['lang_to_subset']
print(' '.join(langs))
")}"
NS="${NS:-100 250 500 750 full}"
SEEDS="${SEEDS:-42 123 456 789 1024}"

RESULTS_DIR="results"
if [[ "$DATASET" != "masakhanews" ]]; then
  RESULTS_DIR="results/${DATASET}"
fi

FIRST_SEED=$(echo "$SEEDS" | awk '{print $1}')

for model in $MODELS; do
  # Read the short_name from the model config; train.py uses it as the
  # run_id prefix, so the skip check must use the same value.
  short_name=$(python -c "import yaml; print(yaml.safe_load(open('configs/${model}.yaml'))['short_name'])")
  for lang in $LANGS; do
    for n in $NS; do
      for seed in $SEEDS; do
        if [[ "$n" == "full" && "$seed" != "$FIRST_SEED" ]]; then
          continue
        fi
        run_id="${short_name}_${lang}_${n}_${seed}"
        marker="${RESULTS_DIR}/logs/${run_id}.json"
        if [[ -f "$marker" ]]; then
          echo "[skip] $run_id"
          continue
        fi
        echo "[run]  $run_id"
        python -m src.train \
          --model-config "configs/${model}.yaml" \
          --dataset "$DATASET" \
          --lang "$lang" --n "$n" --seed "$seed"
      done
    done
  done
done
