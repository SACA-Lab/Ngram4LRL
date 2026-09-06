#!/usr/bin/env bash
# Iterate the (language, training-size, seed) grid for the from-scratch model.
#
# Slice via environment variables:
#   DATASET  masakhanews | afrisenti | sib200   (default: masakhanews)
#   LANGS    space-separated ISO 639-3 codes    (default: all of $DATASET's languages)
#   NS       space-separated sizes              (default: 100 250 500 750 full)
#   SEEDS    space-separated integers           (default: 42 123 456 789 1024)
#
# Behaviour:
#   - Idempotent: any run whose log already exists in results/[<DATASET>/]logs/
#     is skipped.
#   - At N=full only the first seed is used (training data is deterministic).
#
# Tokenisers must be trained first via `scripts/train_tokenizer.py [--dataset ...]`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-masakhanews}"
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
SHORT_NAME=$(python -c "import yaml; print(yaml.safe_load(open('configs/base.yaml'))['short_name'])")

for lang in $LANGS; do
  if [[ ! -f "${RESULTS_DIR}/tokenizers/${lang}.json" ]]; then
    echo "[error] tokeniser missing for ${lang}; run 'python -m scripts.train_tokenizer --dataset ${DATASET}' first"
    exit 1
  fi
  for n in $NS; do
    for seed in $SEEDS; do
      if [[ "$n" == "full" && "$seed" != "$FIRST_SEED" ]]; then
        continue
      fi
      run_id="${SHORT_NAME}_${lang}_${n}_${seed}"
      marker="${RESULTS_DIR}/logs/${run_id}.json"
      if [[ -f "$marker" ]]; then
        echo "[skip] $run_id"
        continue
      fi
      echo "[run]  $run_id"
      python -m src.train --dataset "$DATASET" --lang "$lang" --n "$n" --seed "$seed"
    done
  done
done
