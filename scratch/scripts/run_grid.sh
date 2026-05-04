#!/usr/bin/env bash
# Iterate the (language, training-size, seed) grid for the from-scratch model.
#
# Slice via environment variables:
#   LANGS   space-separated ISO 639-3 codes   (default: lug run sna swa)
#   NS      space-separated sizes             (default: 100 250 500 750 full)
#   SEEDS   space-separated integers          (default: 42 123 456 789 1024)
#
# Behaviour:
#   - Idempotent: any run whose log already exists in results/logs/ is skipped.
#   - At N=full only the first seed is used (training data is deterministic).
#
# Tokenisers must be trained first via scripts/train_tokenizer.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LANGS="${LANGS:-lug run sna swa}"
NS="${NS:-100 250 500 750 full}"
SEEDS="${SEEDS:-42 123 456 789 1024}"

FIRST_SEED=$(echo "$SEEDS" | awk '{print $1}')
SHORT_NAME=$(python -c "import yaml; print(yaml.safe_load(open('configs/base.yaml'))['short_name'])")

for lang in $LANGS; do
  if [[ ! -f "results/tokenizers/${lang}.json" ]]; then
    echo "[error] tokeniser missing for ${lang}; run scripts/train_tokenizer.py first"
    exit 1
  fi
  for n in $NS; do
    for seed in $SEEDS; do
      if [[ "$n" == "full" && "$seed" != "$FIRST_SEED" ]]; then
        continue
      fi
      run_id="${SHORT_NAME}_${lang}_${n}_${seed}"
      marker="results/logs/${run_id}.json"
      if [[ -f "$marker" ]]; then
        echo "[skip] $run_id"
        continue
      fi
      echo "[run]  $run_id"
      python -m src.train --lang "$lang" --n "$n" --seed "$seed"
    done
  done
done
