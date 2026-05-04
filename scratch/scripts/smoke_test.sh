#!/usr/bin/env bash
# Single end-to-end run used to validate the pipeline.
# Trains the from-scratch transformer on Luganda at N=250 with seed=42.
# Requires that the Luganda tokeniser has already been trained.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "results/tokenizers/lug.json" ]]; then
  echo "[info] training tokenisers"
  python -m scripts.train_tokenizer
fi

python -m src.train --lang lug --n 250 --seed 42
