#!/usr/bin/env bash
# Single end-to-end run used to validate the pipeline.
# Trains XLM-R-base on Luganda at N=250 with seed=42.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python -m src.train \
  --model-config configs/xlm-r.yaml \
  --lang lug --n 250 --seed 42
