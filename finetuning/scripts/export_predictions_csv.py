"""Consolidate per-run prediction parquet files into one CSV of test-set
probability scores, for hybrid n-gram/neural interpolation (see
hybrid/run_hybrid.py in Ngram4LRL).

Reads every results/<dataset>/predictions/{model}_{lang}_{n}_{seed}.parquet
and stacks them into a single tidy CSV with one row per (model, lang, n,
seed, test instance): true_label, pred_label, and p_<class> for every class.
Raw text and logits are dropped -- only what's needed for interpolation.

Usage:
    python -m scripts.export_predictions_csv --dataset afrisenti
    python -m scripts.export_predictions_csv --dataset sib200
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("masakhanews", "afrisenti", "sib200")

RUN_ID_RE = re.compile(r"^(?P<model>[a-z0-9]+)_(?P<lang>[a-z]+)_(?P<n>\d+|full)_(?P<seed>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="masakhanews", choices=DATASET_NAMES)
    args = ap.parse_args()

    results_dir = REPO_ROOT / "results"
    if args.dataset != "masakhanews":
        results_dir = results_dir / args.dataset
    pred_dir = results_dir / "predictions"

    files = sorted(pred_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No prediction files found in {pred_dir}")

    rows = []
    skipped = []
    for f in files:
        m = RUN_ID_RE.match(f.stem)
        if not m:
            skipped.append(f.name)
            continue
        df = pd.read_parquet(f)
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        df = df[["id", "true_label", "pred_label", *prob_cols]].copy()
        df.insert(0, "seed", int(m["seed"]))
        df.insert(0, "n", m["n"])
        df.insert(0, "lang", m["lang"])
        df.insert(0, "model", m["model"])
        rows.append(df)

    if skipped:
        print(f"skipped {len(skipped)} file(s) with unexpected names: {skipped}")

    out = pd.concat(rows, ignore_index=True)
    out_path = results_dir / "predictions_probs.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows ({len(files) - len(skipped)} runs) -> {out_path.relative_to(REPO_ROOT)}")
    print(f"models: {sorted(out['model'].unique())}")
    print(f"langs:  {sorted(out['lang'].unique())}")
    print(f"n:      {sorted(out['n'].unique(), key=lambda x: (x != 'full', x))}")
    print(f"seeds:  {sorted(out['seed'].unique())}")


if __name__ == "__main__":
    main()
