"""Aggregate per-run results into per-(language, N) summary tables.

Reads results/metrics.csv (or results/<dataset>/metrics.csv for afrisenti and
sib200) and prints the weighted-F1 table plus a long-form view.

Usage:
    python -m scripts.aggregate
    python -m scripts.aggregate --dataset afrisenti
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("masakhanews", "afrisenti", "sib200")

N_ORDER_BY_DATASET = {
    "masakhanews": ["100", "250", "500", "750", "full"],
    "afrisenti": ["100", "250", "500", "750", "full"],
    "sib200": ["100", "250", "500", "full"],
}
LANG_ORDER_BY_DATASET = {
    "masakhanews": ["lug", "run", "sna", "swa", "amh", "ibo", "yor", "orm", "pcm", "hau"],
    "afrisenti": ["amh", "hau", "ibo", "orm", "pcm", "swa", "yor"],
    "sib200": ["hau", "ibo", "lug", "gaz", "run", "sna", "swh", "yor", "amh"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="masakhanews", choices=DATASET_NAMES)
    args = ap.parse_args()

    results_dir = REPO_ROOT / "results"
    if args.dataset != "masakhanews":
        results_dir = results_dir / args.dataset
    metrics_path = results_dir / "metrics.csv"

    if not metrics_path.exists():
        print(f"no metrics file at {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    if df.empty:
        print("metrics.csv is empty")
        return

    df["n"] = df["n"].astype(str)
    n_order = N_ORDER_BY_DATASET[args.dataset]
    lang_order = LANG_ORDER_BY_DATASET[args.dataset]

    grouped = (
        df.groupby(["lang", "n"])["test_f1_weighted"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["cell"] = grouped.apply(
        lambda r: f"{r['mean']:.3f} +/- {r['std']:.3f} (k={int(r['count'])})",
        axis=1,
    )

    pivot = grouped.pivot(index="lang", columns="n", values="cell")
    cols = [n for n in n_order if n in pivot.columns]
    rows = [l for l in lang_order if l in pivot.index]
    pivot = pivot.reindex(index=rows, columns=cols)
    print("scratch transformer - test weighted F1 (mean +/- std over seeds)")
    print(pivot.fillna("-").to_string())

    print("\nlong form")
    print(grouped[["lang", "n", "mean", "std", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
