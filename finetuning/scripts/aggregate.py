"""Aggregate per-run results into per-(model, language, N) summary tables.

Reads ``results/metrics.csv`` and prints one weighted-F1 table per model
plus a long-form view.

Usage:
    python -m scripts.aggregate
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = REPO_ROOT / "results" / "metrics.csv"

N_ORDER = ["100", "250", "500", "750", "full"]
LANG_ORDER = ["lug", "run", "sna", "swa"]


def main() -> None:
    if not METRICS.exists():
        print(f"no metrics file at {METRICS}")
        return

    df = pd.read_csv(METRICS)
    if df.empty:
        print("metrics.csv is empty")
        return

    df["n"] = df["n"].astype(str)

    grouped = (
        df.groupby(["model", "lang", "n"])["test_f1_weighted"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["cell"] = grouped.apply(
        lambda r: f"{r['mean']:.3f} +/- {r['std']:.3f} (k={int(r['count'])})",
        axis=1,
    )

    for model in sorted(grouped["model"].unique()):
        sub = grouped[grouped["model"] == model]
        pivot = sub.pivot(index="lang", columns="n", values="cell")
        cols = [n for n in N_ORDER if n in pivot.columns]
        rows = [l for l in LANG_ORDER if l in pivot.index]
        pivot = pivot.reindex(index=rows, columns=cols)
        print(f"\n{model} - test weighted F1 (mean +/- std over seeds)")
        print(pivot.fillna("-").to_string())

    print("\nlong form")
    print(grouped[["model", "lang", "n", "mean", "std", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
