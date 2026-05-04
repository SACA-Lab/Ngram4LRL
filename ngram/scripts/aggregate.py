"""
Aggregate raw trial results into summary statistics.

Reads  : results/raw_results.csv  (written by modal_app/run_experiments.py)
Writes : results/summary.csv          — mean ± std test_f1 per (lang, scale, clf)
         results/feature_selection.csv — dominant feature type per cell

Usage
─────
python scripts/aggregate.py
python scripts/aggregate.py --input results/raw_results.csv --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCALE_ORDER = ["100", "250", "500", "750", "full"]
LANG_DISPLAY = {"lug": "Luganda", "run": "Rundi", "sna": "chiShona", "swh": "Kiswahili"}
CLF_DISPLAY  = {"naive_bayes": "NaiveBayes", "svm": "SVM", "xgboost": "XGBoost"}


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["scale"] = df["scale"].astype(str)

    summary = (
        df.groupby(["lang", "scale", "classifier"])["test_f1"]
        .agg(mean_f1="mean", std_f1="std", n_trials="count")
        .reset_index()
        .round({"mean_f1": 4, "std_f1": 4})
    )
    # std_f1 is NaN for "full" scale (single trial) — fill with 0
    summary["std_f1"] = summary["std_f1"].fillna(0.0)

    feature_sel = (
        df.groupby(["lang", "scale", "classifier"])["best_feature_type"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"best_feature_type": "dominant_feature"})
    )

    return summary, feature_sel


def print_pivot(summary: pd.DataFrame) -> None:
    summary["scale"] = pd.Categorical(summary["scale"], categories=SCALE_ORDER, ordered=True)
    summary["lang_name"]  = summary["lang"].map(LANG_DISPLAY)
    summary["clf_name"]   = summary["classifier"].map(CLF_DISPLAY)
    summary["mean_f1_str"] = summary.apply(
        lambda r: f"{r.mean_f1:.3f} ±{r.std_f1:.3f}", axis=1
    )
    pivot = summary.pivot_table(
        index=["lang_name", "scale"],
        columns="clf_name",
        values="mean_f1_str",
        aggfunc="first",
        sort=True,
    )
    print("\n─── Weighted F1 (mean ± std, 5 trials) ───────────────────────────────")
    print(pivot.to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="results/raw_results.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}\nRun the Modal experiments first.")

    df = pd.read_csv(inp)
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    summary, feature_sel = aggregate(df)

    summary.to_csv(out / "summary.csv", index=False)
    feature_sel.to_csv(out / "feature_selection.csv", index=False)

    print(f"Loaded {len(df)} trial records from {inp}")
    print(f"Saved summary          → {out}/summary.csv")
    print(f"Saved feature_selection → {out}/feature_selection.csv")
    print_pivot(summary)
    print("Next step: python scripts/make_tables.py")


if __name__ == "__main__":
    main()
