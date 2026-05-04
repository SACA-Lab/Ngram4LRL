"""
Cross-track comparison: n-gram baselines vs fine-tuning vs scratch transformer.

Prints a comparison table of mean weighted test F1 for each method at each
(language, training-set size). Run from the repo root:

    python analysis/compare.py
"""

from __future__ import annotations

import pathlib
import textwrap
import warnings

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent

LANG_NAMES = {
    "lug": "Luganda",
    "run": "Rundi",
    "sna": "chiShona",
    "swa": "Kiswahili",
}
SCALE_ORDER = [100, 250, 500, 750, "full"]


# ── loaders ────────────────────────────────────────────────────────────────────

def _load_ngram() -> pd.DataFrame:
    path = ROOT / "ngram" / "results" / "raw_results.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"scale": "n", "test_f1": "test_f1_weighted"})
    df["n"] = df["n"].astype(str).replace({"full": "full"})
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)
    df.loc[df["n"] == -1, "n"] = -1  # sentinel for "full"

    # Best F1 per (lang, n, seed) across classifiers
    best = (
        df.groupby(["lang", "n", "seed"])["test_f1_weighted"]
        .max()
        .reset_index()
    )
    # Mean over seeds
    agg = (
        best.groupby(["lang", "n"])["test_f1_weighted"]
        .agg(mean="mean", std="std", n_trials="count")
        .reset_index()
    )
    agg["method"] = "ngram_best"
    return agg


def _load_finetuning() -> pd.DataFrame:
    path = ROOT / "finetuning" / "results" / "metrics.csv"
    df = pd.read_csv(path)
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)

    # Best F1 per (lang, n, seed) across models
    best = (
        df.groupby(["lang", "n", "seed"])["test_f1_weighted"]
        .max()
        .reset_index()
    )
    agg = (
        best.groupby(["lang", "n"])["test_f1_weighted"]
        .agg(mean="mean", std="std", n_trials="count")
        .reset_index()
    )
    agg["method"] = "finetune_best"

    # Also report per-model
    per_model = (
        df.groupby(["model", "lang", "n"])["test_f1_weighted"]
        .agg(mean="mean", std="std", n_trials="count")
        .reset_index()
        .rename(columns={"model": "method"})
    )
    return pd.concat([agg, per_model], ignore_index=True)


def _load_scratch() -> pd.DataFrame:
    path = ROOT / "scratch" / "results" / "metrics.csv"
    df = pd.read_csv(path)
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)
    agg = (
        df.groupby(["lang", "n"])["test_f1_weighted"]
        .agg(mean="mean", std="std", n_trials="count")
        .reset_index()
    )
    agg["method"] = "scratch"
    return agg


# ── helpers ────────────────────────────────────────────────────────────────────

def _n_label(n: int) -> str:
    return "full" if n == -1 else str(n)


def _fmt(mean: float, std: float | None) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std) or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ±{std:.3f}"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ngram = _load_ngram()
        ft = _load_finetuning()
        scratch = _load_scratch()

    summary = pd.concat([ngram, ft, scratch], ignore_index=True)

    methods_main = ["ngram_best", "finetune_best", "scratch"]
    methods_detail = ["ngram_best", "xlmr", "afroxlmr", "scratch"]

    def pivot(df: pd.DataFrame, methods: list[str]) -> None:
        for lang_code, lang_name in LANG_NAMES.items():
            sub = df[df["lang"] == lang_code].copy()
            if sub.empty:
                continue
            sub = sub[sub["method"].isin(methods)]
            sub["n_label"] = sub["n"].apply(_n_label)

            rows = []
            for n_val in SCALE_ORDER:
                n_int = -1 if n_val == "full" else int(n_val)
                row: dict = {"N": str(n_val)}
                for m in methods:
                    hit = sub[(sub["method"] == m) & (sub["n"] == n_int)]
                    if hit.empty:
                        row[m] = "—"
                    else:
                        r = hit.iloc[0]
                        row[m] = _fmt(r["mean"], r.get("std", float("nan")))
                rows.append(row)

            table = pd.DataFrame(rows).set_index("N")
            table.columns = [m.replace("_", " ") for m in table.columns]
            print(f"\n{lang_name} ({lang_code})")
            print(table.to_string())

    print("=" * 70)
    print("CROSS-TRACK COMPARISON  —  mean weighted test F1")
    print("(best-per-seed averaged over seeds; ±std shown where applicable)")
    print("=" * 70)
    print("\n--- Best per track ---")
    pivot(summary, methods_main)

    print("\n\n--- Per model detail ---")
    pivot(summary, methods_detail)

    # Summary: at which scale does neural first beat n-gram?
    print("\n" + "=" * 70)
    print("CROSSOVER ANALYSIS  —  first N where finetune_best > ngram_best")
    print("=" * 70)
    for lang_code, lang_name in LANG_NAMES.items():
        ng = summary[(summary["lang"] == lang_code) & (summary["method"] == "ngram_best")].set_index("n")["mean"]
        ft_b = summary[(summary["lang"] == lang_code) & (summary["method"] == "finetune_best")].set_index("n")["mean"]
        crossover = "never"
        for n_val in SCALE_ORDER:
            n_int = -1 if n_val == "full" else int(n_val)
            if n_int in ng.index and n_int in ft_b.index:
                if ft_b[n_int] > ng[n_int]:
                    crossover = str(n_val)
                    break
        print(f"  {lang_name:<12}: {crossover}")


if __name__ == "__main__":
    main()
