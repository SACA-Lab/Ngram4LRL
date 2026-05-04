"""
Type-to-Token Ratio (TTR) ablation.

Computes TTR for each (language, training-set size, seed) combination using
the same stratified subsampling as the experiments, then analyses the
relationship between TTR and the n-gram vs fine-tuning crossover point.

Outputs
-------
analysis/results/ttr_raw.csv       one row per (lang, scale, seed)
analysis/results/ttr_summary.csv   mean ± std over seeds
analysis/results/ttr_table.tex     publication LaTeX table
analysis/figures/ttr_vs_scale.{pdf,png}
analysis/figures/ttr_vs_crossover.{pdf,png}
analysis/figures/ttr_vs_advantage.{pdf,png}

Run from the repo root:

    python analysis/ttr.py
"""

from __future__ import annotations

import pathlib
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Allow importing from the ngram package without installing it.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ngram"))

from ngram.data import load_masakhanews, subsample  # noqa: E402

RESULTS_DIR = ROOT / "analysis" / "results"
FIG_DIR = ROOT / "analysis" / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

LANGUAGES = {
    "lug": "Luganda",
    "run": "Rundi",
    "sna": "chiShona",
    "swa": "Kiswahili",
}
SCALES = [100, 250, 500, 750, "full"]
SEEDS = [42, 123, 456, 789, 1024]

# Crossover points from analysis/compare.py
# (first N where mean fine-tuning F1 > mean n-gram F1)
CROSSOVER = {"lug": 500, "run": 100, "sna": 250, "swa": 100}

COLORS = {
    "lug": "#2196F3",
    "run": "#F44336",
    "sna": "#4CAF50",
    "swa": "#FF9800",
}
MARKERS = {"lug": "o", "run": "s", "sna": "^", "swa": "D"}

# Actual full training-set sizes (for x-axis positioning of the "full" point)
FULL_N = {"lug": 771, "run": 1117, "sna": 1288, "swa": 1658}


# ── TTR computation ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation (Unicode-aware)."""
    return re.findall(r"\w+", text.lower())


def _ttr(texts: list[str]) -> dict[str, float]:
    tokens: list[str] = []
    for t in texts:
        tokens.extend(_tokenize(t))
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    return {
        "total_tokens": n_tokens,
        "unique_types": n_types,
        "ttr": n_types / n_tokens if n_tokens else 0.0,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def compute_ttr() -> pd.DataFrame:
    rows: list[dict] = []

    for lang, lang_name in LANGUAGES.items():
        print(f"  {lang_name}...", end=" ", flush=True)
        train, _val, _test = load_masakhanews(lang)

        for scale in SCALES:
            n = None if scale == "full" else int(scale)
            seed_list = [SEEDS[0]] if scale == "full" else SEEDS

            for seed in seed_list:
                subset = subsample(train, n, seed) if n is not None else train
                stats_dict = _ttr(subset.texts)
                rows.append({
                    "lang": lang,
                    "language": lang_name,
                    "scale": scale,
                    "seed": seed,
                    "n_examples": len(subset),
                    **stats_dict,
                })
        print("done")

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["lang", "language", "scale"])
        .agg(
            n_examples=("n_examples", "first"),
            mean_ttr=("ttr", "mean"),
            std_ttr=("ttr", "std"),
            mean_total_tokens=("total_tokens", "mean"),
            mean_unique_types=("unique_types", "mean"),
        )
        .reset_index()
    )
    return agg


# ── LaTeX table ────────────────────────────────────────────────────────────────

def make_latex_table(agg: pd.DataFrame) -> str:
    col_order = [100, 250, 500, 750, "full"]
    lang_order = ["lug", "run", "sna", "swa"]

    header_cols = " & ".join([f"$N={c}$" if c != "full" else "full" for c in col_order])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Type-to-Token Ratio (TTR) by language and training-set size."
        r" Values are mean over 5 seeds; single seed at \emph{full} scale.}",
        r"\label{tab:ttr}",
        r"\begin{tabular}{l" + "r" * len(col_order) + "}",
        r"\toprule",
        r"\textbf{Language} & " + header_cols + r" \\",
        r"\midrule",
    ]

    for lang in lang_order:
        lang_name = LANGUAGES[lang]
        row_vals = []
        for scale in col_order:
            hit = agg[(agg["lang"] == lang) & (agg["scale"] == scale)]
            if hit.empty:
                row_vals.append("—")
            else:
                r = hit.iloc[0]
                if pd.isna(r["std_ttr"]) or r["std_ttr"] == 0:
                    row_vals.append(f"{r['mean_ttr']:.4f}")
                else:
                    row_vals.append(f"{r['mean_ttr']:.4f} {{\\tiny $\\pm${r['std_ttr']:.4f}}}")
        lines.append(f"{lang_name} & " + " & ".join(row_vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── figures ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved analysis/figures/{stem}.{{pdf,png}}")


def fig_ttr_vs_scale(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for lang, lang_name in LANGUAGES.items():
        sub = agg[agg["lang"] == lang].copy()
        sub["x"] = sub["scale"].apply(
            lambda s: FULL_N[lang] if s == "full" else int(s)
        )
        sub = sub.sort_values("x")
        ax.plot(
            sub["x"], sub["mean_ttr"],
            marker=MARKERS[lang], color=COLORS[lang],
            label=lang_name, linewidth=1.8, markersize=7,
        )
        std = sub["std_ttr"].fillna(0)
        ax.fill_between(
            sub["x"],
            sub["mean_ttr"] - std,
            sub["mean_ttr"] + std,
            alpha=0.15, color=COLORS[lang],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training examples", fontsize=11)
    ax.set_ylabel("Type-to-Token Ratio (TTR)", fontsize=11)
    ax.set_title("Vocabulary Sparsity by Language and Training Scale", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    _save(fig, "ttr_vs_scale")


def fig_ttr_vs_crossover(agg: pd.DataFrame) -> None:
    """Scatter: TTR at full scale vs crossover N (4 language points)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    full_agg = agg[agg["scale"] == "full"].set_index("lang")

    xs, ys = [], []
    for lang, lang_name in LANGUAGES.items():
        ttr_val = full_agg.loc[lang, "mean_ttr"]
        crossover_n = CROSSOVER[lang]
        xs.append(ttr_val)
        ys.append(crossover_n)
        ax.scatter(ttr_val, crossover_n, color=COLORS[lang], s=130, zorder=5,
                   marker=MARKERS[lang], label=lang_name)
        ax.annotate(
            lang_name, (ttr_val, crossover_n),
            textcoords="offset points", xytext=(7, 4), fontsize=10,
        )

    # Pearson correlation (4 points — illustrative)
    r, p = stats.pearsonr(xs, ys)
    ax.set_xlabel("TTR at full training scale", fontsize=11)
    ax.set_ylabel("Crossover N\n(first N where fine-tuning > n-gram)", fontsize=10)
    ax.set_title(
        f"Morphological Complexity vs Crossover Point\n"
        f"(Pearson r = {r:+.2f}, p = {p:.2f}, n = 4)",
        fontsize=11,
    )
    ax.set_yscale("log")
    ax.set_yticks([100, 250, 500])
    ax.set_yticklabels(["100", "250", "500"])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "ttr_vs_crossover")


def fig_ttr_vs_advantage(agg: pd.DataFrame) -> None:
    """Scatter: TTR vs (n-gram F1 − fine-tuning F1) at each (lang, scale)."""
    # Load F1 data
    ngram_df = pd.read_csv(ROOT / "ngram" / "results" / "raw_results.csv")
    ft_df = pd.read_csv(ROOT / "finetuning" / "results" / "metrics.csv")

    ngram_df = ngram_df.rename(columns={"scale": "n", "test_f1": "test_f1_weighted"})
    ngram_df["n"] = pd.to_numeric(ngram_df["n"], errors="coerce").fillna(-1).astype(int)
    ngram_best = (
        ngram_df.groupby(["lang", "n", "seed"])["test_f1_weighted"].max().reset_index()
    )
    ngram_agg = (
        ngram_best.groupby(["lang", "n"])["test_f1_weighted"]
        .mean().reset_index()
        .rename(columns={"test_f1_weighted": "ngram_f1"})
    )

    ft_df["n"] = pd.to_numeric(ft_df["n"], errors="coerce").fillna(-1).astype(int)
    ft_agg = (
        ft_df.groupby(["lang", "n"])["test_f1_weighted"]
        .mean().reset_index()
        .rename(columns={"test_f1_weighted": "ft_f1"})
    )

    merged = ngram_agg.merge(ft_agg, on=["lang", "n"])
    merged["advantage"] = merged["ngram_f1"] - merged["ft_f1"]

    # Map to TTR
    agg2 = agg.copy()
    agg2["n_int"] = agg2["scale"].apply(lambda s: -1 if s == "full" else int(s))
    ttr_lookup = agg2[["lang", "n_int", "mean_ttr"]].rename(columns={"n_int": "n"})
    plot_df = merged.merge(ttr_lookup, on=["lang", "n"])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for lang, lang_name in LANGUAGES.items():
        sub = plot_df[plot_df["lang"] == lang]
        ax.scatter(
            sub["mean_ttr"], sub["advantage"],
            color=COLORS[lang], marker=MARKERS[lang],
            label=lang_name, s=70, zorder=5,
        )

    # Overall regression line
    all_x = plot_df["mean_ttr"].values
    all_y = plot_df["advantage"].values
    slope, intercept, r, p, _ = stats.linregress(all_x, all_y)
    x_line = np.linspace(all_x.min(), all_x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.2, alpha=0.7,
            label=f"OLS fit (r={r:+.2f}, p={p:.2f})")

    ax.axhline(0, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Type-to-Token Ratio (TTR)", fontsize=11)
    ax.set_ylabel("N-gram advantage\n(n-gram F1 − fine-tuning F1)", fontsize=10)
    ax.set_title(
        "Higher TTR → Greater N-gram Advantage?",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, "ttr_vs_advantage")

    # Print regression summary
    n_pts = len(all_x)
    print(f"\n  OLS: advantage = {slope:+.3f} × TTR + {intercept:+.3f}")
    print(f"  Pearson r = {r:+.3f}, p = {p:.3f}, n = {n_pts} (lang×scale pairs)")


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Computing TTR...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = compute_ttr()

    agg = aggregate(df)

    df.to_csv(RESULTS_DIR / "ttr_raw.csv", index=False)
    agg.to_csv(RESULTS_DIR / "ttr_summary.csv", index=False)
    print(f"\nSaved analysis/results/ttr_raw.csv ({len(df)} rows)")
    print(f"Saved analysis/results/ttr_summary.csv ({len(agg)} rows)")

    # LaTeX table
    latex = make_latex_table(agg)
    (RESULTS_DIR / "ttr_table.tex").write_text(latex)
    print("Saved analysis/results/ttr_table.tex")

    # Print summary
    scale_order = [100, 250, 500, 750, "full"]
    pivot = (
        agg.pivot(index="language", columns="scale", values="mean_ttr")
        .reindex(columns=[c for c in scale_order if c in agg["scale"].values])
    )
    print("\nMean TTR by language and training-set size:")
    print(pivot.round(4).to_string())

    print("\nCrossover N vs TTR at full scale:")
    full_rows = agg[agg["scale"] == "full"].set_index("lang")
    for lang, lang_name in LANGUAGES.items():
        ttr_val = full_rows.loc[lang, "mean_ttr"]
        print(f"  {lang_name:<12}: TTR = {ttr_val:.4f}, crossover N = {CROSSOVER[lang]}")

    print("\nGenerating figures...")
    fig_ttr_vs_scale(agg)
    fig_ttr_vs_crossover(agg)
    fig_ttr_vs_advantage(agg)
    print("\nDone.")


if __name__ == "__main__":
    main()
