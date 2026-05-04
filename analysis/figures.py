"""
Cross-track comparison figures.

Generates a single publication figure with one subplot per language, showing
mean weighted test F1 vs training-set size for all three experiment tracks.
Output is written to analysis/figures/ as both PDF and PNG.

Run from the repo root:

    python analysis/figures.py
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
OUT = pathlib.Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

LANG_NAMES = {"lug": "Luganda", "run": "Rundi", "sna": "chiShona", "swa": "Kiswahili"}
SCALE_ORDER = [100, 250, 500, 750]  # exclude "full" (not a fixed x-axis value)
FULL_N = {                           # actual training-set size at "full"
    "lug": 771, "run": 1117, "sna": 1288, "swa": 1658,
}

STYLE = {
    "ngram_best":    {"label": "N-gram (best clf)", "color": "#2196F3", "marker": "o", "ls": "-"},
    "afroxlmr":      {"label": "AfroXLMR",          "color": "#F44336", "marker": "s", "ls": "--"},
    "xlmr":          {"label": "XLM-R",             "color": "#FF9800", "marker": "^", "ls": "--"},
    "scratch":       {"label": "Scratch Transformer","color": "#9C27B0", "marker": "D", "ls": ":"},
}


# ── loaders (mirrors compare.py) ───────────────────────────────────────────────

def _load_all() -> pd.DataFrame:
    frames = []

    # n-gram
    df = pd.read_csv(ROOT / "ngram" / "results" / "raw_results.csv")
    df = df.rename(columns={"scale": "n", "test_f1": "test_f1_weighted"})
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)
    best = df.groupby(["lang", "n", "seed"])["test_f1_weighted"].max().reset_index()
    agg = best.groupby(["lang", "n"])["test_f1_weighted"].agg(mean="mean", std="std").reset_index()
    agg["method"] = "ngram_best"
    frames.append(agg)

    # finetuning — per model
    df = pd.read_csv(ROOT / "finetuning" / "results" / "metrics.csv")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        agg = sub.groupby(["lang", "n"])["test_f1_weighted"].agg(mean="mean", std="std").reset_index()
        agg["method"] = model_name
        frames.append(agg)

    # scratch
    df = pd.read_csv(ROOT / "scratch" / "results" / "metrics.csv")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)
    agg = df.groupby(["lang", "n"])["test_f1_weighted"].agg(mean="mean", std="std").reset_index()
    agg["method"] = "scratch"
    frames.append(agg)

    return pd.concat(frames, ignore_index=True)


# ── plotting ───────────────────────────────────────────────────────────────────

def _plot_lang(ax: plt.Axes, data: pd.DataFrame, lang: str, methods: list[str]) -> None:
    for method in methods:
        style = STYLE.get(method, {})
        sub = data[(data["lang"] == lang) & (data["method"] == method)].copy()
        if sub.empty:
            continue

        # Replace -1 (full) with actual training size for x-axis
        sub["x"] = sub["n"].apply(lambda n: FULL_N[lang] if n == -1 else n)
        sub = sub.sort_values("x")

        ax.plot(
            sub["x"], sub["mean"],
            label=style.get("label", method),
            color=style.get("color", None),
            marker=style.get("marker", "o"),
            linestyle=style.get("ls", "-"),
            linewidth=1.8,
            markersize=6,
        )
        if sub["std"].notna().any():
            ax.fill_between(
                sub["x"],
                sub["mean"] - sub["std"],
                sub["mean"] + sub["std"],
                alpha=0.12,
                color=style.get("color", None),
            )


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _load_all()

    methods = [m for m in ["ngram_best", "xlmr", "afroxlmr", "scratch"] if m in data["method"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, (lang, lang_name) in enumerate(LANG_NAMES.items()):
        ax = axes[i]
        _plot_lang(ax, data, lang, methods)

        full_n = FULL_N[lang]
        ax.axvline(full_n, color="grey", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(lang_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Weighted F1 (test)")
        ax.set_xscale("log")
        ax.set_xlim(80, full_n * 1.3)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(methods), fontsize=10,
        bbox_to_anchor=(0.5, -0.02), frameon=True,
    )

    fig.suptitle(
        "N-gram vs Neural: Weighted F1 by Training Scale",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"f1_comparison.{ext}", bbox_inches="tight", dpi=150)
        print(f"Saved analysis/figures/f1_comparison.{ext}")

    plt.close(fig)


if __name__ == "__main__":
    main()
