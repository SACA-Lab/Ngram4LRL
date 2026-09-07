"""
Cross-track comparison figures.

Generates a single publication figure with one subplot per language, showing
mean weighted test F1 vs training-set size for all three experiment tracks.
Output is written to analysis/figures/[<dataset>/] as both PDF and PNG.

Run from the repo root:

    python analysis/figures.py
    python analysis/figures.py --dataset afrisenti
    python analysis/figures.py --dataset sib200
"""

from __future__ import annotations

import argparse
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
OUT_ROOT = pathlib.Path(__file__).parent / "figures"

DATASET_NAMES = ("masakhanews", "afrisenti", "sib200")

LANG_NAMES_BY_DATASET = {
    "masakhanews": {"lug": "Luganda", "run": "Rundi", "sna": "chiShona", "swa": "Kiswahili"},
    "afrisenti": {
        "amh": "Amharic", "hau": "Hausa", "ibo": "Igbo", "orm": "Oromo",
        "pcm": "Nig. Pidgin", "swa": "Kiswahili", "yor": "Yoruba",
    },
    "sib200": {
        "hau": "Hausa", "ibo": "Igbo", "lug": "Luganda", "gaz": "Oromo",
        "run": "Rundi", "sna": "chiShona", "swh": "Swahili", "yor": "Yoruba",
        "amh": "Amharic",
    },
}
# Actual training-set size at "full" scale, per (dataset, lang) -- used to
# place the "full" point correctly on the log-scaled x-axis.
FULL_N_BY_DATASET = {
    "masakhanews": {"lug": 771, "run": 1117, "sna": 1288, "swa": 1658},
    "afrisenti": {
        "amh": 5984, "hau": 14172, "ibo": 10192, "orm": 11763,
        "pcm": 5121, "swa": 1810, "yor": 8522,
    },
    # Every SIB-200 language has the same raw split (701 train rows); after
    # filtering to the 7 configured topic labels, 525 rows remain -- uniform
    # across languages since the label set is shared.
    "sib200": {lang: 525 for lang in LANG_NAMES_BY_DATASET["sib200"]},
}
DATASET_TITLE = {
    "masakhanews": "MasakhaNEWS",
    "afrisenti": "AfriSenti",
    "sib200": "SIB-200",
}

STYLE = {
    "ngram_best":    {"label": "N-gram (best clf)", "color": "#2196F3", "marker": "o", "ls": "-"},
    "afroxlmr":      {"label": "AfroXLMR",          "color": "#F44336", "marker": "s", "ls": "--"},
    "xlmr":          {"label": "XLM-R",             "color": "#FF9800", "marker": "^", "ls": "--"},
    "scratch":       {"label": "Scratch Transformer","color": "#9C27B0", "marker": "D", "ls": ":"},
}


# ── path resolution (mirrors compare.py) ────────────────────────────────────────

def _ngram_path(dataset: str) -> pathlib.Path:
    if dataset == "masakhanews":
        return ROOT / "ngram" / "results" / "raw_results.csv"
    return ROOT / "results" / dataset / "raw_results.csv"


def _finetuning_path(dataset: str) -> pathlib.Path:
    if dataset == "masakhanews":
        return ROOT / "finetuning" / "results" / "metrics.csv"
    return ROOT / "finetuning" / "results" / dataset / "metrics.csv"


def _scratch_path(dataset: str) -> pathlib.Path:
    if dataset == "masakhanews":
        return ROOT / "scratch" / "results" / "metrics.csv"
    return ROOT / "scratch" / "results" / dataset / "metrics.csv"


# ── loaders ──────────────────────────────────────────────────────────────────────

def _load_all(dataset: str) -> pd.DataFrame:
    frames = []

    # n-gram
    df = pd.read_csv(_ngram_path(dataset))
    df = df.rename(columns={"scale": "n", "test_f1": "test_f1_weighted"})
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)
    best = df.groupby(["lang", "n", "seed"])["test_f1_weighted"].max().reset_index()
    agg = best.groupby(["lang", "n"])["test_f1_weighted"].agg(mean="mean", std="std").reset_index()
    agg["method"] = "ngram_best"
    frames.append(agg)

    # finetuning — per model
    df = pd.read_csv(_finetuning_path(dataset))
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        agg = sub.groupby(["lang", "n"])["test_f1_weighted"].agg(mean="mean", std="std").reset_index()
        agg["method"] = model_name
        frames.append(agg)

    # scratch
    df = pd.read_csv(_scratch_path(dataset))
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(-1).astype(int)
    agg = df.groupby(["lang", "n"])["test_f1_weighted"].agg(mean="mean", std="std").reset_index()
    agg["method"] = "scratch"
    frames.append(agg)

    return pd.concat(frames, ignore_index=True)


# ── plotting ─────────────────────────────────────────────────────────────────────

def _plot_lang(ax: plt.Axes, data: pd.DataFrame, lang: str, methods: list[str], full_n: dict) -> None:
    for method in methods:
        style = STYLE.get(method, {})
        sub = data[(data["lang"] == lang) & (data["method"] == method)].copy()
        if sub.empty:
            continue

        # Replace -1 (full) with actual training size for x-axis
        sub["x"] = sub["n"].apply(lambda n: full_n[lang] if n == -1 else n)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="masakhanews", choices=DATASET_NAMES)
    args = ap.parse_args()
    dataset = args.dataset
    lang_names = LANG_NAMES_BY_DATASET[dataset]
    full_n = FULL_N_BY_DATASET[dataset]
    title = DATASET_TITLE[dataset]

    out_dir = OUT_ROOT if dataset == "masakhanews" else OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _load_all(dataset)

    methods = [m for m in ["ngram_best", "xlmr", "afroxlmr", "scratch"] if m in data["method"].unique()]

    n_langs = len(lang_names)
    ncols = 2 if n_langs <= 4 else 3
    nrows = -(-n_langs // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=False, sharey=False)
    axes = axes.flatten() if n_langs > 1 else [axes]

    for i, (lang, lang_name) in enumerate(lang_names.items()):
        ax = axes[i]
        _plot_lang(ax, data, lang, methods, full_n)

        fn = full_n[lang]
        ax.axvline(fn, color="grey", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(lang_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Weighted F1 (test)")
        ax.set_xscale("log")
        ax.set_xlim(80, fn * 1.3)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(labelsize=9)

    for ax in axes[n_langs:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(methods), fontsize=10,
        bbox_to_anchor=(0.5, -0.02), frameon=True,
    )

    fig.suptitle(
        f"N-gram vs Neural: Weighted F1 by Training Scale ({title})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"f1_comparison.{ext}", bbox_inches="tight", dpi=150)
        print(f"Saved {out_dir.relative_to(ROOT)}/f1_comparison.{ext}")

    plt.close(fig)


if __name__ == "__main__":
    main()
