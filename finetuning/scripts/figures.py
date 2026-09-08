"""Generate result figures from results/metrics.csv, results/logs/, and
results/predictions/ (or results/<dataset>/... for afrisenti and sib200).

Each figure is written to <results_dir>/figures/ as both PDF (vector, paper)
and PNG (raster, slides).

Usage:
    python -m scripts.figures
    python -m scripts.figures --dataset afrisenti
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("masakhanews", "afrisenti", "sib200")

N_ORDER_BY_DATASET = {
    "masakhanews": ["100", "250", "500", "750", "full"],
    "afrisenti": ["100", "250", "500", "750", "full"],
    "sib200": ["100", "250", "500", "full"],
}
N_TO_X_BY_DATASET = {
    "masakhanews": {"100": 100, "250": 250, "500": 500, "750": 750, "full": 2000},
    "afrisenti": {"100": 100, "250": 250, "500": 500, "750": 750, "full": 2000},
    "sib200": {"100": 100, "250": 250, "500": 500, "full": 701},
}
LANG_NAMES_BY_DATASET = {
    "masakhanews": {
        "lug": "Luganda", "run": "Rundi", "sna": "chiShona", "swa": "Kiswahili",
        "amh": "Amharic", "ibo": "Igbo", "yor": "Yoruba", "orm": "Oromo",
        "pcm": "Nig. Pidgin", "hau": "Hausa",
    },
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
LANG_ORDER_BY_DATASET = {
    "masakhanews": ["lug", "run", "sna", "swa", "amh", "ibo", "yor", "orm", "pcm", "hau"],
    "afrisenti": ["amh", "hau", "ibo", "orm", "pcm", "swa", "yor"],
    "sib200": ["hau", "ibo", "lug", "gaz", "run", "sna", "swh", "yor", "amh"],
}
DATASET_TITLE = {
    "masakhanews": "MasakhaNEWS",
    "afrisenti": "AfriSenti",
    "sib200": "SIB-200",
}
MODEL_NAMES = {"xlmr": "XLM-R", "afroxlmr": "AfroXLMR"}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(figures_dir: Path, fig, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = figures_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def _load_metrics(metrics_csv: Path, n_to_x: dict) -> pd.DataFrame | None:
    if not metrics_csv.exists():
        return None
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return None
    df["n"] = df["n"].astype(str)
    df["n_x"] = df["n"].map(n_to_x)
    return df


def _load_base_cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))


def fig_f1_vs_n(df: pd.DataFrame, figures_dir: Path, lang_order: list, lang_names: dict, n_order: list, n_to_x: dict) -> None:
    """One panel per language; lines per model; error bars across seeds."""
    nrows, ncols = (2, 5) if len(lang_order) > 4 else (2, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    for ax, lang in zip(axes.flat, lang_order):
        sub = df[df["lang"] == lang]
        if sub.empty:
            ax.set_title(f"{lang_names[lang]} (no data)")
            continue
        for model in sorted(sub["model"].unique()):
            ms = (
                sub[sub["model"] == model]
                .groupby("n_x")["test_f1_weighted"]
                .agg(["mean", "std", "count"])
                .sort_index()
            )
            ax.errorbar(
                ms.index,
                ms["mean"],
                yerr=ms["std"].fillna(0.0),
                marker="o",
                capsize=3,
                label=MODEL_NAMES.get(model, model),
            )
        ax.set_title(lang_names[lang])
        ax.set_xlabel("Training-set size")
        ax.set_ylabel("Test weighted F1")
        ax.set_xscale("log")
        ax.set_xticks([n_to_x[n] for n in n_order])
        ax.set_xticklabels(n_order)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
    for ax in axes.flat[len(lang_order):]:
        ax.set_visible(False)
    fig.suptitle("Test weighted F1 vs training-set size", y=1.02)
    plt.tight_layout()
    _save(figures_dir, fig, "f1_vs_n")


def fig_training_time_vs_n(df: pd.DataFrame, figures_dir: Path, n_order: list, n_to_x: dict) -> None:
    """Mean wall-clock training time per scale, by model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    g = df.groupby(["model", "n_x"])["train_seconds"].mean().reset_index()
    for model in sorted(g["model"].unique()):
        sub = g[g["model"] == model].sort_values("n_x")
        ax.plot(
            sub["n_x"],
            sub["train_seconds"] / 60.0,
            marker="o",
            label=MODEL_NAMES.get(model, model),
        )
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("Mean training time (minutes)")
    ax.set_xscale("log")
    ax.set_xticks([n_to_x[n] for n in n_order])
    ax.set_xticklabels(n_order)
    ax.legend()
    plt.tight_layout()
    _save(figures_dir, fig, "training_time_vs_n")


def fig_class_distribution(figures_dir: Path, dataset: str, lang_order: list, lang_names: dict) -> None:
    """Per-language stacked bars of class counts in train, val, and test."""
    from src.data import class_counts, load_dataset_config, load_dataset_split

    base_cfg = _load_base_cfg()
    spec = load_dataset_config(dataset, base_cfg, REPO_ROOT / "configs")
    labels = spec.labels

    rows = []
    for lang in lang_order:
        ds = load_dataset_split(spec, lang)
        for split_name, split_ds in [
            ("train", ds["train"]),
            ("val", ds["validation"]),
            ("test", ds["test"]),
        ]:
            counts = class_counts(split_ds, len(labels))
            for label_name, count in zip(labels, counts.tolist()):
                rows.append(
                    {"lang": lang, "split": split_name, "class": label_name, "count": int(count)}
                )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    palette = sns.color_palette("colorblind", n_colors=len(labels))
    for ax, split in zip(axes, ["train", "val", "test"]):
        sub = df[df["split"] == split]
        pivot = sub.pivot(index="lang", columns="class", values="count").fillna(0)
        pivot = pivot.reindex(index=lang_order, columns=labels)
        pivot.index = [lang_names[l] for l in pivot.index]
        pivot.plot(kind="bar", stacked=True, ax=ax, color=palette, width=0.75)
        ax.set_title(split.capitalize())
        ax.set_xlabel("")
        ax.set_ylabel("Examples")
        ax.legend(
            title="Class", loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7, ncol=1
        )
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle(f"{DATASET_TITLE[dataset]} class distribution", y=1.02)
    plt.tight_layout()
    _save(figures_dir, fig, "class_distribution")


def fig_confusion_matrix(
    predictions_dir: Path, figures_dir: Path, labels: list, lang_names: dict, model: str, lang: str, seed: int, n: str = "full",
) -> None:
    """Confusion matrix for the first-seed prediction at the given (model, lang, n)."""
    pq = predictions_dir / f"{model}_{lang}_{n}_{seed}.parquet"
    if not pq.exists():
        return
    df = pd.read_parquet(pq)
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=range(len(labels)))
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        cbar=True,
        cbar_kws={"label": "Recall"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"{MODEL_NAMES.get(model, model)} - {lang_names.get(lang, lang)} (N={n})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    _save(figures_dir, fig, f"confusion_{model}_{lang}_{n}")


def fig_per_class_f1_heatmap(
    predictions_dir: Path, figures_dir: Path, labels: list, lang_order: list, lang_names: dict, model: str, seed: int, n: str = "full",
) -> None:
    """Per-class F1 heatmap across languages for one model at one scale."""
    rows = []
    for lang in lang_order:
        pq = predictions_dir / f"{model}_{lang}_{n}_{seed}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        f1s = f1_score(
            df["true_label"],
            df["pred_label"],
            labels=range(len(labels)),
            average=None,
            zero_division=0,
        )
        for label_name, f1 in zip(labels, f1s):
            rows.append({"lang": lang, "class": label_name, "f1": float(f1)})

    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="lang", columns="class", values="f1")
    pivot = pivot.reindex(index=lang_order, columns=labels)
    pivot.index = [lang_names[l] for l in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "F1"},
        ax=ax,
    )
    ax.set_title(f"{MODEL_NAMES.get(model, model)} per-class F1 (N={n})")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _save(figures_dir, fig, f"per_class_f1_{model}_{n}")


def fig_train_history(logs_dir: Path, figures_dir: Path, lang_names: dict, model: str, lang: str, n: str, seed: int) -> None:
    """Validation F1 and loss across epochs, first seed."""
    log_path = logs_dir / f"{model}_{lang}_{n}_{seed}.json"
    if not log_path.exists():
        return
    log = json.loads(log_path.read_text())
    history = log.get("training_history", [])
    if not history:
        return

    eval_records = [r for r in history if "eval_f1_weighted" in r]
    if not eval_records:
        return
    epochs = [r["epoch"] for r in eval_records]
    f1s = [r["eval_f1_weighted"] for r in eval_records]
    losses = [r.get("eval_loss") for r in eval_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(epochs, f1s, marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation weighted F1")
    ax1.set_ylim(0, 1)

    if all(l is not None for l in losses):
        ax2.plot(epochs, losses, marker="o", color=sns.color_palette("colorblind")[1])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation loss")

    fig.suptitle(f"{MODEL_NAMES.get(model, model)} / {lang_names[lang]} / N={n}, seed={seed}")
    plt.tight_layout()
    _save(figures_dir, fig, f"train_history_{model}_{lang}_{n}_{seed}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="masakhanews", choices=DATASET_NAMES)
    args = ap.parse_args()

    _setup_style()

    results_dir = REPO_ROOT / "results"
    if args.dataset != "masakhanews":
        results_dir = results_dir / args.dataset
    figures_dir = results_dir / "figures"
    logs_dir = results_dir / "logs"
    predictions_dir = results_dir / "predictions"
    metrics_csv = results_dir / "metrics.csv"

    n_order = N_ORDER_BY_DATASET[args.dataset]
    n_to_x = N_TO_X_BY_DATASET[args.dataset]
    lang_order = LANG_ORDER_BY_DATASET[args.dataset]
    lang_names = LANG_NAMES_BY_DATASET[args.dataset]
    base_cfg = _load_base_cfg()
    seed = base_cfg["seeds"][0]

    from src.data import load_dataset_config
    labels = load_dataset_config(args.dataset, base_cfg, REPO_ROOT / "configs").labels

    df = _load_metrics(metrics_csv, n_to_x)
    if df is not None:
        print("F1 vs N")
        fig_f1_vs_n(df, figures_dir, lang_order, lang_names, n_order, n_to_x)
        print("Training time vs N")
        fig_training_time_vs_n(df, figures_dir, n_order, n_to_x)
    else:
        print("metrics.csv not found - skipping aggregate figures")

    print("Class distribution")
    try:
        fig_class_distribution(figures_dir, args.dataset, lang_order, lang_names)
    except Exception as exc:
        print(f"  failed: {exc}")

    for model in ["xlmr", "afroxlmr"]:
        for n in (n_order[-1], n_order[-2]):
            print(f"Per-class F1 heatmap: {model} N={n}")
            try:
                fig_per_class_f1_heatmap(predictions_dir, figures_dir, labels, lang_order, lang_names, model, seed, n)
            except Exception as exc:
                print(f"  failed: {exc}")
        for lang in lang_order:
            for n in (n_order[-1], n_order[-2]):
                print(f"Confusion matrix: {model}/{lang}/{n}")
                try:
                    fig_confusion_matrix(predictions_dir, figures_dir, labels, lang_names, model, lang, seed, n)
                except Exception as exc:
                    print(f"  failed: {exc}")

    print("Training history samples")
    sample_runs = [(model, lang_order[0], n_order[1]) for model in ("xlmr", "afroxlmr")] + [
        (model, lang_order[-1], n_order[-1]) for model in ("xlmr", "afroxlmr")
    ]
    for model, lang, n in sample_runs:
        try:
            fig_train_history(logs_dir, figures_dir, lang_names, model, lang, n, seed)
        except Exception as exc:
            print(f"  {model}/{lang}/{n} failed: {exc}")


if __name__ == "__main__":
    main()
