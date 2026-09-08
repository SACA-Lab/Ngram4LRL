"""Generate LaTeX tables from results/metrics.csv and results/predictions/
(or results/<dataset>/... for afrisenti and sib200).

Tables are written to <results_dir>/tables/ as standalone .tex files.

Usage:
    python -m scripts.tables
    python -m scripts.tables --dataset afrisenti
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("masakhanews", "afrisenti", "sib200")

N_ORDER_BY_DATASET = {
    "masakhanews": ["100", "250", "500", "750", "full"],
    "afrisenti": ["100", "250", "500", "750", "full"],
    "sib200": ["100", "250", "500", "full"],
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


def _save(tables_dir: Path, text: str, name: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"{name}.tex"
    path.write_text(text)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def _load_base_cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))


def table_main_results(metrics_csv: Path, tables_dir: Path, n_order: list, lang_order: list, lang_names: dict) -> None:
    """Mean +/- std test weighted F1 per (model, language, N)."""
    if not metrics_csv.exists():
        return
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return
    df["n"] = df["n"].astype(str)

    g = (
        df.groupby(["model", "lang", "n"])["test_f1_weighted"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g["std"] = g["std"].fillna(0.0)
    g["cell"] = g.apply(lambda r: f"${r['mean']:.3f} \\pm {r['std']:.3f}$", axis=1)

    parts = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Test weighted F1 by language and training-set size, "
        "mean $\\pm$ standard deviation across five seeds (one seed at the full size).}",
        "\\label{tab:main_results}",
        "\\small",
        "\\begin{tabular}{ll" + "c" * len(n_order) + "}",
        "\\toprule",
        "Model & Language & " + " & ".join(n_order) + " \\\\",
        "\\midrule",
    ]
    models = sorted(g["model"].unique())
    for mi, model in enumerate(models):
        for li, lang in enumerate(lang_order):
            row = g[(g["model"] == model) & (g["lang"] == lang)]
            cells = []
            for n in n_order:
                m = row[row["n"] == n]
                cells.append(m["cell"].iloc[0] if len(m) else "--")
            label = MODEL_NAMES.get(model, model) if li == 0 else ""
            parts.append(f"{label} & {lang_names[lang]} & " + " & ".join(cells) + " \\\\")
        if mi < len(models) - 1:
            parts.append("\\midrule")
    parts += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    _save(tables_dir, "\n".join(parts), "main_results")


def table_per_class_f1(
    predictions_dir: Path, tables_dir: Path, labels: list, lang_order: list, lang_names: dict, seed: int, n: str = "full",
) -> None:
    """Per-class F1 at one scale (default the full training set)."""
    rows = []
    for model in ["xlmr", "afroxlmr"]:
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
                rows.append(
                    {"model": model, "lang": lang, "class": label_name, "f1": float(f1)}
                )

    if not rows:
        return
    df = pd.DataFrame(rows)
    parts = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Per-class test F1 at the {n} training-set size, "
        f"first seed.}}",
        "\\label{tab:per_class_f1}",
        "\\small",
        "\\begin{tabular}{ll" + "c" * len(labels) + "}",
        "\\toprule",
        "Model & Language & " + " & ".join(labels) + " \\\\",
        "\\midrule",
    ]
    models = sorted(df["model"].unique())
    for mi, model in enumerate(models):
        for li, lang in enumerate(lang_order):
            sub = df[(df["model"] == model) & (df["lang"] == lang)]
            if sub.empty:
                continue
            sub = sub.set_index("class")["f1"]
            cells = []
            for c in labels:
                v = sub.get(c, np.nan)
                cells.append(f"{v:.2f}" if not np.isnan(v) else "--")
            label = MODEL_NAMES.get(model, model) if li == 0 else ""
            parts.append(f"{label} & {lang_names[lang]} & " + " & ".join(cells) + " \\\\")
        if mi < len(models) - 1:
            parts.append("\\midrule")
    parts += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    _save(tables_dir, "\n".join(parts), f"per_class_f1_{n}")


def table_dataset_stats(tables_dir: Path, dataset: str, lang_order: list, lang_names: dict) -> None:
    """Per-language split sizes for the configured label set."""
    from src.data import load_dataset_config, load_dataset_split

    base_cfg = _load_base_cfg()
    spec = load_dataset_config(dataset, base_cfg, REPO_ROOT / "configs")
    title = DATASET_TITLE[dataset]

    parts = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{title} split sizes for the configured label set.}}",
        "\\label{tab:dataset_stats}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Language & Train & Validation & Test \\\\",
        "\\midrule",
    ]
    for lang in lang_order:
        ds = load_dataset_split(spec, lang)
        parts.append(
            f"{lang_names[lang]} & {len(ds['train'])} & {len(ds['validation'])} & {len(ds['test'])} \\\\"
        )
    parts += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    _save(tables_dir, "\n".join(parts), "dataset_stats")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="masakhanews", choices=DATASET_NAMES)
    args = ap.parse_args()

    results_dir = REPO_ROOT / "results"
    if args.dataset != "masakhanews":
        results_dir = results_dir / args.dataset
    tables_dir = results_dir / "tables"
    metrics_csv = results_dir / "metrics.csv"
    predictions_dir = results_dir / "predictions"

    n_order = N_ORDER_BY_DATASET[args.dataset]
    lang_order = LANG_ORDER_BY_DATASET[args.dataset]
    lang_names = LANG_NAMES_BY_DATASET[args.dataset]
    base_cfg = _load_base_cfg()
    seed = base_cfg["seeds"][0]

    print("main results")
    table_main_results(metrics_csv, tables_dir, n_order, lang_order, lang_names)

    from src.data import load_dataset_config
    labels = load_dataset_config(args.dataset, base_cfg, REPO_ROOT / "configs").labels

    print("per-class F1")
    table_per_class_f1(predictions_dir, tables_dir, labels, lang_order, lang_names, seed, "full")
    table_per_class_f1(predictions_dir, tables_dir, labels, lang_order, lang_names, seed, n_order[-2])

    print("dataset statistics")
    try:
        table_dataset_stats(tables_dir, args.dataset, lang_order, lang_names)
    except Exception as exc:
        print(f"  failed: {exc}")


if __name__ == "__main__":
    main()
