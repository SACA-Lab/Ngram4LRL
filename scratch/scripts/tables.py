"""Generate LaTeX tables from results/metrics.csv and results/predictions/.

Tables are written to ``results/tables/`` as standalone .tex files.

Usage:
    python -m scripts.tables
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
TABLES = RESULTS / "tables"
PREDICTIONS = RESULTS / "predictions"
METRICS_CSV = RESULTS / "metrics.csv"

LANG_NAMES = {"lug": "Luganda", "run": "Rundi", "sna": "chiShona", "swa": "Kiswahili"}
LANG_ORDER = ["lug", "run", "sna", "swa"]
N_ORDER = ["100", "250", "500", "750", "full"]


def _save(text: str, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.tex"
    path.write_text(text)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def _load_base_cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))


def table_main_results() -> None:
    if not METRICS_CSV.exists():
        return
    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        return
    df["n"] = df["n"].astype(str)

    g = (
        df.groupby(["lang", "n"])["test_f1_weighted"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g["std"] = g["std"].fillna(0.0)
    g["cell"] = g.apply(lambda r: f"${r['mean']:.3f} \\pm {r['std']:.3f}$", axis=1)

    parts = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Scratch transformer test weighted F1 by language and "
        "training-set size, mean $\\pm$ standard deviation across five seeds "
        "(one seed at the full size).}",
        "\\label{tab:scratch_main_results}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(N_ORDER) + "}",
        "\\toprule",
        "Language & " + " & ".join(N_ORDER) + " \\\\",
        "\\midrule",
    ]
    for lang in LANG_ORDER:
        row = g[g["lang"] == lang]
        cells = []
        for n in N_ORDER:
            m = row[row["n"] == n]
            cells.append(m["cell"].iloc[0] if len(m) else "--")
        parts.append(f"{LANG_NAMES[lang]} & " + " & ".join(cells) + " \\\\")
    parts += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    _save("\n".join(parts), "main_results")


def table_per_class_f1(n: str = "full") -> None:
    cfg = _load_base_cfg()
    labels = cfg["labels"]
    seed = cfg["seeds"][0]
    short = cfg["short_name"]

    rows = []
    for lang in LANG_ORDER:
        pq = PREDICTIONS / f"{short}_{lang}_{n}_{seed}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        f1s = f1_score(
            df["true_label"], df["pred_label"],
            labels=range(len(labels)), average=None, zero_division=0,
        )
        for label_name, f1 in zip(labels, f1s):
            rows.append({"lang": lang, "class": label_name, "f1": float(f1)})

    if not rows:
        return
    df = pd.DataFrame(rows)
    parts = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Scratch transformer per-class test F1 at the {n} "
        f"training-set size, first seed.}}",
        f"\\label{{tab:scratch_per_class_f1_{n}}}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(labels) + "}",
        "\\toprule",
        "Language & " + " & ".join(labels) + " \\\\",
        "\\midrule",
    ]
    for lang in LANG_ORDER:
        sub = df[df["lang"] == lang]
        if sub.empty:
            continue
        sub = sub.set_index("class")["f1"]
        cells = []
        for c in labels:
            v = sub.get(c, np.nan)
            cells.append(f"{v:.2f}" if not np.isnan(v) else "--")
        parts.append(f"{LANG_NAMES[lang]} & " + " & ".join(cells) + " \\\\")
    parts += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    _save("\n".join(parts), f"per_class_f1_{n}")


def table_dataset_stats() -> None:
    from src.data import load_masakhanews

    cfg = _load_base_cfg()
    labels = cfg["labels"]
    text_field = cfg.get("text_field", "text")

    parts = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{MasakhaNEWS split sizes for the configured label set.}",
        "\\label{tab:dataset_stats}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Language & Train & Validation & Test \\\\",
        "\\midrule",
    ]
    for lang in LANG_ORDER:
        ds = load_masakhanews(lang, labels, text_field=text_field)
        parts.append(
            f"{LANG_NAMES[lang]} & {len(ds['train'])} & "
            f"{len(ds['validation'])} & {len(ds['test'])} \\\\"
        )
    parts += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    _save("\n".join(parts), "dataset_stats")


def main() -> None:
    print("main results")
    table_main_results()
    print("per-class F1")
    table_per_class_f1("full")
    table_per_class_f1("750")
    print("dataset statistics")
    try:
        table_dataset_stats()
    except Exception as exc:
        print(f"  failed: {exc}")


if __name__ == "__main__":
    main()
