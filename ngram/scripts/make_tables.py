"""
Generate publication-ready tables from results/summary.csv.

Outputs (in results/tables/)
────────────────────────────
main_results.csv   — wide table: (lang, scale) × classifier, values = mean±std
main_results.tex   — LaTeX version of the above
feature_wins.csv   — which feature type dominated per (lang, scale, clf)
feature_wins.tex   — LaTeX version of feature_wins

Usage
─────
python scripts/make_tables.py
python scripts/make_tables.py --results-dir results --output-dir results/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCALE_ORDER  = ["100", "250", "500", "750", "full"]
LANG_ORDER   = ["Luganda", "Rundi", "chiShona", "Kiswahili",
                "Amharic", "Hausa", "Igbo", "Yoruba", "Oromo", "Nig. Pidgin"]
LANG_DISPLAY = {
    "lug": "Luganda", "run": "Rundi", "sna": "chiShona", "swa": "Kiswahili",
    "amh": "Amharic", "hau": "Hausa", "ibo": "Igbo", "yor": "Yoruba",
    "orm": "Oromo", "pcm": "Nig. Pidgin",
}
CLF_DISPLAY  = {"naive_bayes": "NB", "svm": "SVM", "xgboost": "XGB"}
FEAT_DISPLAY = {
    "word_unigram": "W-1",
    "word_bigram":  "W-2",
    "char_3to5":    "C-35",
}


def _load(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = results_dir / "summary.csv"
    feature_path = results_dir / "feature_selection.csv"
    if not summary_path.exists():
        raise SystemExit(f"Not found: {summary_path}\nRun aggregate.py first.")
    summary    = pd.read_csv(summary_path, dtype={"scale": str})
    feature_sel = pd.read_csv(feature_path, dtype={"scale": str}) if feature_path.exists() else None
    return summary, feature_sel


def make_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows = (Language, Scale), cols = classifier, values = mean±std."""
    summary = summary.copy()
    summary["Language"] = summary["lang"].map(LANG_DISPLAY)
    summary["Classifier"] = summary["classifier"].map(CLF_DISPLAY)
    summary["value"] = summary.apply(
        lambda r: f"{r.mean_f1:.3f}" + (f" $\\pm$ {r.std_f1:.3f}" if r.n_trials > 1 else ""),
        axis=1,
    )
    summary["Scale"] = pd.Categorical(summary["scale"], categories=SCALE_ORDER, ordered=True)
    summary["Language"] = pd.Categorical(summary["Language"], categories=LANG_ORDER, ordered=True)

    table = summary.pivot_table(
        index=["Language", "Scale"],
        columns="Classifier",
        values="value",
        aggfunc="first",
        sort=True,
        observed=True,
    )[["NB", "SVM", "XGB"]]   # enforce column order
    return table


def make_feature_table(feature_sel: pd.DataFrame) -> pd.DataFrame:
    """Which feature type won most often per (Language, Scale, Classifier)."""
    feature_sel = feature_sel.copy()
    feature_sel["Language"]   = feature_sel["lang"].map(LANG_DISPLAY)
    feature_sel["Classifier"] = feature_sel["classifier"].map(CLF_DISPLAY)
    feature_sel["Feature"]    = feature_sel["dominant_feature"].map(FEAT_DISPLAY)
    feature_sel["Scale"] = pd.Categorical(feature_sel["scale"], categories=SCALE_ORDER, ordered=True)
    feature_sel["Language"] = pd.Categorical(feature_sel["Language"], categories=LANG_ORDER, ordered=True)

    return feature_sel.pivot_table(
        index=["Language", "Scale"],
        columns="Classifier",
        values="Feature",
        aggfunc="first",
        sort=True,
        observed=True,
    )[["NB", "SVM", "XGB"]]


def _to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    return table.to_latex(
        caption=caption,
        label=label,
        position="t",
        column_format="ll" + "c" * len(table.columns),
        escape=False,   # allow \pm and other LaTeX in cell values
        bold_rows=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="results/tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, feature_sel = _load(results_dir)

    # ── Main results table ─────────────────────────────────────────────────────
    main_table = make_main_table(summary)
    main_table.to_csv(output_dir / "main_results.csv")
    (output_dir / "main_results.tex").write_text(
        _to_latex(
            main_table,
            caption="Weighted F1 (mean $\\pm$ std over 5 trials) on MasakhaNEWS test sets. "
                    "NB = Multinomial Naive Bayes, SVM = Linear SVM (L2), XGB = XGBoost.",
            label="tab:main_results",
        )
    )
    print("Main results table:")
    print(main_table.to_string())

    # ── Feature selection table ────────────────────────────────────────────────
    if feature_sel is not None:
        feat_table = make_feature_table(feature_sel)
        feat_table.to_csv(output_dir / "feature_wins.csv")
        (output_dir / "feature_wins.tex").write_text(
            _to_latex(
                feat_table,
                caption="Dominant TF-IDF feature type selected by validation-set tuning. "
                        "W-1 = word unigrams, W-2 = word bigrams, C-35 = char 3--5-grams.",
                label="tab:feature_wins",
            )
        )
        print("\nFeature selection table:")
        print(feat_table.to_string())

    print(f"\nTables saved to {output_dir}/")


if __name__ == "__main__":
    main()
