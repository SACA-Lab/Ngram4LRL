"""
Plot hybrid sensitivity curves.

For each language: test weighted-F1 vs λ (the N-gram weight).
★ marks the λ chosen on the validation set.

Run after: modal run hybrid/run_hybrid.py
Requires:  hybrid/results/hybrid_sweep.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANG_DISPLAY = {
    "lug": "Luganda",
    "run": "Rundi",
    "sna": "chiShona",
    "swa": "Kiswahili",
    "amh": "Amharic",
    "ibo": "Igbo",
    "yor": "Yoruba",
    "orm": "Oromo",
    "pcm": "Nig. Pidgin",
    "hau": "Hausa",
}
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# ── Load results ─────────────────────────────────────────────────────────────
csv_path = Path("hybrid/results/hybrid_sweep.csv")
if not csv_path.exists():
    raise FileNotFoundError(
        f"{csv_path} not found.\n"
        "Run the experiment first:  modal run hybrid/run_hybrid.py"
    )

df = pd.read_csv(csv_path)
langs = sorted(df["lang"].unique())

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

for lang, color in zip(langs, COLORS):
    sub  = df[df["lang"] == lang].sort_values("lambda")
    best = sub[sub["is_best"]]

    ax.plot(
        sub["lambda"], sub["test_f1"],
        marker="o", markersize=4.5, linewidth=1.8,
        label=LANG_DISPLAY.get(lang, lang), color=color,
    )
    if not best.empty:
        ax.scatter(
            best["lambda"], best["test_f1"],
            s=180, marker="*", color=color, zorder=5,
            edgecolors="black", linewidths=0.6,
        )

lambdas = sorted(df["lambda"].unique())
ax.set_xticks(lambdas)
ax.set_xticklabels([f"{l:.1f}" for l in lambdas], fontsize=8)
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("λ  (N-gram SVM weight)", fontsize=12)
ax.set_ylabel("Weighted F1  (test)", fontsize=12)
ax.set_title(
    "Hybrid interpolation sensitivity\n"
    "Best SVM + AfroXLMR (★ = best λ on validation set)",
    fontsize=11,
)
ax.legend(
    framealpha=0.9, fontsize=9,
    loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.30),
)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.grid(axis="x", linestyle=":", alpha=0.3)

# Annotate λ=0 and λ=1 endpoints
ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.text(0.02, ax.get_ylim()[0], "Neural\nonly", fontsize=7, color="gray", va="bottom")
ax.text(0.98, ax.get_ylim()[0], "SVM\nonly",  fontsize=7, color="gray", va="bottom", ha="right")

fig.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
fig_dir = Path("hybrid/results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

for fmt in ("pdf", "png"):
    p = fig_dir / f"sensitivity_curve.{fmt}"
    kw = dict(bbox_inches="tight")
    if fmt == "png":
        kw["dpi"] = 150
    fig.savefig(p, **kw)
    print(f"Saved → {p}")

# ── Print numerical summary ───────────────────────────────────────────────────
print("\n── Test F1 at best λ (val-tuned) vs standalone models ───────────────")
print(f"{'lang':>6}  {'best_λ':>6}  {'hybrid':>8}  {'SVM':>8}  {'Neural':>8}  {'Δ_SVM':>8}  {'Δ_Neural':>9}")
print("─" * 70)
for lang in langs:
    sub      = df[df["lang"] == lang]
    best_row = sub[sub["is_best"]].iloc[0]
    svm_row  = sub[sub["lambda"] == 1.0].iloc[0]
    nn_row   = sub[sub["lambda"] == 0.0].iloc[0]
    print(
        f"{lang:>6}  {best_row['best_lambda']:>6.1f}"
        f"  {best_row['test_f1']:>8.4f}"
        f"  {svm_row['test_f1']:>8.4f}"
        f"  {nn_row['test_f1']:>8.4f}"
        f"  {best_row['test_f1'] - svm_row['test_f1']:>+8.4f}"
        f"  {best_row['test_f1'] - nn_row['test_f1']:>+9.4f}"
    )

plt.show()
