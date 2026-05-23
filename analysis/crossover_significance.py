"""
Paired t-test: AfroXLMR vs. SVM (best N-gram) across seeds.

For each (language, data scale), tests whether AfroXLMR achieves a
statistically significant improvement over the best SVM (p < 0.05,
paired t-test across 5 seeds: 42, 123, 456, 789, 1024).

Usage:
    python analysis/crossover_significance.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

SEEDS = [42, 123, 456, 789, 1024]
LANGS = ["lug", "run", "sna", "swa", "amh", "ibo", "yor", "orm", "pcm", "hau"]
SCALES = [100, 250, 500, 750, "full"]

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

# ── Load data ─────────────────────────────────────────────────────────────────
root = Path(__file__).resolve().parents[1]

ngram_df = pd.read_csv(root / "ngram/results/raw_results.csv")
ft_df    = pd.read_csv(root / "finetuning/results/metrics.csv")

# Keep only SVM rows from ngram results
svm_df = ngram_df[ngram_df["classifier"] == "svm"].copy()

# Keep only AfroXLMR rows from finetuning results
afro_df = ft_df[ft_df["model"] == "afroxlmr"].copy()
# Normalise scale column: finetuning uses "n" with "full" as string
afro_df["scale"] = afro_df["n"].apply(lambda x: x if x == "full" else int(x))

# ── Run tests ─────────────────────────────────────────────────────────────────
rows = []

for lang in LANGS:
    for scale in SCALES:
        # --- SVM per-seed F1 ---
        svm_sub = svm_df[(svm_df["lang"] == lang) & (svm_df["scale"].astype(str) == str(scale))]
        svm_by_seed = {int(r["seed"]): r["test_f1"] for _, r in svm_sub.iterrows()}

        # --- AfroXLMR per-seed F1 ---
        afro_sub = afro_df[(afro_df["lang"] == lang) & (afro_df["scale"] == scale)]
        afro_by_seed = {int(r["seed"]): r["test_f1_weighted"] for _, r in afro_sub.iterrows()}

        # Only proceed if we have all 5 seeds for both models
        available_seeds = [s for s in SEEDS if s in svm_by_seed and s in afro_by_seed]
        if len(available_seeds) < 2:
            rows.append({
                "lang": lang, "scale": scale,
                "n_seeds": len(available_seeds),
                "svm_mean": float("nan"), "afro_mean": float("nan"),
                "delta_mean": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"),
                "significant": None, "direction": "—",
            })
            continue

        svm_vals  = np.array([svm_by_seed[s]  for s in available_seeds])
        afro_vals = np.array([afro_by_seed[s] for s in available_seeds])
        delta = afro_vals - svm_vals

        t_stat, p_value = ttest_rel(afro_vals, svm_vals)
        significant = p_value < 0.05
        direction = "afro > svm" if delta.mean() > 0 else "svm > afro"

        rows.append({
            "lang":       lang,
            "scale":      scale,
            "n_seeds":    len(available_seeds),
            "svm_mean":   svm_vals.mean(),
            "svm_std":    svm_vals.std(),
            "afro_mean":  afro_vals.mean(),
            "afro_std":   afro_vals.std(),
            "delta_mean": delta.mean(),
            "t_stat":     t_stat,
            "p_value":    p_value,
            "significant": significant,
            "direction":  direction,
        })

results = pd.DataFrame(rows)

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'─'*95}")
print(f"{'Lang':>8}  {'Scale':>5}  {'N':>2}  {'SVM':>7}  {'AfroXLM':>7}  {'Δ':>7}  {'t':>7}  {'p':>8}  {'Sig?':>5}  Direction")
print(f"{'─'*95}")

for _, r in results.iterrows():
    if pd.isna(r["svm_mean"]):
        print(f"{LANG_DISPLAY[r['lang']]:>8}  {str(r['scale']):>5}  {int(r['n_seeds']):>2}  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>8}  {'—':>5}  not enough data")
        continue

    sig_str = "✓" if r["significant"] else "✗"
    print(
        f"{LANG_DISPLAY[r['lang']]:>8}  {str(r['scale']):>5}  {int(r['n_seeds']):>2}"
        f"  {r['svm_mean']:.4f}  {r['afro_mean']:.4f}"
        f"  {r['delta_mean']:+.4f}  {r['t_stat']:+7.3f}  {r['p_value']:8.4f}"
        f"  {sig_str:>5}  {r['direction']}"
    )

print(f"{'─'*95}")

# ── Summary: crossover points ─────────────────────────────────────────────────
print("\n── Crossover summary (first scale where afro > svm AND p < 0.05) ──")
for lang in LANGS:
    sub = results[(results["lang"] == lang) & (results["direction"] == "afro > svm") & (results["significant"] == True)]
    if sub.empty:
        print(f"  {LANG_DISPLAY[lang]:>10}: no significant crossover found")
    else:
        first = sub.iloc[0]
        print(f"  {LANG_DISPLAY[lang]:>10}: N={first['scale']}  (Δ={first['delta_mean']:+.4f}, p={first['p_value']:.4f})")

# ── Save results ──────────────────────────────────────────────────────────────
out_path = root / "analysis/crossover_results.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(out_path, index=False, float_format="%.6f")
print(f"\nSaved → {out_path}")
