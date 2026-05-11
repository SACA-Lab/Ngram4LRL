"""
Hybrid interpolation: best SVM (N-gram) + AfroXLMR (full data)

    P_hybrid(y | x) = λ · P_ngram(y | x) + (1 − λ) · P_neural(y | x)

λ is tuned on the validation set via grid search over {0.0, 0.1, …, 1.0}.
One Modal GPU function is launched per language (runs in parallel).

Usage
─────
# All 4 languages
modal run hybrid/run_hybrid.py

# Single language (e.g. for smoke-testing)
modal run hybrid/run_hybrid.py --langs lug

Results
───────
Per-language JSON written to Modal volume `hybrid-results`.
hybrid/results/hybrid_sweep.csv saved locally after the run.
Then plot:  python hybrid/plot_sensitivity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("hybrid-interpolation")

# ── Container image ─────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.18",
        "scikit-learn>=1.4",
        "numpy>=1.26",
        "transformers>=4.40",
        "torch>=2.2",
        "accelerate>=0.27",
        "sentencepiece>=0.1.99",   # XLM-R tokenizer fallback
    )
)

# ── Volumes ──────────────────────────────────────────────────────────────────
checkpoints_vol = modal.Volume.from_name("Afroxlm-finetune-results")
hybrid_vol      = modal.Volume.from_name("hybrid-results", create_if_missing=True)

# ── Config ────────────────────────────────────────────────────────────────────
LANGUAGES = ["lug", "run", "sna", "swa"]
LABELS    = [
    "business", "entertainment", "health", "politics",
    "religion", "sports", "technology",
]
# 11 points: 0.0, 0.1, ..., 1.0
LAMBDAS = [round(i * 0.1, 1) for i in range(11)]

# Best SVM config at full scale (from ngram val-set grid search)
SVM_CONFIG = {
    "lug": {"feature_type": "char_3to5",   "C": 1.0},
    "run": {"feature_type": "char_3to5",   "C": 10.0},
    "sna": {"feature_type": "word_bigram", "C": 10.0},
    "swa": {"feature_type": "char_3to5",   "C": 1.0},
}

MAX_LENGTH = 256
BATCH_SIZE = 64


# ── Remote function (one per language) ───────────────────────────────────────
@app.function(
    image=image,
    volumes={
        "/checkpoints":    checkpoints_vol,
        "/hybrid-results": hybrid_vol,
    },
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def run_hybrid_lang(lang: str) -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import f1_score
    from sklearn.svm import LinearSVC
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"[{lang}] loading MasakhaNEWS")
    ds = load_dataset("masakhane/masakhanews", lang)

    # Use the dataset's ClassLabel str2int for label encoding — same as the
    # ngram training code (alphabetical: business=0 … technology=6).
    feat = ds["train"].features["category"]
    if hasattr(feat, "str2int"):
        label2id = feat.str2int
    else:
        sorted_names = sorted({ex["category"] for ex in ds["train"]})
        label2id = {n: i for i, n in enumerate(sorted_names)}

    def _get_labels(split: str) -> "np.ndarray":
        return np.array([label2id[ex["category"]] for ex in ds[split]])

    y_train = _get_labels("train")
    y_val   = _get_labels("validation")
    y_test  = _get_labels("test")

    train_texts = ds["train"]["text"]
    val_texts   = ds["validation"]["text"]
    test_texts  = ds["test"]["text"]

    print(f"[{lang}] train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    # ── N-gram SVM (calibrated) ────────────────────────────────────────────────
    cfg = SVM_CONFIG[lang]
    ft  = cfg["feature_type"]

    feature_kwargs = dict(sublinear_tf=True, max_features=50000, min_df=1)
    if ft == "char_3to5":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **feature_kwargs)
    elif ft == "word_bigram":
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), **feature_kwargs)
    else:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), **feature_kwargs)

    X_train = vec.fit_transform(train_texts)
    X_val   = vec.transform(val_texts)
    X_test  = vec.transform(test_texts)

    print(f"[{lang}] training calibrated SVM  (feature={ft}, C={cfg['C']})")
    base_svm = LinearSVC(C=cfg["C"], penalty="l2", max_iter=4000, random_state=42)
    # cv=5: calibration uses only training data — val set is never touched here.
    cal_svm  = CalibratedClassifierCV(base_svm, cv=5, method="sigmoid")
    cal_svm.fit(X_train, y_train)

    # Some languages are missing one or more categories in their training split.
    # CalibratedClassifierCV.predict_proba only returns columns for classes it
    # saw, so the output may be narrower than 7.  Expand to the full label
    # space by mapping cal_svm.classes_ back to the canonical 7-wide matrix.
    def _expand(probs: "np.ndarray") -> "np.ndarray":
        n_full = len(LABELS)
        if probs.shape[1] == n_full:
            return probs
        full = np.zeros((probs.shape[0], n_full), dtype=np.float64)
        for col, cls in enumerate(cal_svm.classes_):
            full[:, cls] = probs[:, col]
        return full

    ngram_val_probs  = _expand(cal_svm.predict_proba(X_val))   # (n_val,  7)
    ngram_test_probs = _expand(cal_svm.predict_proba(X_test))  # (n_test, 7)

    svm_val_f1  = float(f1_score(y_val,  ngram_val_probs.argmax(1),  average="weighted"))
    svm_test_f1 = float(f1_score(y_test, ngram_test_probs.argmax(1), average="weighted"))
    missing = sorted(set(range(len(LABELS))) - set(cal_svm.classes_.tolist()))
    if missing:
        print(f"[{lang}] SVM note: classes {missing} absent from training data — padded with 0")
    print(f"[{lang}] SVM       val={svm_val_f1:.4f}  test={svm_test_f1:.4f}")

    # ── AfroXLMR (fine-tuned, full data, seed 42) ─────────────────────────────
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = f"/checkpoints/afroxlmr_{lang}_full_42"

    print(f"[{lang}] loading AfroXLMR from {ckpt_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    except Exception:
        # Fall back to base tokenizer if the checkpoint directory is missing
        # special_tokens_map.json (not written by the fast tokenizer path).
        tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-base")

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device).eval()

    def _neural_probs(texts: list) -> "np.ndarray":
        all_probs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[i: i + BATCH_SIZE])
            enc   = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH,
                padding=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    print(f"[{lang}] AfroXLMR inference — validation set")
    neural_val_probs  = _neural_probs(val_texts)
    print(f"[{lang}] AfroXLMR inference — test set")
    neural_test_probs = _neural_probs(test_texts)

    nn_val_f1  = float(f1_score(y_val,  neural_val_probs.argmax(1),  average="weighted"))
    nn_test_f1 = float(f1_score(y_test, neural_test_probs.argmax(1), average="weighted"))
    print(f"[{lang}] AfroXLMR  val={nn_val_f1:.4f}  test={nn_test_f1:.4f}")

    # ── λ sweep ───────────────────────────────────────────────────────────────
    val_f1s:  list[float] = []
    test_f1s: list[float] = []

    for lam in LAMBDAS:
        h_val  = lam * ngram_val_probs  + (1.0 - lam) * neural_val_probs
        h_test = lam * ngram_test_probs + (1.0 - lam) * neural_test_probs
        val_f1s.append( float(f1_score(y_val,  h_val.argmax(1),  average="weighted", zero_division=0)))
        test_f1s.append(float(f1_score(y_test, h_test.argmax(1), average="weighted", zero_division=0)))

    best_idx = int(np.argmax(val_f1s))
    result = {
        "lang":                lang,
        "lambdas":             LAMBDAS,
        "val_f1":              val_f1s,
        "test_f1":             test_f1s,
        "best_lambda":         LAMBDAS[best_idx],
        "best_val_f1":         val_f1s[best_idx],
        "best_test_f1":        test_f1s[best_idx],
        "svm_only_val_f1":     svm_val_f1,
        "svm_only_test_f1":    svm_test_f1,
        "neural_only_val_f1":  nn_val_f1,
        "neural_only_test_f1": nn_test_f1,
    }
    print(
        f"[{lang}] best λ={result['best_lambda']:.1f}"
        f"  val={result['best_val_f1']:.4f}  test={result['best_test_f1']:.4f}"
    )

    out = Path(f"/hybrid-results/{lang}_hybrid.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    hybrid_vol.commit()

    return result


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(langs: str = ",".join(LANGUAGES)):
    import pandas as pd

    lang_list = langs.split(",")
    print(f"Launching hybrid interpolation for: {lang_list}")

    results = list(run_hybrid_lang.map(lang_list))

    # Flatten to tidy CSV
    rows = []
    for r in results:
        for lam, vf1, tf1 in zip(r["lambdas"], r["val_f1"], r["test_f1"]):
            rows.append({
                "lang":        r["lang"],
                "lambda":      lam,
                "val_f1":      round(vf1, 6),
                "test_f1":     round(tf1, 6),
                "best_lambda": r["best_lambda"],
                "is_best":     abs(lam - r["best_lambda"]) < 1e-9,
            })

    df  = pd.DataFrame(rows)
    out = Path("hybrid/results/hybrid_sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows → {out}")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"{'lang':>6}  {'best_λ':>6}  {'val':>7}  {'test':>7}  │  {'SVM':>7}  {'Neural':>7}")
    print("─" * 60)
    for r in sorted(results, key=lambda x: x["lang"]):
        print(
            f"{r['lang']:>6}  {r['best_lambda']:>6.1f}"
            f"  {r['best_val_f1']:>7.4f}  {r['best_test_f1']:>7.4f}"
            f"  │  {r['svm_only_test_f1']:>7.4f}  {r['neural_only_test_f1']:>7.4f}"
        )
    print("\nNext step: python hybrid/plot_sensitivity.py")
