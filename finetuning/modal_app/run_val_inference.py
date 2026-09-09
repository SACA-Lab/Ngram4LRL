"""
Validation-set inference for the saved AfroXLMR full-scale checkpoints.

Loads each saved_models/afroxlmr_{lang}_full_{seed} checkpoint from the
Modal volume and runs inference on the *validation* split (predictions/*.parquet
from training only ever covered the test split), writing per-instance
true_label, pred_label, and p_<class> -- same shape as
scripts/export_predictions_csv.py's test-set CSV, for the hybrid
interpolation.

AfroXLMR only, full scale only, all 5 seeds: the only combination with a
saved checkpoint (see modal_app/run_experiments.py).

Usage
─────
modal run modal_app/run_val_inference.py --dataset afrisenti
modal run modal_app/run_val_inference.py --dataset sib200

Writes results/<dataset>/val_predictions_probs.csv locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_config import load_dataset_config  # noqa: E402

DATASET_NAMES = ("afrisenti", "sib200")
SEEDS = [42, 123, 456, 789, 1024]
MAX_LENGTH = 256
BATCH_SIZE = 64


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


app = modal.App("finetuning-val-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.45,<5.0",
        "datasets>=2.18",
        "accelerate>=1.1.0,<5.0",
        "scikit-learn>=1.3",
        "numpy>=1.26",
        "pandas>=2.1",
        "pyarrow>=14",
        "pyyaml>=6.0",
    )
    .add_local_python_source("src")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/configs")
)

hf_cache = modal.Volume.from_name("finetuning-hf-cache", create_if_missing=True)
results_vol_afrisenti = modal.Volume.from_name("finetuning-afrisenti-results", create_if_missing=True)
results_vol_sib200 = modal.Volume.from_name("finetuning-sib200-results", create_if_missing=True)
_VOLUMES = {"afrisenti": results_vol_afrisenti, "sib200": results_vol_sib200}


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results_vol_afrisenti},
    gpu="T4",
    memory=8192,
    timeout=1800,
    max_containers=10,
)
def run_val_inference_afrisenti(lang: str) -> list[dict]:
    return _run_val_inference("afrisenti", lang)


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results_vol_sib200},
    gpu="T4",
    memory=8192,
    timeout=1800,
    max_containers=10,
)
def run_val_inference_sib200(lang: str) -> list[dict]:
    return _run_val_inference("sib200", lang)


_REMOTE_FNS = {"afrisenti": run_val_inference_afrisenti, "sib200": run_val_inference_sib200}


def _run_val_inference(dataset: str, lang: str) -> list[dict]:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.data import load_dataset_split

    configs_dir = Path("/root/configs")
    base_cfg = load_yaml(configs_dir / "base.yaml")
    spec = load_dataset_config(dataset, base_cfg, configs_dir)

    print(f"[{lang}] loading dataset ({dataset})")
    raw = load_dataset_split(spec, lang)
    val_ds = raw["validation"]
    texts = list(val_ds["text"])
    true_labels = list(val_ds["label"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    for seed in SEEDS:
        run_id = f"afroxlmr_{lang}_full_{seed}"
        ckpt_dir = f"/results/saved_models/{run_id}"
        if not Path(ckpt_dir).exists():
            print(f"[{lang}] SKIP {run_id}: no checkpoint at {ckpt_dir}")
            continue

        print(f"[{lang}] loading checkpoint {run_id}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        model.to(device).eval()

        all_probs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
        preds = probs.argmax(axis=1)

        for idx in range(len(texts)):
            row = {
                "model": "afroxlmr",
                "lang": lang,
                "n": "full",
                "seed": seed,
                "id": idx,
                "true_label": true_labels[idx],
                "pred_label": int(preds[idx]),
            }
            for k, label_name in enumerate(spec.labels):
                row[f"p_{label_name}"] = float(probs[idx, k])
            rows.append(row)

        print(f"[{lang}] {run_id}: {len(texts)} val instances done")

    return rows


@app.local_entrypoint()
def main(dataset: str):
    import pandas as pd

    if dataset not in DATASET_NAMES:
        raise SystemExit(f"Unsupported dataset {dataset!r}. Choose from: {DATASET_NAMES}")

    base_cfg = load_yaml(REPO_ROOT / "configs" / "base.yaml")
    spec = load_dataset_config(dataset, base_cfg, REPO_ROOT / "configs")
    langs = list(spec.lang_to_subset)

    print(f"Running AfroXLMR validation-set inference for {dataset}: {langs}")
    remote_fn = _REMOTE_FNS[dataset]
    all_rows = []
    for rows in remote_fn.map(langs):
        all_rows.extend(rows)

    if not all_rows:
        print("No rows produced -- check that full-scale AfroXLMR checkpoints exist.")
        return

    df = pd.DataFrame(all_rows)
    out_dir = REPO_ROOT / "results" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "val_predictions_probs.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {len(df)} rows -> {out_path.relative_to(REPO_ROOT)}")
    print(f"langs: {sorted(df['lang'].unique())}")
    print(f"seeds: {sorted(df['seed'].unique())}")
