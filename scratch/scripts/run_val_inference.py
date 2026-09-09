"""
Validation-set inference for the saved from-scratch checkpoints.

Loads each saved_models/scratch_{lang}_full_{seed} checkpoint (local; a
state dict + tokenizer + architecture config, not a HF PreTrainedModel --
see src/train.py's save_model block) and runs inference on the
*validation* split (predictions/*.parquet from training only ever covered
the test split), writing per-instance true_label, pred_label, and
p_<class> -- same shape as scripts/export_predictions_csv.py's test-set
CSV, for the hybrid interpolation.

Full scale only, all 5 seeds: the only combination with a saved checkpoint
(see modal_app/run_experiments.py). Runs locally -- checkpoints are a few
MB each and already on disk, so no need for Modal/GPU here.

Usage:
    python -m scripts.run_val_inference --dataset afrisenti
    python -m scripts.run_val_inference --dataset sib200

Writes results/<dataset>/val_predictions_probs.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import DATASET_NAMES, load_dataset_config, load_dataset_split
from src.model import SmallTransformerClassifier
from src.tokenizer import load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 456, 789, 1024]
BATCH_SIZE = 64


def _load_checkpoint(ckpt_dir: Path):
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    tokenizer = load_tokenizer(ckpt_dir / "tokenizer.json")
    model = SmallTransformerClassifier(
        vocab_size=cfg["vocab_size"],
        num_labels=cfg["num_labels"],
        d_model=cfg["d_model"],
        nhead=cfg["n_heads"],
        num_layers=cfg["n_layers"],
        dim_feedforward=cfg["d_ff"],
        max_position_embeddings=cfg["max_position_embeddings"],
        dropout=cfg["dropout"],
        pad_token_id=tokenizer.pad_token_id,
    )
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, tokenizer, cfg["max_length"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=[d for d in DATASET_NAMES if d != "masakhanews"])
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))
    spec = load_dataset_config(args.dataset, base_cfg, REPO_ROOT / "configs")
    results_dir = REPO_ROOT / "results" / args.dataset
    saved_models_dir = results_dir / "saved_models"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for lang in spec.lang_to_subset:
        print(f"[{lang}] loading dataset ({args.dataset})")
        raw = load_dataset_split(spec, lang)
        val_ds = raw["validation"]
        texts = list(val_ds["text"])
        true_labels = list(val_ds["label"])

        for seed in SEEDS:
            run_id = f"scratch_{lang}_full_{seed}"
            ckpt_dir = saved_models_dir / run_id
            if not ckpt_dir.exists():
                print(f"[{lang}] SKIP {run_id}: no checkpoint at {ckpt_dir}")
                continue

            print(f"[{lang}] loading checkpoint {run_id}")
            model, tokenizer, max_length = _load_checkpoint(ckpt_dir)
            model.to(device)

            all_probs = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                enc = tokenizer(
                    batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt",
                )
                with torch.no_grad():
                    logits = model(
                        input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                    ).logits
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            probs = np.concatenate(all_probs, axis=0)
            preds = probs.argmax(axis=1)

            for idx in range(len(texts)):
                row = {
                    "model": "scratch",
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

    if not rows:
        raise SystemExit("No rows produced -- check that full-scale scratch checkpoints exist.")

    df = pd.DataFrame(rows)
    out_path = results_dir / "val_predictions_probs.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {len(df)} rows -> {out_path.relative_to(REPO_ROOT)}")
    print(f"langs: {sorted(df['lang'].unique())}")
    print(f"seeds: {sorted(df['seed'].unique())}")


if __name__ == "__main__":
    main()
