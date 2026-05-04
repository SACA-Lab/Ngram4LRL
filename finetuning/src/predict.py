"""Test-set inference and per-instance prediction export."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import Trainer

from .metrics import weighted_f1


def save_test_predictions(
    trainer: Trainer,
    test_ds: Dataset,
    tokenizer,
    max_length: int,
    out_path: Path,
    label_names: Sequence[str],
) -> float:
    """Run inference on ``test_ds`` and write a parquet with per-instance
    logits and softmax probabilities. Returns weighted F1."""
    def _tok(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length, padding=False
        )

    test_tok = test_ds.map(
        _tok,
        batched=True,
        remove_columns=[c for c in test_ds.column_names if c != "label"],
    )
    output = trainer.predict(test_tok)

    logits = output.predictions
    labels = output.label_ids
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    df = pd.DataFrame(
        {
            "id": np.arange(len(test_ds), dtype=np.int64),
            "text": list(test_ds["text"]),
            "true_label": labels.astype(np.int64),
            "pred_label": preds.astype(np.int64),
        }
    )
    for i, name in enumerate(label_names):
        df[f"logit_{name}"] = logits[:, i].astype(np.float32)
        df[f"p_{name}"] = probs[:, i].astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return weighted_f1(labels, preds)
