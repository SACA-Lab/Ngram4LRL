"""Single from-scratch training trial: one (language, N, seed) cell of the grid.

The model is initialised from random weights; only the per-language
SentencePiece tokeniser is reused across (N, seed) cells. Run
``scripts/train_tokenizer.py`` once before launching the grid.

Usage:
    python -m src.train --lang lug --n 250 --seed 42

Outputs (under ``output_root`` from configs/base.yaml):
    logs/{run_id}.json          run metadata, validation and test F1, history
    predictions/{run_id}.parquet  per-instance logits and softmax probabilities
    metrics.csv                 one row appended per completed run

``run_id = "{short_name}_{lang}_{n}_{seed}"`` (default short_name = ``scratch``).
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import LANG_TO_SUBSET, class_counts, load_masakhanews, stratified_subsample
from .metrics import weighted_f1
from .model import SmallTransformerClassifier, count_parameters
from .predict import save_test_predictions
from .tokenizer import load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunConfig:
    short_name: str
    lang: str
    n: Optional[int]
    seed: int
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_position_embeddings: int
    dropout: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_epochs: int
    early_stopping_patience: int
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    num_labels: int

    @property
    def n_label(self) -> str:
        return "full" if self.n is None else str(self.n)

    @property
    def run_id(self) -> str:
        return f"{self.short_name}_{self.lang}_{self.n_label}_{self.seed}"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_config(
    base_cfg: dict,
    lang: str,
    n: Optional[int],
    seed: int,
) -> RunConfig:
    bs_key = "full" if n is None else n
    train_bs = base_cfg["batch_size_schedule"][bs_key]
    return RunConfig(
        short_name=base_cfg["short_name"],
        lang=lang,
        n=n,
        seed=seed,
        vocab_size=int(base_cfg["vocab_size"]),
        d_model=int(base_cfg["d_model"]),
        n_heads=int(base_cfg["n_heads"]),
        n_layers=int(base_cfg["n_layers"]),
        d_ff=int(base_cfg["d_ff"]),
        max_position_embeddings=int(base_cfg["max_position_embeddings"]),
        dropout=float(base_cfg["dropout"]),
        learning_rate=float(base_cfg["learning_rate"]),
        weight_decay=float(base_cfg["weight_decay"]),
        warmup_ratio=float(base_cfg["warmup_ratio"]),
        max_epochs=int(base_cfg["max_epochs"]),
        early_stopping_patience=int(base_cfg["early_stopping_patience"]),
        max_length=int(base_cfg["max_length"]),
        train_batch_size=int(train_bs),
        eval_batch_size=int(base_cfg["eval_batch_size"]),
        num_labels=len(base_cfg["labels"]),
    )


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tokenize_fn(tokenizer, max_length: int):
    def _fn(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length, padding=False
        )
    return _fn


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_weighted": weighted_f1(labels, preds)}


def append_metrics_row(metrics_csv: Path, row: dict) -> None:
    write_header = not metrics_csv.exists()
    with open(metrics_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default=REPO_ROOT / "configs" / "base.yaml", type=Path)
    ap.add_argument("--lang", required=True, choices=sorted(LANG_TO_SUBSET))
    ap.add_argument("--n", default=None, help="train subsample size (integer or 'full')")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--output-root", default=None, type=Path)
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs")
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    n = None if args.n in (None, "full", "None") else int(args.n)
    cfg = build_run_config(base_cfg, args.lang, n, args.seed)
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    output_root = args.output_root or (REPO_ROOT / base_cfg["output_root"])

    seed_everything(cfg.seed, base_cfg.get("cudnn_deterministic", True))

    print(f"[{cfg.run_id}] loading dataset")
    raw = load_masakhanews(
        cfg.lang,
        base_cfg["labels"],
        text_field=base_cfg.get("text_field", "text"),
    )
    train_ds = stratified_subsample(raw["train"], cfg.n, cfg.seed)
    val_ds = raw["validation"]
    test_ds = raw["test"]

    counts = class_counts(train_ds, cfg.num_labels)
    print(
        f"[{cfg.run_id}] train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} class_counts={counts.tolist()}"
    )

    tok_path = output_root / "tokenizers" / f"{cfg.lang}.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokeniser not found at {tok_path}. "
            f"Run `python -m scripts.train_tokenizer` first."
        )
    print(f"[{cfg.run_id}] loading tokeniser: {tok_path}")
    tokenizer = load_tokenizer(tok_path)

    print(f"[{cfg.run_id}] building model")
    model = SmallTransformerClassifier(
        vocab_size=cfg.vocab_size,
        num_labels=cfg.num_labels,
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        num_layers=cfg.n_layers,
        dim_feedforward=cfg.d_ff,
        max_position_embeddings=cfg.max_position_embeddings,
        dropout=cfg.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"[{cfg.run_id}] parameters: {count_parameters(model):,}")

    tok = tokenize_fn(tokenizer, cfg.max_length)
    train_tok = train_ds.map(
        tok, batched=True, remove_columns=[c for c in train_ds.column_names if c != "label"]
    )
    val_tok = val_ds.map(
        tok, batched=True, remove_columns=[c for c in val_ds.column_names if c != "label"]
    )

    # Write checkpoints to local storage rather than the (possibly Drive-mounted)
    # output_root: per-epoch checkpoint writes through Drive trigger API quota
    # errors and slow training significantly.
    ckpt_dir = Path(tempfile.gettempdir()) / "scratch-ckpts" / cfg.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=cfg.max_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=20,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to="none",
        fp16=use_fp16,
    )

    callbacks = []
    if not args.no_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    print(f"[{cfg.run_id}] training")
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    print(f"[{cfg.run_id}] evaluating on validation split")
    val_metrics = trainer.evaluate(val_tok)

    print(f"[{cfg.run_id}] predicting on test split")
    pred_path = output_root / "predictions" / f"{cfg.run_id}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    test_f1 = save_test_predictions(
        trainer=trainer,
        test_ds=test_ds,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        out_path=pred_path,
        label_names=base_cfg["labels"],
    )

    log_path = output_root / "logs" / f"{cfg.run_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "n_parameters": count_parameters(model),
                "train_seconds": train_secs,
                "val_loss": float(val_metrics.get("eval_loss", float("nan"))),
                "val_f1_weighted": float(val_metrics.get("eval_f1_weighted", float("nan"))),
                "test_f1_weighted": test_f1,
                "train_class_counts": counts.tolist(),
                "train_size": len(train_ds),
                "training_history": list(trainer.state.log_history),
            },
            f,
            indent=2,
        )

    append_metrics_row(
        output_root / "metrics.csv",
        {
            "run_id": cfg.run_id,
            "model": cfg.short_name,
            "lang": cfg.lang,
            "n": cfg.n_label,
            "seed": cfg.seed,
            "val_loss": float(val_metrics.get("eval_loss", float("nan"))),
            "val_f1_weighted": float(val_metrics.get("eval_f1_weighted", float("nan"))),
            "test_f1_weighted": test_f1,
            "train_seconds": train_secs,
        },
    )

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"[{cfg.run_id}] done - test F1 = {test_f1:.4f}")


if __name__ == "__main__":
    main()
