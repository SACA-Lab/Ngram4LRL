"""Train one SentencePiece unigram tokeniser per language using the full
training split. The resulting tokenisers are reused across all (N, seed)
training cells.

Tokenisers are dataset-scoped (results/[<dataset>/]tokenizers/{lang}.json) so
a language code shared across datasets (e.g. `swa` in both MasakhaNEWS and
AfriSenti) doesn't reuse a tokeniser trained on a different corpus.

Usage:
    python -m scripts.train_tokenizer
    python -m scripts.train_tokenizer --dataset afrisenti
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data import DATASET_NAMES, load_dataset_config, load_dataset_split
from src.tokenizer import train_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="masakhanews", choices=DATASET_NAMES)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))
    spec = load_dataset_config(args.dataset, base_cfg, REPO_ROOT / "configs")
    vocab_size = int(base_cfg["vocab_size"])

    output_root = REPO_ROOT / base_cfg["output_root"]
    if args.dataset != "masakhanews":
        output_root = output_root / args.dataset
    out_dir = output_root / "tokenizers"

    for lang in spec.lang_to_subset:
        out = out_dir / f"{lang}.json"
        if out.exists():
            print(f"  skip {lang} (tokeniser exists at {out})")
            continue
        print(f"  training tokeniser for {lang}")
        ds = load_dataset_split(spec, lang)
        train_tokenizer(ds["train"]["text"], vocab_size, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
