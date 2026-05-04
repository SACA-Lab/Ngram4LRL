"""Train one SentencePiece unigram tokeniser per language using the full
training split. The resulting tokenisers are reused across all (N, seed)
training cells.

Usage:
    python -m scripts.train_tokenizer
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.data import load_masakhanews
from src.tokenizer import train_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))
    labels = cfg["labels"]
    text_field = cfg.get("text_field", "text")
    vocab_size = int(cfg["vocab_size"])
    out_dir = REPO_ROOT / cfg["output_root"] / "tokenizers"

    for lang in cfg["languages"]:
        out = out_dir / f"{lang}.json"
        if out.exists():
            print(f"  skip {lang} (tokeniser exists at {out})")
            continue
        print(f"  training tokeniser for {lang}")
        ds = load_masakhanews(lang, labels, text_field=text_field)
        train_tokenizer(ds["train"]["text"], vocab_size, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
