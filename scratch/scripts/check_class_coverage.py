"""Print per-class counts on each split and on every (size, seed) subsample.

Usage:
    python -m scripts.check_class_coverage
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.data import LANG_TO_SUBSET, class_counts, load_masakhanews, stratified_subsample

REPO_ROOT = Path(__file__).resolve().parents[1]


def _summarise(name: str, ds, labels) -> None:
    counts = class_counts(ds, len(labels))
    empty = [labels[i] for i, c in enumerate(counts) if c == 0]
    tag = f"  empty_classes={empty}" if empty else ""
    print(f"  {name} ({len(ds)}): {dict(zip(labels, counts.tolist()))}{tag}")


def main() -> None:
    base_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))
    labels = base_cfg["labels"]
    seeds = base_cfg["seeds"]
    sizes = [s for s in base_cfg["data_scales"] if s is not None]
    text_field = base_cfg.get("text_field", "text")

    print(f"labels (k={len(labels)}): {labels}")

    for lang in base_cfg["languages"]:
        print(f"\n{lang} ({LANG_TO_SUBSET[lang]})")
        ds = load_masakhanews(lang, labels, text_field=text_field)
        _summarise("train", ds["train"], labels)
        _summarise("val  ", ds["validation"], labels)
        _summarise("test ", ds["test"], labels)

        full = ds["train"]
        for n in sizes:
            print(f"  subsample N={n}:")
            for seed in seeds:
                sub = stratified_subsample(full, n, seed)
                counts = class_counts(sub, len(labels))
                empty = [labels[i] for i, c in enumerate(counts) if c == 0]
                tag = f"  empty_classes={empty}" if empty else ""
                print(f"    seed={seed}: {counts.tolist()}{tag}")


if __name__ == "__main__":
    main()
