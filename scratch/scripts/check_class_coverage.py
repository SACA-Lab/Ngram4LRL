"""Print per-class counts on each split and on every (size, seed) subsample.

Usage:
    python -m scripts.check_class_coverage
    python -m scripts.check_class_coverage --dataset afrisenti
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data import DATASET_NAMES, class_counts, load_dataset_config, load_dataset_split, stratified_subsample

REPO_ROOT = Path(__file__).resolve().parents[1]


def _summarise(name: str, ds, labels) -> None:
    counts = class_counts(ds, len(labels))
    empty = [labels[i] for i, c in enumerate(counts) if c == 0]
    tag = f"  empty_classes={empty}" if empty else ""
    print(f"  {name} ({len(ds)}): {dict(zip(labels, counts.tolist()))}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="masakhanews", choices=DATASET_NAMES)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))
    spec = load_dataset_config(args.dataset, base_cfg, REPO_ROOT / "configs")
    seeds = base_cfg["seeds"]
    sizes = [s for s in spec.data_scales if s is not None]

    print(f"dataset={args.dataset}  labels (k={len(spec.labels)}): {spec.labels}")

    for lang, subset in spec.lang_to_subset.items():
        print(f"\n{lang} ({subset})")
        ds = load_dataset_split(spec, lang)
        _summarise("train", ds["train"], spec.labels)
        _summarise("val  ", ds["validation"], spec.labels)
        _summarise("test ", ds["test"], spec.labels)

        full = ds["train"]
        for n in sizes:
            print(f"  subsample N={n}:")
            for seed in seeds:
                sub = stratified_subsample(full, n, seed)
                counts = class_counts(sub, len(spec.labels))
                empty = [spec.labels[i] for i, c in enumerate(counts) if c == 0]
                tag = f"  empty_classes={empty}" if empty else ""
                print(f"    seed={seed}: {counts.tolist()}{tag}")


if __name__ == "__main__":
    main()
