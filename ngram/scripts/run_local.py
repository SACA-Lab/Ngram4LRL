"""
Run a single trial locally — useful for verifying the pipeline before submitting to Modal.

Usage
─────
python scripts/run_local.py
python scripts/run_local.py --lang lug --scale 100 --seed 42 --classifier svm
python scripts/run_local.py --dataset afrisenti --lang swa --scale 100 --classifier naive_bayes
python scripts/run_local.py --dataset sib200 --lang lug --scale 100 --classifier naive_bayes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ngram.data import DATASETS
from ngram.train import run_trial

CLASSIFIERS = ["naive_bayes", "svm", "xgboost"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="masakhanews", choices=list(DATASETS))
    parser.add_argument("--lang",       default=None,
                        help="Defaults to the first language for --dataset")
    parser.add_argument("--scale",      default="100",
                        help="Training scale, e.g. 100 | 250 | 500 | 750 | full "
                             "(valid values depend on --dataset)")
    parser.add_argument("--seed",       default=42,  type=int)
    parser.add_argument("--classifier", default="naive_bayes", choices=CLASSIFIERS)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    lang = args.lang or spec.languages[0]
    if lang not in spec.languages:
        raise SystemExit(f"lang '{lang}' not valid for dataset '{args.dataset}'. "
                         f"Choose from: {spec.languages}")

    scale = "full" if args.scale == "full" else int(args.scale)
    valid_scales = {s if s == "full" else int(s) for s in spec.scales}
    if scale not in valid_scales:
        raise SystemExit(f"scale '{scale}' not valid for dataset '{args.dataset}'. "
                         f"Choose from: {spec.scales}")

    print(f"dataset={args.dataset}  lang={lang}  scale={scale}  "
          f"seed={args.seed}  clf={args.classifier}")
    t0 = time.perf_counter()

    result = run_trial(
        lang=lang,
        scale=scale,
        seed=args.seed,
        classifier_name=args.classifier,
        dataset=args.dataset,
    )

    elapsed = time.perf_counter() - t0
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
