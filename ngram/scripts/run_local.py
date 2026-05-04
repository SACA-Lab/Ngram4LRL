"""
Run a single trial locally — useful for verifying the pipeline before submitting to Modal.

Usage
─────
python scripts/run_local.py
python scripts/run_local.py --lang lug --scale 100 --seed 42 --classifier svm
python scripts/run_local.py --lang swh --scale full --classifier xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ngram.train import run_trial

LANGUAGES   = ["lug", "run", "sna", "swh"]
CLASSIFIERS = ["naive_bayes", "svm", "xgboost"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",       default="lug",         choices=LANGUAGES)
    parser.add_argument("--scale",      default="100",
                        help="Training scale: 100 | 250 | 500 | 750 | full")
    parser.add_argument("--seed",       default=42,  type=int)
    parser.add_argument("--classifier", default="naive_bayes", choices=CLASSIFIERS)
    args = parser.parse_args()

    scale = "full" if args.scale == "full" else int(args.scale)

    print(f"lang={args.lang}  scale={scale}  seed={args.seed}  clf={args.classifier}")
    t0 = time.perf_counter()

    result = run_trial(
        lang=args.lang,
        scale=scale,
        seed=args.seed,
        classifier_name=args.classifier,
    )

    elapsed = time.perf_counter() - t0
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
