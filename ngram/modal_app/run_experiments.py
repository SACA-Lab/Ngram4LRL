"""
Main Modal app. Runs all (lang × scale × seed × classifier) trials in parallel.

Typical usage
─────────────
# Dry run — print job list without executing
modal run modal_app/run_experiments.py --dry-run

# Full experiment suite (resumes automatically from where it left off)
modal run modal_app/run_experiments.py

# Subset (useful for smoke-testing)
modal run modal_app/run_experiments.py --langs lug --scales 100,250 --classifiers naive_bayes

Results
───────
Each trial writes a JSON file to the Modal volume `ngram-results`.
Rerunning after a disconnect is safe: already-completed trials are skipped.
After the run, results/raw_results.csv is saved locally.
Run  `python scripts/aggregate.py`  to produce summary statistics.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import modal

app = modal.App("ngram-experiments")

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.18",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "pandas>=2.0",
        "numpy>=1.26",
    )
    .add_local_python_source("ngram")
)

# ── Persistent volumes ─────────────────────────────────────────────────────────
hf_cache    = modal.Volume.from_name("ngram-hf-cache",  create_if_missing=True)
results_vol = modal.Volume.from_name("ngram-results",   create_if_missing=True)

# ── Experiment parameters ──────────────────────────────────────────────────────
LANGUAGES   = ["lug", "run", "sna", "swa"]
SCALES      = [100, 250, 500, 750, "full"]
SEEDS       = [42, 123, 456, 789, 1024]
CLASSIFIERS = ["naive_bayes", "svm", "xgboost"]

TFIDF_KWARGS = dict(sublinear_tf=True, max_features=50000, min_df=1)


# ── Remote function (one invocation = one trial) ───────────────────────────────
@app.function(
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
    cpu=2.0,
    memory=4096,
    timeout=24*3600,
    retries=2,
    max_containers=90,  # stay under Modal free-tier 100-container cap
)
def run_trial_remote(
    lang: str,
    scale: int | str,
    seed: int,
    classifier_name: str,
) -> dict:
    from ngram.train import run_trial

    result = run_trial(
        lang=lang,
        scale=scale,
        seed=seed,
        classifier_name=classifier_name,
        tfidf_kwargs=TFIDF_KWARGS,
    )
    result_dict = result.to_dict()

    out_dir = Path(f"/results/{lang}/{classifier_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"scale_{scale}_seed_{seed}.json"
    fname.write_text(json.dumps(result_dict, indent=2))
    results_vol.commit()

    return result_dict


# ── Resumption helpers (run locally, talk to the volume) ──────────────────────
def _completed_jobs(vol: modal.Volume) -> set[tuple]:
    """Return the set of (lang, scale, seed, clf) tuples already in the volume."""
    completed: set[tuple] = set()
    for lang in LANGUAGES:
        for clf in CLASSIFIERS:
            try:
                entries = vol.listdir(f"{lang}/{clf}")
            except Exception:
                continue  # directory doesn't exist yet
            for entry in entries:
                name = Path(entry.path).name
                if not name.endswith(".json"):
                    continue
                # filename pattern: scale_{scale}_seed_{seed}.json
                stem = name[:-5]
                try:
                    parts = stem.split("_")          # ['scale', val, 'seed', val]
                    scale_str = parts[1]
                    seed_val  = int(parts[3])
                    scale = "full" if scale_str == "full" else int(scale_str)
                    completed.add((lang, scale, seed_val, clf))
                except (IndexError, ValueError):
                    continue
    return completed


def _collect_all_results(vol: modal.Volume) -> list[dict]:
    """Read every result JSON from the volume — used to build the final CSV."""
    records = []
    for lang in LANGUAGES:
        for clf in CLASSIFIERS:
            try:
                entries = vol.listdir(f"{lang}/{clf}")
            except Exception:
                continue
            for entry in entries:
                if not entry.path.endswith(".json"):
                    continue
                try:
                    raw = b"".join(vol.read_file(entry.path))
                    records.append(json.loads(raw))
                except Exception:
                    continue
    return records


# ── Job list builder ───────────────────────────────────────────────────────────
def _build_jobs(
    langs: list[str],
    scales: list,
    classifiers: list[str],
    seeds: list[int],
) -> list[tuple]:
    jobs = []
    for lang, clf in product(langs, classifiers):
        for scale in scales:
            trial_seeds = [seeds[0]] if scale == "full" else seeds
            for seed in trial_seeds:
                jobs.append((lang, scale, seed, clf))
    return jobs


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    langs: str = ",".join(LANGUAGES),
    classifiers: str = ",".join(CLASSIFIERS),
    scales: str = "100,250,500,750,full",
    dry_run: bool = False,
    collect_only: bool = False,
    status: bool = False,
):
    import pandas as pd

    # ── Status / collect modes don't submit any jobs ───────────────────────────
    if status:
        done = _completed_jobs(results_vol)
        all_jobs = _build_jobs(LANGUAGES, SCALES, CLASSIFIERS, SEEDS)
        print(f"Done : {len(done)} / {len(all_jobs)} trials")
        return

    if collect_only:
        _save_csv(results_vol)
        return

    # ── Build job list and filter already-completed trials ─────────────────────
    lang_list  = langs.split(",")
    clf_list   = classifiers.split(",")
    scale_list = [
        "full" if s.strip() == "full" else int(s.strip())
        for s in scales.split(",")
    ]

    all_jobs = _build_jobs(lang_list, scale_list, clf_list, SEEDS)
    done     = _completed_jobs(results_vol)
    pending  = [j for j in all_jobs if j not in done]

    print(f"Total trials : {len(all_jobs)}")
    print(f"Already done : {len(done)}")
    print(f"To submit    : {len(pending)}")

    if dry_run:
        print("\nFirst 12 pending jobs (dry run — not submitting):")
        for j in pending[:12]:
            print(f"  lang={j[0]:<4}  scale={str(j[1]):<5}  seed={j[2]}  clf={j[3]}")
        return

    if not pending:
        print("All trials complete.")
        _save_csv(results_vol)
        return

    list(run_trial_remote.starmap(pending, order_outputs=False))
    print(f"\n{len(pending)} trials finished.")
    _save_csv(results_vol)


def _save_csv(vol: modal.Volume) -> None:
    import pandas as pd

    all_results = _collect_all_results(vol)
    if not all_results:
        print("No results in volume yet.")
        return

    df = pd.DataFrame(all_results)
    out_path = Path("results/raw_results.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(all_results)} results → {out_path}")
    print("\nMean test weighted-F1 by (lang, scale, classifier):")
    pivot = (
        df.groupby(["lang", "scale", "classifier"])["test_f1"]
        .mean().round(4)
        .unstack("classifier")
    )
    print(pivot.to_string())
    print("\nNext step: python scripts/aggregate.py")
