"""
Main Modal app. Runs all (lang × scale × seed × classifier) trials in parallel,
for any of the three registered datasets (see ngram.data.DATASETS):
masakhanews, afrisenti, sib200.

Typical usage
─────────────
# Dry run — print job list without executing
modal run modal_app/run_experiments.py --dry-run
modal run modal_app/run_experiments.py --dataset afrisenti --dry-run
modal run modal_app/run_experiments.py --dataset sib200 --dry-run

# Full experiment suite (resumes automatically from where it left off)
modal run modal_app/run_experiments.py
modal run modal_app/run_experiments.py --dataset afrisenti
modal run modal_app/run_experiments.py --dataset sib200

# Subset (useful for smoke-testing)
modal run modal_app/run_experiments.py --langs lug --scales 100,250 --classifiers naive_bayes
modal run modal_app/run_experiments.py --dataset afrisenti --langs swa --scales 100 --classifiers naive_bayes

Results
───────
Each trial writes a JSON file to a dataset-specific Modal volume:
  masakhanews → ngram-results          (unchanged, matches results already collected)
  afrisenti   → ngram-afrisenti-results
  sib200      → ngram-sib200-results
Rerunning after a disconnect is safe: already-completed trials are skipped.
After the run, results are saved locally:
  masakhanews → results/raw_results.csv          (unchanged path)
  afrisenti   → results/afrisenti/raw_results.csv
  sib200      → results/sib200/raw_results.csv
Run  `python scripts/aggregate.py --dataset <dataset>`  to produce summary statistics.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent.parent))

from ngram.data import DATASETS

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

# ── HF Hub auth (avoids rate-limiting on dataset downloads after preemption) ───
hf_secret = modal.Secret.from_name("huggingface-secret")

# ── Persistent volumes ─────────────────────────────────────────────────────────
# masakhanews keeps its original volume/mount name so already-completed (paid)
# trials aren't orphaned; afrisenti/sib200 get their own volumes.
hf_cache               = modal.Volume.from_name("ngram-hf-cache",            create_if_missing=True)
results_vol_masakhanews = modal.Volume.from_name("ngram-results",            create_if_missing=True)
results_vol_afrisenti   = modal.Volume.from_name("ngram-afrisenti-results",  create_if_missing=True)
results_vol_sib200      = modal.Volume.from_name("ngram-sib200-results",     create_if_missing=True)

# Dataset name -> (mount path inside the container, Volume object, local CSV path)
_DATASET_INFO: dict[str, tuple[str, modal.Volume, Path]] = {
    "masakhanews": ("/results",            results_vol_masakhanews, Path("results/raw_results.csv")),
    "afrisenti":   ("/results_afrisenti",  results_vol_afrisenti,   Path("results/afrisenti/raw_results.csv")),
    "sib200":      ("/results_sib200",     results_vol_sib200,      Path("results/sib200/raw_results.csv")),
}

# ── Experiment parameters ──────────────────────────────────────────────────────
SEEDS       = [42, 123, 456, 789, 1024]
CLASSIFIERS = ["naive_bayes", "svm", "xgboost"]

TFIDF_KWARGS = dict(sublinear_tf=True, max_features=50000, min_df=1)


# ── Remote function (one invocation = one trial) ───────────────────────────────
@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results":            results_vol_masakhanews,
        "/results_afrisenti":  results_vol_afrisenti,
        "/results_sib200":     results_vol_sib200,
    },
    cpu=2.0,
    memory=16384,
    timeout=24*3600,
    retries=5,
    max_containers=90,  # stay under Modal free-tier 100-container cap
)
def run_trial_remote(
    dataset: str,
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
        dataset=dataset,
        tfidf_kwargs=TFIDF_KWARGS,
    )
    result_dict = result.to_dict()

    mount_path, vol, _ = _DATASET_INFO[dataset]
    out_dir = Path(f"{mount_path}/{lang}/{classifier_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"scale_{scale}_seed_{seed}.json"
    fname.write_text(json.dumps(result_dict, indent=2))
    vol.commit()

    return result_dict


# ── Resumption helpers (run locally, talk to the volume) ──────────────────────
def _completed_jobs(dataset: str, langs: list[str]) -> set[tuple]:
    """Return the set of (lang, scale, seed, clf) tuples already in the volume."""
    _, vol, _ = _DATASET_INFO[dataset]
    completed: set[tuple] = set()
    for lang in langs:
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


def _collect_all_results(dataset: str, langs: list[str]) -> list[dict]:
    """Read every result JSON from the volume — used to build the final CSV."""
    _, vol, _ = _DATASET_INFO[dataset]
    records = []
    for lang in langs:
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
    dataset: str,
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
                jobs.append((dataset, lang, scale, seed, clf))
    return jobs


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    dataset: str = "masakhanews",
    langs: str = "",
    classifiers: str = ",".join(CLASSIFIERS),
    scales: str = "",
    dry_run: bool = False,
    collect_only: bool = False,
    status: bool = False,
):
    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset '{dataset}'. Choose from: {list(DATASETS)}")
    spec = DATASETS[dataset]

    lang_list  = langs.split(",") if langs else list(spec.languages)
    clf_list   = classifiers.split(",")
    scale_list = (
        ["full" if s.strip() == "full" else int(s.strip()) for s in scales.split(",")]
        if scales else list(spec.scales)
    )

    # ── Status / collect modes don't submit any jobs ───────────────────────────
    if status:
        done = _completed_jobs(dataset, lang_list)
        all_jobs = _build_jobs(dataset, lang_list, scale_list, clf_list, SEEDS)
        print(f"Done : {len(done)} / {len(all_jobs)} trials")
        return

    if collect_only:
        _save_csv(dataset, lang_list)
        return

    # ── Build job list and filter already-completed trials ─────────────────────
    all_jobs = _build_jobs(dataset, lang_list, scale_list, clf_list, SEEDS)
    done     = _completed_jobs(dataset, lang_list)
    pending  = [j for j in all_jobs if (j[1], j[2], j[3], j[4]) not in done]

    print(f"Dataset      : {dataset}")
    print(f"Total trials : {len(all_jobs)}")
    print(f"Already done : {len(done)}")
    print(f"To submit    : {len(pending)}")

    if dry_run:
        print("\nFirst 12 pending jobs (dry run — not submitting):")
        for j in pending[:12]:
            print(f"  dataset={j[0]:<11}  lang={j[1]:<4}  scale={str(j[2]):<5}  seed={j[3]}  clf={j[4]}")
        return

    if not pending:
        print("All trials complete.")
        _save_csv(dataset, lang_list)
        return

    list(run_trial_remote.starmap(pending, order_outputs=False))
    print(f"\n{len(pending)} trials finished.")
    _save_csv(dataset, lang_list)


def _save_csv(dataset: str, langs: list[str]) -> None:
    import pandas as pd

    all_results = _collect_all_results(dataset, langs)
    if not all_results:
        print("No results in volume yet.")
        return

    _, _, out_path = _DATASET_INFO[dataset]
    df = pd.DataFrame(all_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(all_results)} results → {out_path}")
    print("\nMean test weighted-F1 by (lang, scale, classifier):")
    pivot = (
        df.groupby(["lang", "scale", "classifier"])["test_f1"]
        .mean().round(4)
        .unstack("classifier")
    )
    print(pivot.to_string())
    print(f"\nNext step: python scripts/aggregate.py --dataset {dataset}")
