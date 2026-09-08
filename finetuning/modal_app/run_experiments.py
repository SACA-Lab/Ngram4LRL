"""
Main Modal app for the fine-tuning track. Runs all (model x lang x scale x
seed) trials on GPU, for AfriSenti or SIB-200 (MasakhaNEWS already has a
complete local results/metrics.csv from prior Colab runs and doesn't need
this).

Typical usage
─────────────
# Dry run — print job list without executing
modal run modal_app/run_experiments.py --dataset afrisenti --dry-run

# Full grid (resumes automatically from where it left off)
modal run modal_app/run_experiments.py --dataset afrisenti
modal run modal_app/run_experiments.py --dataset sib200

# Subset (useful for smoke-testing)
modal run modal_app/run_experiments.py --dataset afrisenti --models xlm-r --langs swa --scales 100

# Progress check / re-sync the local CSV without submitting new jobs
modal run modal_app/run_experiments.py --dataset afrisenti --status
modal run modal_app/run_experiments.py --dataset afrisenti --collect-only

Results
───────
Each trial writes logs/{run_id}.json and predictions/{run_id}.parquet to a
dataset-specific Modal volume (finetuning-afrisenti-results,
finetuning-sib200-results). Rerunning after a disconnect is safe: already
completed trials (a matching logs/{run_id}.json in the volume) are skipped.
After the run, results/<dataset>/{metrics.csv,logs/,predictions/} are
resynced locally from the volume.

Full-scale trials (all 5 seeds) additionally save the trained model and
tokeniser to saved_models/{run_id}/ on the volume, for ablation studies.
These are NOT auto-synced locally (each is ~500MB-1.1GB; the full set would
be tens of GB) -- fetch one on demand with:
    modal volume get finetuning-<dataset>-results saved_models/<run_id> <local_dir>

Run  `python -m scripts.aggregate --dataset <dataset>`  next.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import modal
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import from the lightweight dataset_config module (not src.data or
# src.train) so this local entrypoint doesn't need torch/transformers/
# datasets installed on the machine driving `modal run` -- only pyyaml.
from src.dataset_config import load_dataset_config  # noqa: E402


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())

app = modal.App("finetuning-experiments")

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.45,<5.0",
        "datasets>=2.18",
        "accelerate>=1.1.0,<5.0",
        "scikit-learn>=1.3",
        "numpy>=1.26",
        "pandas>=2.1",
        "pyarrow>=14",
        "pyyaml>=6.0",
    )
    .add_local_python_source("src")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/configs")
)

# ── Persistent volumes ─────────────────────────────────────────────────────────
hf_cache             = modal.Volume.from_name("finetuning-hf-cache", create_if_missing=True)
results_vol_afrisenti = modal.Volume.from_name("finetuning-afrisenti-results", create_if_missing=True)
results_vol_sib200    = modal.Volume.from_name("finetuning-sib200-results",    create_if_missing=True)

# Dataset name -> (mount path inside the container, Volume object, local results dir)
_DATASET_INFO: dict[str, tuple[str, modal.Volume, Path]] = {
    "afrisenti": ("/results", results_vol_afrisenti, REPO_ROOT / "results" / "afrisenti"),
    "sib200":    ("/results", results_vol_sib200,    REPO_ROOT / "results" / "sib200"),
}

SEEDS = [42, 123, 456, 789, 1024]


# ── Remote function (one invocation = one trial) ───────────────────────────────
@app.function(
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol_afrisenti,
    },
    gpu="T4",
    memory=16384,
    timeout=3 * 3600,
    retries=2,
    max_containers=20,
)
def run_trial_remote_afrisenti(model: str, lang: str, n: str, seed: int) -> dict:
    return _run_and_sync("afrisenti", model, lang, n, seed)


@app.function(
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol_sib200,
    },
    gpu="T4",
    memory=16384,
    timeout=3 * 3600,
    retries=2,
    max_containers=20,
)
def run_trial_remote_sib200(model: str, lang: str, n: str, seed: int) -> dict:
    return _run_and_sync("sib200", model, lang, n, seed)


_REMOTE_FNS = {
    "afrisenti": run_trial_remote_afrisenti,
    "sib200": run_trial_remote_sib200,
}


def _run_and_sync(dataset: str, model: str, lang: str, n: str, seed: int) -> dict:
    """Runs inside the container: train on local disk, then copy the
    per-trial log and prediction into the shared volume. Training writes to
    ephemeral local storage (not the network-backed volume) so concurrent
    containers never race on the same metrics.csv. Full-scale trials also
    save the trained model + tokenizer, for ablation studies."""
    import shutil
    import tempfile

    from src.data import load_dataset_config
    from src.train import load_yaml, run_trial

    configs_dir = Path("/root/configs")
    base_cfg = load_yaml(configs_dir / "base.yaml")
    dataset_spec = load_dataset_config(dataset, base_cfg, configs_dir)
    n_val = None if n == "full" else int(n)
    save_model = n == "full"

    local_root = Path(tempfile.mkdtemp(prefix="ft-trial-"))
    metrics_row = run_trial(
        model_config=configs_dir / f"{model}.yaml",
        base_cfg=base_cfg,
        dataset_spec=dataset_spec,
        dataset=dataset,
        lang=lang,
        n=n_val,
        seed=seed,
        output_root=local_root,
        save_model=save_model,
    )

    run_id = metrics_row["run_id"]
    vol_root = Path("/results")
    for sub, ext in (("logs", "json"), ("predictions", "parquet")):
        src_path = local_root / sub / f"{run_id}.{ext}"
        dst_path = vol_root / sub / f"{run_id}.{ext}"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst_path)

    if save_model:
        src_dir = local_root / "saved_models" / run_id
        dst_dir = vol_root / "saved_models" / run_id
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    _DATASET_INFO[dataset][1].commit()
    shutil.rmtree(local_root, ignore_errors=True)
    return metrics_row


# ── Resumption helpers (run locally, talk to the volume) ──────────────────────
def _completed_run_ids(dataset: str) -> set[str]:
    _, vol, _ = _DATASET_INFO[dataset]
    try:
        entries = vol.listdir("logs")
    except Exception:
        return set()
    return {Path(e.path).stem for e in entries if e.path.endswith(".json")}


def _collect_all_metrics(dataset: str) -> list[dict]:
    """Rebuild metrics rows from every logs/*.json in the volume — the
    source of truth, since metrics.csv itself is never written to the volume
    (see ``_run_and_sync``'s docstring)."""
    _, vol, _ = _DATASET_INFO[dataset]
    rows = []
    try:
        entries = vol.listdir("logs")
    except Exception:
        return rows
    for entry in entries:
        if not entry.path.endswith(".json"):
            continue
        try:
            raw = b"".join(vol.read_file(entry.path))
            log = json.loads(raw)
            cfg = log["config"]
            rows.append(
                {
                    "run_id": f'{cfg["short_name"]}_{cfg["lang"]}_{cfg["n"] if cfg["n"] is not None else "full"}_{cfg["seed"]}',
                    "model": cfg["short_name"],
                    "dataset": cfg["dataset"],
                    "lang": cfg["lang"],
                    "n": "full" if cfg["n"] is None else str(cfg["n"]),
                    "seed": cfg["seed"],
                    "val_f1_weighted": log["val_f1_weighted"],
                    "test_f1_weighted": log["test_f1_weighted"],
                    "train_seconds": log["train_seconds"],
                }
            )
        except Exception:
            continue
    return rows


def _sync_predictions_locally(dataset: str) -> int:
    _, vol, results_dir = _DATASET_INFO[dataset]
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        entries = vol.listdir("predictions")
    except Exception:
        return 0
    for entry in entries:
        if not entry.path.endswith(".parquet"):
            continue
        local_path = pred_dir / Path(entry.path).name
        if local_path.exists():
            continue
        data = b"".join(vol.read_file(entry.path))
        local_path.write_bytes(data)
        n += 1
    return n


def _save_csv_and_sync(dataset: str) -> None:
    import pandas as pd

    rows = _collect_all_metrics(dataset)
    _, _, results_dir = _DATASET_INFO[dataset]
    results_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        df = pd.DataFrame(rows)
        out_path = results_dir / "metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(rows)} results -> {out_path}")

    n_new = _sync_predictions_locally(dataset)
    print(f"Synced {n_new} new prediction file(s) -> {results_dir / 'predictions'}")

    if rows:
        print("\nMean test weighted-F1 by (model, lang, n):")
        df["n"] = df["n"].astype(str)
        pivot = df.groupby(["model", "lang", "n"])["test_f1_weighted"].mean().round(4).unstack("n")
        print(pivot.to_string())
    print(f"\nNext step: python -m scripts.aggregate --dataset {dataset}")


def _build_jobs(spec, models: list[str]) -> list[tuple]:
    # Unlike the local run_grid.sh/masakhanews convention, all 5 seeds run at
    # full scale too (not just the first) -- so there are 5 full-scale
    # checkpoints per cell to save for ablation studies, at the cost of ~4x
    # more of the most expensive trials for afrisenti/sib200 specifically.
    jobs = []
    for model, lang in product(models, spec.lang_to_subset):
        for scale in spec.data_scales:
            n = "full" if scale is None else str(scale)
            for seed in SEEDS:
                jobs.append((model, lang, n, seed))
    return jobs


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    dataset: str,
    models: str = "xlm-r,afroxlmr",
    langs: str = "",
    scales: str = "",
    dry_run: bool = False,
    collect_only: bool = False,
    status: bool = False,
):
    if dataset not in _REMOTE_FNS:
        raise SystemExit(f"Unsupported dataset {dataset!r}. Choose from: {list(_REMOTE_FNS)}")

    base_cfg = load_yaml(REPO_ROOT / "configs" / "base.yaml")
    spec = load_dataset_config(dataset, base_cfg, REPO_ROOT / "configs")

    model_list = models.split(",")
    if langs:
        spec.lang_to_subset = {l: spec.lang_to_subset[l] for l in langs.split(",")}
    if scales:
        spec.data_scales = ["full" if s.strip() == "full" else int(s.strip()) for s in scales.split(",")]

    if status:
        done = _completed_run_ids(dataset)
        all_jobs = _build_jobs(spec, model_list)
        n_full = sum(1 for j in all_jobs if j[2] == "full")
        _, vol, _ = _DATASET_INFO[dataset]
        try:
            n_saved = len([e for e in vol.listdir("saved_models") if e.type == modal.volume.FileEntryType.DIRECTORY])
        except Exception:
            n_saved = 0
        print(f"Done         : {len(done)} / {len(all_jobs)} trials")
        print(f"Saved models : {n_saved} / {n_full} full-scale trials")
        return

    if collect_only:
        _save_csv_and_sync(dataset)
        return

    all_jobs = _build_jobs(spec, model_list)
    done = _completed_run_ids(dataset)

    def run_id_of(job: tuple) -> str:
        model, lang, n, seed = job
        short_name = load_yaml(REPO_ROOT / "configs" / f"{model}.yaml")["short_name"]
        return f"{short_name}_{lang}_{n}_{seed}"

    pending = [j for j in all_jobs if run_id_of(j) not in done]

    print(f"Dataset      : {dataset}")
    print(f"Total trials : {len(all_jobs)}")
    print(f"Already done : {len(done)}")
    print(f"To submit    : {len(pending)}")

    if dry_run:
        print("\nFirst 12 pending jobs (dry run - not submitting):")
        for j in pending[:12]:
            print(f"  model={j[0]:<9} lang={j[1]:<4} n={j[2]:<5} seed={j[3]}")
        return

    if not pending:
        print("All trials complete.")
        _save_csv_and_sync(dataset)
        return

    remote_fn = _REMOTE_FNS[dataset]
    list(remote_fn.starmap(pending, order_outputs=False))
    print(f"\n{len(pending)} trials finished.")
    _save_csv_and_sync(dataset)
