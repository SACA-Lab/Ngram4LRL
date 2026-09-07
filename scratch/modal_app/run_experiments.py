"""
Main Modal app for the from-scratch transformer track. Trains per-language
tokenisers once (CPU), then runs all (lang x scale x seed) training trials on
GPU, for AfriSenti or SIB-200 (MasakhaNEWS already has a complete local
results/metrics.csv from prior Colab runs and doesn't need this).

Typical usage
─────────────
# Dry run — print job list without executing
modal run modal_app/run_experiments.py --dataset afrisenti --dry-run

# Full grid (trains missing tokenisers first, then resumes automatically)
modal run modal_app/run_experiments.py --dataset afrisenti
modal run modal_app/run_experiments.py --dataset sib200

# Subset (useful for smoke-testing)
modal run modal_app/run_experiments.py --dataset afrisenti --langs swa --scales 100

# Progress check / re-sync the local CSV without submitting new jobs
modal run modal_app/run_experiments.py --dataset afrisenti --status
modal run modal_app/run_experiments.py --dataset afrisenti --collect-only

Results
───────
Tokenisers and each trial's logs/{run_id}.json + predictions/{run_id}.parquet
are written to a dataset-specific Modal volume (scratch-afrisenti-results,
scratch-sib200-results). Rerunning after a disconnect is safe: already
completed trials (a matching logs/{run_id}.json in the volume) are skipped.
After the run, results/<dataset>/{metrics.csv,logs/,predictions/,tokenizers/}
are resynced locally from the volume.
Run  `python -m scripts.aggregate --dataset <dataset>`  next.
"""
from __future__ import annotations

import json
import sys
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

app = modal.App("scratch-experiments")

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.45,<5.0",
        "tokenizers>=0.20",
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
hf_cache               = modal.Volume.from_name("scratch-hf-cache", create_if_missing=True)
results_vol_afrisenti  = modal.Volume.from_name("scratch-afrisenti-results", create_if_missing=True)
results_vol_sib200     = modal.Volume.from_name("scratch-sib200-results",    create_if_missing=True)

# Dataset name -> (Volume object, local results dir)
_DATASET_INFO: dict[str, tuple[modal.Volume, Path]] = {
    "afrisenti": (results_vol_afrisenti, REPO_ROOT / "results" / "afrisenti"),
    "sib200":    (results_vol_sib200,    REPO_ROOT / "results" / "sib200"),
}

SEEDS = [42, 123, 456, 789, 1024]


# ── Tokeniser training (one invocation = one language, CPU only) ──────────────
@app.function(image=image, volumes={"/root/.cache/huggingface": hf_cache, "/results": results_vol_afrisenti}, cpu=2.0, memory=4096, timeout=1800)
def train_tokenizer_remote_afrisenti(lang: str) -> str:
    return _train_tokenizer_and_sync("afrisenti", lang)


@app.function(image=image, volumes={"/root/.cache/huggingface": hf_cache, "/results": results_vol_sib200}, cpu=2.0, memory=4096, timeout=1800)
def train_tokenizer_remote_sib200(lang: str) -> str:
    return _train_tokenizer_and_sync("sib200", lang)


_TOKENIZER_FNS = {"afrisenti": train_tokenizer_remote_afrisenti, "sib200": train_tokenizer_remote_sib200}


def _train_tokenizer_and_sync(dataset: str, lang: str) -> str:
    import shutil
    import tempfile

    from src.data import load_dataset_config, load_dataset_split
    from src.tokenizer import train_tokenizer

    configs_dir = Path("/root/configs")
    base_cfg = load_yaml(configs_dir / "base.yaml")
    spec = load_dataset_config(dataset, base_cfg, configs_dir)

    vol_path = Path(f"/results/tokenizers/{lang}.json")
    if vol_path.exists():
        return f"skip {lang} (already exists)"

    ds = load_dataset_split(spec, lang)
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / f"{lang}.json"
        train_tokenizer(ds["train"]["text"], int(base_cfg["vocab_size"]), local_path)
        vol_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, vol_path)

    _DATASET_INFO[dataset][0].commit()
    return f"trained {lang}"


# ── Remote training function (one invocation = one trial, GPU) ────────────────
@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results_vol_afrisenti},
    gpu="T4",
    memory=8192,
    timeout=2 * 3600,
    retries=2,
    max_containers=20,
)
def run_trial_remote_afrisenti(lang: str, n: str, seed: int) -> dict:
    return _run_and_sync("afrisenti", lang, n, seed)


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache, "/results": results_vol_sib200},
    gpu="T4",
    memory=8192,
    timeout=2 * 3600,
    retries=2,
    max_containers=20,
)
def run_trial_remote_sib200(lang: str, n: str, seed: int) -> dict:
    return _run_and_sync("sib200", lang, n, seed)


_REMOTE_FNS = {"afrisenti": run_trial_remote_afrisenti, "sib200": run_trial_remote_sib200}


def _run_and_sync(dataset: str, lang: str, n: str, seed: int) -> dict:
    """Runs inside the container: copy the pre-trained tokeniser in, train on
    local disk, then copy the per-trial log and prediction into the shared
    volume. Training writes to ephemeral local storage (not the network-
    backed volume) so concurrent containers never race on metrics.csv.
    Full-scale trials also save the trained model (a few MB -- this model is
    tiny -- so unlike finetuning's, these get synced locally by default)."""
    import shutil
    import tempfile

    from src.data import load_dataset_config
    from src.train import load_yaml, run_trial

    configs_dir = Path("/root/configs")
    base_cfg = load_yaml(configs_dir / "base.yaml")
    dataset_spec = load_dataset_config(dataset, base_cfg, configs_dir)
    n_val = None if n == "full" else int(n)
    save_model = n == "full"

    local_root = Path(tempfile.mkdtemp(prefix="scratch-trial-"))
    (local_root / "tokenizers").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(f"/results/tokenizers/{lang}.json"), local_root / "tokenizers" / f"{lang}.json")

    metrics_row = run_trial(
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

    _DATASET_INFO[dataset][0].commit()
    shutil.rmtree(local_root, ignore_errors=True)
    return metrics_row


# ── Resumption helpers (run locally, talk to the volume) ──────────────────────
def _existing_tokenizers(dataset: str) -> set[str]:
    vol, _ = _DATASET_INFO[dataset]
    try:
        entries = vol.listdir("tokenizers")
    except Exception:
        return set()
    return {Path(e.path).stem for e in entries if e.path.endswith(".json")}


def _completed_run_ids(dataset: str) -> set[str]:
    vol, _ = _DATASET_INFO[dataset]
    try:
        entries = vol.listdir("logs")
    except Exception:
        return set()
    return {Path(e.path).stem for e in entries if e.path.endswith(".json")}


def _collect_all_metrics(dataset: str) -> list[dict]:
    vol, _ = _DATASET_INFO[dataset]
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
            n_label = "full" if cfg["n"] is None else str(cfg["n"])
            rows.append(
                {
                    "run_id": f'{cfg["short_name"]}_{cfg["lang"]}_{n_label}_{cfg["seed"]}',
                    "model": cfg["short_name"],
                    "dataset": cfg["dataset"],
                    "lang": cfg["lang"],
                    "n": n_label,
                    "seed": cfg["seed"],
                    "val_loss": log.get("val_loss"),
                    "val_f1_weighted": log["val_f1_weighted"],
                    "test_f1_weighted": log["test_f1_weighted"],
                    "train_seconds": log["train_seconds"],
                }
            )
        except Exception:
            continue
    return rows


def _sync_dir_locally(dataset: str, subdir: str, suffix: str) -> int:
    vol, results_dir = _DATASET_INFO[dataset]
    local_dir = results_dir / subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        entries = vol.listdir(subdir)
    except Exception:
        return 0
    for entry in entries:
        if not entry.path.endswith(suffix):
            continue
        local_path = local_dir / Path(entry.path).name
        if local_path.exists():
            continue
        local_path.write_bytes(b"".join(vol.read_file(entry.path)))
        n += 1
    return n


def _sync_saved_models_locally(dataset: str) -> int:
    """Recursively sync saved_models/{run_id}/{model.pt,tokenizer.json,
    config.json} -- unlike finetuning's, this model is a few MB, so syncing
    the whole set locally is cheap."""
    vol, results_dir = _DATASET_INFO[dataset]
    local_root = results_dir / "saved_models"
    n = 0
    try:
        run_dirs = vol.listdir("saved_models")
    except Exception:
        return 0
    for entry in run_dirs:
        if entry.type != modal.volume.FileEntryType.DIRECTORY:
            continue
        run_id = Path(entry.path).name
        local_run_dir = local_root / run_id
        if local_run_dir.exists():
            continue
        local_run_dir.mkdir(parents=True, exist_ok=True)
        for f in vol.listdir(entry.path):
            if f.type == modal.volume.FileEntryType.DIRECTORY:
                continue
            (local_run_dir / Path(f.path).name).write_bytes(b"".join(vol.read_file(f.path)))
        n += 1
    return n


def _save_csv_and_sync(dataset: str) -> None:
    import pandas as pd

    rows = _collect_all_metrics(dataset)
    _, results_dir = _DATASET_INFO[dataset]
    results_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        df = pd.DataFrame(rows)
        out_path = results_dir / "metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(rows)} results -> {out_path}")

    n_pred = _sync_dir_locally(dataset, "predictions", ".parquet")
    n_tok = _sync_dir_locally(dataset, "tokenizers", ".json")
    n_models = _sync_saved_models_locally(dataset)
    print(f"Synced {n_pred} new prediction file(s), {n_tok} new tokenizer(s), {n_models} new saved model(s)")

    if rows:
        print("\nMean test weighted-F1 by (lang, n):")
        df["n"] = df["n"].astype(str)
        pivot = df.groupby(["lang", "n"])["test_f1_weighted"].mean().round(4).unstack("n")
        print(pivot.to_string())
    print(f"\nNext step: python -m scripts.aggregate --dataset {dataset}")


def _build_jobs(spec) -> list[tuple]:
    # Unlike the local run_grid.sh/masakhanews convention, all 5 seeds run at
    # full scale too (not just the first) -- so there are 5 full-scale
    # checkpoints per cell to save for ablation studies, at the cost of ~4x
    # more of the most expensive trials for afrisenti/sib200 specifically.
    jobs = []
    for lang in spec.lang_to_subset:
        for scale in spec.data_scales:
            n = "full" if scale is None else str(scale)
            for seed in SEEDS:
                jobs.append((lang, n, seed))
    return jobs


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    dataset: str,
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

    if langs:
        spec.lang_to_subset = {l: spec.lang_to_subset[l] for l in langs.split(",")}
    if scales:
        spec.data_scales = ["full" if s.strip() == "full" else int(s.strip()) for s in scales.split(",")]

    if status:
        done = _completed_run_ids(dataset)
        all_jobs = _build_jobs(spec)
        n_full = sum(1 for j in all_jobs if j[1] == "full")
        vol, _ = _DATASET_INFO[dataset]
        try:
            n_saved = len([e for e in vol.listdir("saved_models") if e.type == modal.volume.FileEntryType.DIRECTORY])
        except Exception:
            n_saved = 0
        print(f"Tokenisers   : {len(_existing_tokenizers(dataset))} / {len(spec.lang_to_subset)}")
        print(f"Done         : {len(done)} / {len(all_jobs)} trials")
        print(f"Saved models : {n_saved} / {n_full} full-scale trials")
        return

    if collect_only:
        _save_csv_and_sync(dataset)
        return

    all_jobs = _build_jobs(spec)
    done = _completed_run_ids(dataset)
    short_name = base_cfg["short_name"]
    pending = [j for j in all_jobs if f"{short_name}_{j[0]}_{j[1]}_{j[2]}" not in done]

    print(f"Dataset      : {dataset}")
    print(f"Total trials : {len(all_jobs)}")
    print(f"Already done : {len(done)}")
    print(f"To submit    : {len(pending)}")

    if dry_run:
        print("\nFirst 12 pending jobs (dry run - not submitting):")
        for j in pending[:12]:
            print(f"  lang={j[0]:<4} n={j[1]:<5} seed={j[2]}")
        return

    if not pending:
        print("All trials complete.")
        _save_csv_and_sync(dataset)
        return

    have_tok = _existing_tokenizers(dataset)
    missing_langs = sorted(set(spec.lang_to_subset) - have_tok)
    if missing_langs:
        print(f"Training {len(missing_langs)} missing tokeniser(s): {missing_langs}")
        tok_fn = _TOKENIZER_FNS[dataset]
        for msg in tok_fn.map(missing_langs):
            print(f"  {msg}")

    remote_fn = _REMOTE_FNS[dataset]
    list(remote_fn.starmap(pending, order_outputs=False))
    print(f"\n{len(pending)} trials finished.")
    _save_csv_and_sync(dataset)
