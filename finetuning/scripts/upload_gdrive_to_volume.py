"""
Upload local ZIP archives to a Modal volume using the Modal CLI.

Works one ZIP at a time:
  1. List *.zip files in the source directory.
  2. For each ZIP: extract locally → upload each model subdir with 'modal volume put' → cleanup.
  3. Track completed ZIPs in a local progress file so re-runs skip already-done archives.

The ZIPs contain a top-level 'new_saved_models/' directory. That prefix is stripped when
uploading so model subdirectories land directly at the volume root.

Usage
─────
# Normal run (from repo root) — resumes from where it left off
modal run finetuning/scripts/upload_gdrive_to_volume.py

# Specify a different ZIP directory
modal run finetuning/scripts/upload_gdrive_to_volume.py --zip-dir /path/to/zips

# Reset progress and start over
modal run finetuning/scripts/upload_gdrive_to_volume.py --reset
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

app = modal.App("upload-zips-to-volume")

VOLUME_NAME   = "Afroxlm-finetune-results"
PROGRESS_FILE = ".upload_progress.json"  # kept locally alongside the ZIPs


# ── Progress helpers ──────────────────────────────────────────────────────────

def _load_progress(progress_path: Path) -> set[str]:
    if progress_path.exists():
        try:
            return set(json.loads(progress_path.read_text()))
        except Exception:
            pass
    return set()


def _save_progress(progress_path: Path, completed: set[str]) -> None:
    progress_path.write_text(json.dumps(sorted(completed), indent=2))


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    zip_dir: str = "finetuning",
    reset: bool = False,
) -> None:
    zip_root      = Path(zip_dir)
    progress_path = zip_root / PROGRESS_FILE

    if reset and progress_path.exists():
        progress_path.unlink()
        print("Progress manifest cleared.\n")

    completed = _load_progress(progress_path)
    zip_files = sorted(zip_root.glob("*.zip"))

    if not zip_files:
        print(f"No .zip files found in {zip_root.resolve()}")
        return

    print(f"Found {len(zip_files)} ZIP file(s):")
    for zf in zip_files:
        done = " ✓" if zf.name in completed else ""
        print(f"  {zf.name}{done}")
    print()

    total = len(zip_files)
    for idx, zf in enumerate(zip_files, 1):
        if zf.name in completed:
            print(f"[{idx}/{total}] SKIP  {zf.name}")
            continue

        print(f"[{idx}/{total}] Extracting {zf.name} …")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(tmp_dir)

                # Strip top-level 'new_saved_models/' wrapper if present so
                # model subdirectories land directly at the volume root.
                children = os.listdir(tmp_dir)
                if (
                    len(children) == 1
                    and os.path.isdir(os.path.join(tmp_dir, children[0]))
                ):
                    upload_root = os.path.join(tmp_dir, children[0])
                else:
                    upload_root = tmp_dir

                model_dirs = sorted(
                    d for d in os.listdir(upload_root)
                    if os.path.isdir(os.path.join(upload_root, d))
                )
                n_dirs = len(model_dirs)
                for i, model_name in enumerate(model_dirs, 1):
                    local_path = os.path.join(upload_root, model_name)
                    print(f"  [{i}/{n_dirs}] Uploading {model_name} …")
                    subprocess.run(
                        [
                            "modal", "volume", "put",
                            VOLUME_NAME,
                            local_path,
                            f"/{model_name}",
                        ],
                        check=True,
                    )

            print(f"  ✅ {zf.name} — {n_dirs} model dir(s) uploaded\n")
            completed.add(zf.name)
            _save_progress(progress_path, completed)

        except Exception as exc:
            print(f"  ❌ ERROR on {zf.name}: {exc}\n"
                  f"     Re-run to retry this archive.\n")

    remaining = total - len(completed)
    if remaining == 0:
        print(f"All {total} ZIP(s) uploaded to volume '{VOLUME_NAME}'.")
    else:
        print(
            f"{len(completed)}/{total} done. "
            f"Re-run to upload the remaining {remaining}."
        )
