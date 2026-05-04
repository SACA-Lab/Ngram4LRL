"""Generate train_colab.ipynb.

Run after editing this file to regenerate the notebook:

    python colab/build_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "train_colab.ipynb"

REPO_URL = "https://github.com/SACA-Lab/Transformer-Scratch.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md("""# From-scratch transformer experiments on MasakhaNEWS

Reproduces the from-scratch encoder track for Luganda, Rundi, chiShona, and
Kiswahili at controlled training-set sizes {100, 250, 500, 750, full} with
five seeds.

Before running:
1. Set the runtime to a GPU (Runtime > Change runtime type > T4 GPU).
2. Authenticate to GitHub (Section 2) so the private repository can be cloned.
3. Mount Google Drive (Section 4) so results persist across sessions.
"""),

    md("## 1. GPU check"),
    code("""!nvidia-smi"""),

    md("""## 2. GitHub authentication

Paste a personal access token with `repo` scope into the masked input."""),
    code(f"""import getpass, subprocess, os, pathlib

token = getpass.getpass(\"GitHub PAT (repo scope): \")
WORK_DIR = pathlib.Path(\"/content/scratch\")
REPO_URL = \"{REPO_URL}\"

if not WORK_DIR.exists():
    url = REPO_URL.replace(\"https://\", f\"https://x-access-token:{{token}}@\")
    subprocess.check_call([\"git\", \"clone\", url, str(WORK_DIR)])

os.chdir(WORK_DIR)
print(\"cwd:\", os.getcwd())
!ls -la"""),

    md("## 3. Install dependencies"),
    code("""!pip install -q -r requirements.txt"""),

    md("""## 4. Mount Google Drive

Symlinks ``results/`` to a Drive directory so tokenisers, logs, predictions,
and ``metrics.csv`` persist across sessions and the grid runner can resume."""),
    code("""from google.colab import drive
drive.mount(\"/content/drive\")

import pathlib, shutil
DRIVE_RESULTS = pathlib.Path(\"/content/drive/MyDrive/research-scratch-results\")
DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)

local_results = pathlib.Path(\"results\")
if local_results.exists() and not local_results.is_symlink():
    for p in local_results.rglob(\"*\"):
        rel = p.relative_to(local_results)
        target = DRIVE_RESULTS / rel
        if p.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    shutil.rmtree(local_results)
if not local_results.exists():
    local_results.symlink_to(DRIVE_RESULTS, target_is_directory=True)

print(\"results ->\", DRIVE_RESULTS)
!ls results/"""),

    md("## 5. Sanity check label coverage"),
    code("""!python -m scripts.check_class_coverage"""),

    md("## 6. Train per-language tokenisers"),
    code("""!python -m scripts.train_tokenizer
!ls results/tokenizers/"""),

    md("## 7. Single-run smoke test"),
    code("""!python -m src.train --lang lug --n 250 --seed 42"""),

    md("""## 8. Grid run

Slice the grid through environment variables. The runner is idempotent:
completed runs (those with a JSON file in ``results/logs/``) are skipped, so
the cell can be interrupted and re-executed safely."""),
    code("""import os
os.environ[\"LANGS\"] = \"lug run sna swa\"
os.environ[\"NS\"]    = \"100 250 500 750 full\"
os.environ[\"SEEDS\"] = \"42 123 456 789 1024\"

!bash scripts/run_grid.sh"""),

    md("## 9. Aggregate results"),
    code("""!python -m scripts.aggregate"""),

    md("""## 10. Figures and LaTeX tables

Generates artefacts under ``results/figures/`` and ``results/tables/``."""),
    code("""!python -m scripts.figures
!python -m scripts.tables"""),
]


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"wrote {OUT}")
