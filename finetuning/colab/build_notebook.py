"""Generate finetune_colab.ipynb.

Run after editing this file to regenerate the notebook:

    python colab/build_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "finetune_colab.ipynb"


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
    md("""# Fine-tuning experiments on MasakhaNEWS

Reproduces the fine-tuning track for XLM-R-base and AfroXLMR-base on Luganda,
Rundi, chiShona, and Kiswahili at controlled training-set sizes
{100, 250, 500, 750, full} with five seeds.

Before running:
1. Set the runtime to a GPU (Runtime > Change runtime type > T4 GPU).
2. Authenticate to GitHub (Section 2) so the private repository can be cloned.
3. Mount Google Drive (Section 4) so results persist across sessions.
"""),

    md("## 1. GPU check"),
    code("""!nvidia-smi"""),

    md("""## 2. GitHub authentication

Choose one of the two cells below. The first uses an interactive web flow;
the second accepts a personal access token (recommended on Colab where the
interactive flow is unreliable)."""),
    code("""# Option A: gh CLI interactive login.
!type -p gh >/dev/null || (apt-get update -qq && apt-get install -y -qq gh)
!gh auth login -h github.com -p https -w
!gh auth setup-git"""),

    code("""# Option B: paste a personal access token (repo scope).
# import getpass, os
# os.environ[\"GH_TOKEN\"] = getpass.getpass(\"GitHub PAT: \")
# !git config --global credential.helper store
# !echo \"https://x-access-token:$GH_TOKEN@github.com\" > ~/.git-credentials"""),

    md("## 3. Clone the repository"),
    code("""import os, subprocess, pathlib

REPO_URL = \"https://github.com/SACA-Lab/Finetuning.git\"
WORK_DIR = pathlib.Path(\"/content/finetuning\")

if not WORK_DIR.exists():
    subprocess.check_call([\"git\", \"clone\", REPO_URL, str(WORK_DIR)])

os.chdir(WORK_DIR)
print(\"cwd:\", os.getcwd())
!ls -la"""),

    md("""**Fallback** - upload the repository as a zip if cloning is not
available. From a local clone:
```
zip -r /tmp/finetuning.zip .
```
"""),
    code("""# from google.colab import files
# import zipfile, os, pathlib
# pathlib.Path(\"/content/finetuning\").mkdir(exist_ok=True)
# uploaded = files.upload()
# zip_name = next(iter(uploaded))
# with zipfile.ZipFile(zip_name) as zf:
#     zf.extractall(\"/content/finetuning\")
# os.chdir(\"/content/finetuning\")
# print(\"cwd:\", os.getcwd())"""),

    md("## 4. Install dependencies"),
    code("""!pip install -q -r requirements.txt"""),

    md("""## 5. Mount Google Drive

Symlinks ``results/`` to a Drive directory so logs, predictions, and
``metrics.csv`` persist across sessions and the grid runner can resume."""),
    code("""from google.colab import drive
drive.mount(\"/content/drive\")

import pathlib, shutil
DRIVE_RESULTS = pathlib.Path(\"/content/drive/MyDrive/research-finetune-results\")
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

    md("## 6. Sanity check label coverage"),
    code("""!python -m scripts.check_class_coverage"""),

    md("## 7. Single-run smoke test"),
    code("""!python -m src.train --model-config configs/xlm-r.yaml --lang lug --n 250 --seed 42"""),

    md("""## 8. Grid run

Slice the grid through environment variables. The runner is idempotent:
completed runs (those with a JSON file in ``results/logs/``) are skipped, so
the cell can be interrupted and re-executed safely."""),
    code("""import os
os.environ[\"MODELS\"] = \"xlm-r afroxlmr\"
os.environ[\"LANGS\"]  = \"lug\"
os.environ[\"NS\"]     = \"100 250 500 750 full\"
os.environ[\"SEEDS\"]  = \"42 123 456 789 1024\"

!bash scripts/run_grid.sh"""),

    md("## 9. Aggregate results"),
    code("""!python -m scripts.aggregate"""),

    md("""## 10. Generate figures and LaTeX tables

Both scripts read whatever has been produced so far and emit artefacts under
``results/figures/`` (PDF and PNG) and ``results/tables/`` (.tex).
"""),
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
