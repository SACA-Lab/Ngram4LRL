"""Per-dataset config resolution: which HF dataset, per-language config
names, and label set to use for --dataset masakhanews|afrisenti|sib200.

Split out of data.py so this (and DATASET_NAMES/DatasetSpec) can be imported
without pulling in torch/datasets/sklearn -- needed by modal_app's local
entrypoint, which only builds a job list and never loads a dataset itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DATASET_ID = "masakhane/masakhanews"
DATASET_NAMES = ("masakhanews", "afrisenti", "sib200")


@dataclass
class DatasetSpec:
    hf_name: str
    lang_to_subset: dict  # ISO 639-3 code -> HF dataset config name
    labels: list
    text_field: str
    data_scales: list  # ints and/or None (None = full training split)


def load_dataset_config(dataset: str, base_cfg: dict, configs_dir: Path) -> DatasetSpec:
    """Resolve the per-dataset loading parameters for ``--dataset``.

    MasakhaNEWS's subset name equals the ISO code, so its mapping is built
    from configs/base.yaml's ``languages`` list rather than duplicated in a
    file. AfriSenti and SIB-200 read configs/<dataset>.yaml (same shape as
    DatasetSpec) since their language codes don't always match the dataset's
    own config names (e.g. SIB-200's Oromo is ``gaz``, not ``orm``).
    """
    if dataset == "masakhanews":
        return DatasetSpec(
            hf_name=DATASET_ID,
            lang_to_subset={lang: lang for lang in base_cfg["languages"]},
            labels=base_cfg["labels"],
            text_field=base_cfg.get("text_field", "text"),
            data_scales=base_cfg["data_scales"],
        )
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASET_NAMES}")
    path = configs_dir / f"{dataset}.yaml"
    cfg = yaml.safe_load(path.read_text())
    return DatasetSpec(
        hf_name=cfg["hf_name"],
        lang_to_subset=cfg["lang_to_subset"],
        labels=cfg["labels"],
        text_field=cfg.get("text_field", "text"),
        data_scales=cfg["data_scales"],
    )
