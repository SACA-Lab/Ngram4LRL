from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class Split:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


def _encode_labels(ds, text_field: str, label_field: str) -> tuple[Split, Split, Split]:
    """Build (train, val, test) Splits, encoding labels to ints.

    A ClassLabel-typed column (e.g. SIB-200's `label`) already yields ints
    when read, so it's passed through as-is. A plain string column (e.g.
    MasakhaNEWS's `category`, AfriSenti's `label` — neither is a ClassLabel
    despite the name) is mapped via the feature's str2int when available,
    else a sorted alphabetical mapping built from the train split.
    """
    sample = ds["train"][0][label_field]
    if not isinstance(sample, str):
        encode = int  # already integer-encoded (ClassLabel column)
    else:
        label_feature = ds["train"].features[label_field]
        if hasattr(label_feature, "str2int"):
            encode = label_feature.str2int
        else:
            unique = sorted({row[label_field] for row in ds["train"]})
            _map = {v: i for i, v in enumerate(unique)}
            encode = _map.__getitem__

    def _to_split(subset) -> Split:
        return Split(
            texts=[row[text_field] for row in subset],
            labels=[encode(row[label_field]) for row in subset],
        )

    return _to_split(ds["train"]), _to_split(ds["validation"]), _to_split(ds["test"])


def load_masakhanews(
    lang: str,
    text_field: str = "text",
    label_field: str = "category",
) -> tuple[Split, Split, Split]:
    """Return (train, val, test) splits for a MasakhaNEWS language."""
    ds = load_dataset("masakhane/masakhanews", lang)
    return _encode_labels(ds, text_field, label_field)


def load_afrisenti(lang: str) -> tuple[Split, Split, Split]:
    """Return (train, val, test) splits for an AfriSenti language."""
    ds = load_dataset("masakhane/afrisenti", lang)
    return _encode_labels(ds, text_field="tweet", label_field="label")


# Short language code -> mteb/sib200 HF config name (FLORES-200 style codes).
SIB200_CONFIGS: dict[str, str] = {
    "hau": "hau_Latn",
    "ibo": "ibo_Latn",
    "lug": "lug_Latn",
    "gaz": "gaz_Latn",   # Oromo
    "run": "run_Latn",
    "sna": "sna_Latn",
    "swh": "swh_Latn",   # Swahili
    "yor": "yor_Latn",
    "amh": "amh_Ethi",
}


def load_sib200(lang: str) -> tuple[Split, Split, Split]:
    """Return (train, val, test) splits for a SIB-200 language."""
    if lang not in SIB200_CONFIGS:
        raise ValueError(f"Unknown SIB-200 language '{lang}'. "
                         f"Choose from: {list(SIB200_CONFIGS)}")
    ds = load_dataset("mteb/sib200", SIB200_CONFIGS[lang])
    return _encode_labels(ds, text_field="text", label_field="label")


@dataclass
class DatasetSpec:
    loader: Callable[..., tuple[Split, Split, Split]]
    languages: list[str]
    scales: list  # ints and/or "full"


DATASETS: dict[str, DatasetSpec] = {
    "masakhanews": DatasetSpec(
        loader=load_masakhanews,
        languages=["lug", "run", "sna", "swa", "amh", "ibo", "yor", "orm", "pcm", "hau"],
        scales=[100, 250, 500, 750, "full"],
    ),
    "afrisenti": DatasetSpec(
        loader=load_afrisenti,
        languages=["amh", "hau", "ibo", "orm", "pcm", "swa", "yor"],
        scales=[100, 250, 500, 750, "full"],
    ),
    "sib200": DatasetSpec(
        loader=load_sib200,
        languages=["hau", "ibo", "lug", "gaz", "run", "sna", "swh", "yor", "amh"],
        scales=[100, 250, 500, "full"],
    ),
}


def subsample(split: Split, n: int, seed: int) -> Split:
    """Stratified subsample of exactly n examples. Returns full split if n >= len(split)."""
    if n >= len(split):
        return split

    indices = np.arange(len(split))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    sampled_idx, _ = next(sss.split(indices, split.labels))

    return Split(
        texts=[split.texts[i] for i in sampled_idx],
        labels=[split.labels[i] for i in sampled_idx],
    )
