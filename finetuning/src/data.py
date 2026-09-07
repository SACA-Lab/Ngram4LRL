"""Dataset loading and stratified subsampling.

Supports MasakhaNEWS (the original task), AfriSenti, and SIB-200. MasakhaNEWS's
per-language config names and labels stay in configs/base.yaml, as before;
AfriSenti and SIB-200 read the equivalent fields from configs/<dataset>.yaml.
See ``dataset_config.load_dataset_config``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

from .dataset_config import DATASET_ID, DATASET_NAMES, DatasetSpec, load_dataset_config

__all__ = [
    "DATASET_ID", "DATASET_NAMES", "DatasetSpec", "load_dataset_config",
    "SplitSizes", "load_dataset_split", "load_masakhanews",
    "stratified_subsample", "class_counts", "split_sizes",
]


@dataclass
class SplitSizes:
    train: int
    val: int
    test: int


def _normalize_split(
    ds: Dataset,
    label_names: Sequence[str],
    text_field: str = "text",
    split_name: str = "",
) -> Dataset:
    """Project the raw dataset to (text, label) and drop rows whose category
    is not in ``label_names``."""
    cols = set(ds.column_names)

    if text_field not in cols:
        raise KeyError(f"text_field {text_field!r} not in dataset columns {sorted(cols)}")

    cat_field = "category" if "category" in cols else ("label" if "label" in cols else None)
    if cat_field is None:
        raise KeyError(f"no category/label column in {sorted(cols)}")

    feat = ds.features[cat_field]
    is_class_label = isinstance(feat, ClassLabel)
    ds_label_names = feat.names if is_class_label else None

    label_to_id = {name.lower(): i for i, name in enumerate(label_names)}

    def _cat_str(ex):
        raw = ex[cat_field]
        return (ds_label_names[raw] if is_class_label else str(raw)).lower()

    n_before = len(ds)
    kept = ds.filter(lambda ex: _cat_str(ex) in label_to_id)
    n_after = len(kept)
    if n_after < n_before:
        tag = f" [{split_name}]" if split_name else ""
        print(
            f"  [data]{tag} dropped {n_before - n_after}/{n_before} rows "
            f"with categories outside {sorted(label_to_id)}"
        )

    def _map(ex):
        return {"text": ex[text_field], "label": label_to_id[_cat_str(ex)]}

    return kept.map(_map, remove_columns=list(cols))


def load_dataset_split(spec: DatasetSpec, lang: str) -> DatasetDict:
    """Return train/validation/test splits normalised to (text, label)."""
    if lang not in spec.lang_to_subset:
        raise ValueError(f"Unknown language code: {lang}")
    raw = load_dataset(spec.hf_name, spec.lang_to_subset[lang])
    return DatasetDict(
        {
            split: _normalize_split(raw[split], spec.labels, spec.text_field, split)
            for split in raw
        }
    )


def load_masakhanews(
    lang: str,
    label_names: Sequence[str],
    text_field: str = "text",
) -> DatasetDict:
    """Return MasakhaNEWS splits for one language.

    Thin convenience wrapper kept for callers that only ever use MasakhaNEWS
    (e.g. scripts/tables.py's dataset_stats table); prefer
    ``load_dataset_split`` for a ``--dataset``-parameterised call site.
    """
    spec = DatasetSpec(
        hf_name=DATASET_ID,
        lang_to_subset={lang: lang},
        labels=list(label_names),
        text_field=text_field,
        data_scales=[],
    )
    return load_dataset_split(spec, lang)


def stratified_subsample(train: Dataset, n: Optional[int], seed: int) -> Dataset:
    """Return a stratified subsample of size ``n``. If ``n`` is ``None`` or
    exceeds the split size, the full split is returned."""
    if n is None or n >= len(train):
        return train

    labels = np.array(train["label"])
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return train.select(sorted(idx.tolist()))


def class_counts(ds: Dataset, num_labels: int) -> np.ndarray:
    """Return per-class example counts as a length-``num_labels`` array."""
    counts = np.zeros(num_labels, dtype=int)
    for y in ds["label"]:
        counts[y] += 1
    return counts


def split_sizes(ds: DatasetDict) -> SplitSizes:
    return SplitSizes(
        train=len(ds["train"]),
        val=len(ds["validation"]),
        test=len(ds["test"]),
    )
