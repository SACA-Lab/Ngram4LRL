from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class Split:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


def load_masakhanews(
    lang: str,
    text_field: str = "text",
    label_field: str = "category",
) -> tuple[Split, Split, Split]:
    """Return (train, val, test) splits for a MasakhaNEWS language."""
    ds = load_dataset("masakhane/masakhanews", lang)

    # Use the dataset's ClassLabel encoding when available (strings → ints).
    # Falls back to a sorted alphabetical mapping for robustness.
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
