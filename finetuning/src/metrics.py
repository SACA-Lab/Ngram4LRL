"""Evaluation metrics and significance tests."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def weighted_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Weighted F1 with zero-division returning zero."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def paired_t_test(
    neural_scores: Sequence[float],
    ngram_scores: Sequence[float],
) -> tuple[float, float]:
    """Paired two-sided t-test on per-seed F1 vectors. Returns (t, p)."""
    a = np.asarray(neural_scores, dtype=float)
    b = np.asarray(ngram_scores, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Score vectors must have the same shape")
    res = stats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def has_crossed_over(
    neural_scores: Sequence[float],
    ngram_scores: Sequence[float],
    alpha: float = 0.05,
) -> bool:
    """Return True if the neural mean exceeds the n-gram mean and the paired
    t-test p-value is below ``alpha``."""
    a = np.asarray(neural_scores, dtype=float)
    b = np.asarray(ngram_scores, dtype=float)
    if a.mean() <= b.mean():
        return False
    _, p = paired_t_test(a, b)
    return p < alpha
