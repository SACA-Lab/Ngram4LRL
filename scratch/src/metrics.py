"""Evaluation metrics and significance tests."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def weighted_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def paired_t_test(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
) -> tuple[float, float]:
    """Paired two-sided t-test. Returns (t-statistic, p-value)."""
    a = np.asarray(a_scores, dtype=float)
    b = np.asarray(b_scores, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Score vectors must have the same shape")
    res = stats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def has_crossed_over(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    alpha: float = 0.05,
) -> bool:
    """True if the mean of ``a_scores`` exceeds that of ``b_scores`` and the
    paired t-test p-value is below ``alpha``."""
    a = np.asarray(a_scores, dtype=float)
    b = np.asarray(b_scores, dtype=float)
    if a.mean() <= b.mean():
        return False
    _, p = paired_t_test(a, b)
    return p < alpha
