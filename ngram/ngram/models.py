from __future__ import annotations

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import xgboost as xgb


def build_naive_bayes(alpha: float = 1.0) -> MultinomialNB:
    return MultinomialNB(alpha=alpha)


def build_svm(C: float = 1.0) -> LinearSVC:
    # L2 penalty is LinearSVC's default; max_iter raised to ensure convergence
    # on the larger vocabulary spaces produced by char n-grams.
    return LinearSVC(C=C, penalty="l2", max_iter=4000, random_state=42)


def build_xgboost(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
