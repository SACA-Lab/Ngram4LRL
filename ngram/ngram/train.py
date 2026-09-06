from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any

from sklearn.metrics import f1_score

from ngram.data import DATASETS, Split, subsample
from ngram.features import FEATURE_CONFIGS, build_vectorizer
from ngram.models import build_naive_bayes, build_svm, build_xgboost

# ── Hyperparameter grids (tuned on val set) ────────────────────────────────────
CLF_GRIDS: dict[str, dict[str, list]] = {
    "naive_bayes": {"alpha": [0.01, 0.1, 0.5, 1.0]},
    "svm":         {"C":     [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "xgboost":     {"n_estimators": [100, 200, 300, 500]},
}

# Fixed XGBoost settings not subject to search
_XGB_FIXED = dict(max_depth=6, learning_rate=0.1, subsample=0.8)


def _make_classifier(name: str, params: dict[str, Any]):
    if name == "naive_bayes":
        return build_naive_bayes(**params)
    if name == "svm":
        return build_svm(**params)
    if name == "xgboost":
        return build_xgboost(**params, **_XGB_FIXED)
    raise ValueError(f"Unknown classifier: {name!r}")


def _tune(
    clf_name: str,
    train: Split,
    val: Split,
    tfidf_kwargs: dict,
) -> tuple[str, dict[str, Any], float]:
    """
    Joint grid search over (feature_type × clf_params).
    Returns (best_feature_type, best_clf_params, val_weighted_f1).
    """
    grid = CLF_GRIDS[clf_name]
    param_names = list(grid.keys())
    param_values = list(grid.values())

    best_val_f1 = -1.0
    best_feature_type = ""
    best_params: dict[str, Any] = {}

    for feature_type in FEATURE_CONFIGS:
        vec = build_vectorizer(feature_type, **tfidf_kwargs)
        X_tr  = vec.fit_transform(train.texts)
        X_val = vec.transform(val.texts)

        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))
            clf = _make_classifier(clf_name, params)
            clf.fit(X_tr, train.labels)
            preds = clf.predict(X_val)
            f1 = f1_score(val.labels, preds, average="weighted", zero_division=0)

            if f1 > best_val_f1:
                best_val_f1   = f1
                best_feature_type = feature_type
                best_params   = params

    return best_feature_type, best_params, best_val_f1


@dataclass
class TrialResult:
    dataset: str                  # "masakhanews" | "afrisenti" | "sib200"
    lang: str
    scale: str                    # "100" | "250" | "500" | "750" | "full"
    seed: int
    classifier: str
    best_feature_type: str
    best_clf_params: dict[str, Any]
    val_f1: float
    test_f1: float
    n_train: int                  # actual training examples used (after subsampling)

    def to_dict(self) -> dict:
        return asdict(self)


def run_trial(
    lang: str,
    scale: int | str,
    seed: int,
    classifier_name: str,
    dataset: str = "masakhanews",
    text_field: str = "text",
    tfidf_kwargs: dict | None = None,
) -> TrialResult:
    if tfidf_kwargs is None:
        tfidf_kwargs = dict(sublinear_tf=True, max_features=50000, min_df=1)

    loader = DATASETS[dataset].loader
    if dataset == "masakhanews":
        train_full, val, test = loader(lang, text_field=text_field)
    else:
        train_full, val, test = loader(lang)

    train = train_full if scale == "full" else subsample(train_full, int(scale), seed=seed)

    best_feature_type, best_params, val_f1 = _tune(
        classifier_name, train, val, tfidf_kwargs
    )

    # Final model: refit on training split with the chosen hyperparameters
    vec     = build_vectorizer(best_feature_type, **tfidf_kwargs)
    X_train = vec.fit_transform(train.texts)
    X_test  = vec.transform(test.texts)

    clf = _make_classifier(classifier_name, best_params)
    clf.fit(X_train, train.labels)
    preds   = clf.predict(X_test)
    test_f1 = f1_score(test.labels, preds, average="weighted", zero_division=0)

    return TrialResult(
        dataset=dataset,
        lang=lang,
        scale=str(scale),
        seed=seed,
        classifier=classifier_name,
        best_feature_type=best_feature_type,
        best_clf_params=best_params,
        val_f1=round(val_f1, 6),
        test_f1=round(test_f1, 6),
        n_train=len(train),
    )
