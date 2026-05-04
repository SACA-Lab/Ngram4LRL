from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

# Candidate feature configurations — the choice is treated as a hyperparameter
# and selected jointly with classifier hyperparameters on the validation set.
FEATURE_CONFIGS: dict[str, dict] = {
    "word_unigram": dict(analyzer="word",    ngram_range=(1, 1)),
    "word_bigram":  dict(analyzer="word",    ngram_range=(1, 2)),
    "char_3to5":    dict(analyzer="char_wb", ngram_range=(3, 5)),
}


def build_vectorizer(
    feature_type: str,
    sublinear_tf: bool = True,
    max_features: int = 50000,
    min_df: int = 1,
) -> TfidfVectorizer:
    if feature_type not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature_type '{feature_type}'. "
                         f"Choose from: {list(FEATURE_CONFIGS)}")
    return TfidfVectorizer(
        **FEATURE_CONFIGS[feature_type],
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        min_df=min_df,
    )
