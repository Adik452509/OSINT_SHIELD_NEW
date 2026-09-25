"""Baselines the fine-tuned encoder has to beat.

Five models, each answering a different question:

``majority``
    What you get for free. Proves accuracy is the wrong headline metric.
``tfidf_logreg``
    Word-level bag of n-grams — **the real bar**. If the transformer does not
    clear this convincingly, look for a bug before looking for a bigger model.
``keyword_rules``
    The rubric applied literally, with no learning at all. Fully interpretable
    and a fair test of "can the annotation scheme classify by itself".
``keyword_logreg``
    Learned weights over rubric-keyword counts. Severity keywords alone reach
    F1(High) ≈ 0.685, versus 0.686 for a 50,000-feature TF-IDF model.
``has_body``
    A control fed **no text at all** — only whether the article had a body.
    Scores F1(High) ≈ 0.478, which is why severity is always reported split by
    that flag. Meaningless in ``body_only`` mode, where the feature is constant.

All models take a DataFrame rather than a text array, because the ``has_body``
control needs a column that is not text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..keywords import KeywordMatcher

TEXT_COLUMN = "text"


class BaselineModel(ABC):
    """Common interface so the CV loop can treat every baseline identically."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "BaselineModel":
        """Fit on a training fold."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict labels for a test fold."""


def _most_frequent(y: np.ndarray):
    values, counts = np.unique(np.asarray(y, dtype=object).astype(str), return_counts=True)
    return values[counts.argmax()]


class MajorityBaseline(BaselineModel):
    """Always predicts the most frequent training label."""

    name = "majority"

    def fit(self, df, y):
        self._label = _most_frequent(y)
        self._dtype = np.asarray(y).dtype
        return self

    def predict(self, df):
        return np.full(len(df), self._label).astype(self._dtype)


class TfidfLogRegBaseline(BaselineModel):
    """TF-IDF over word 1–2grams into a class-weighted logistic regression."""

    name = "tfidf_logreg"

    def __init__(self, max_features: int = 50_000, C: float = 1.0):
        self.max_features = max_features
        self.C = C

    def _make(self):
        return (
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=self.max_features,
                            sublinear_tf=True, strip_accents="unicode"),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=self.C),
        )

    def fit(self, df, y):
        self._vec, self._clf = self._make()
        self._clf.fit(self._vec.fit_transform(df[TEXT_COLUMN]), y)
        return self

    def predict(self, df):
        return self._clf.predict(self._vec.transform(df[TEXT_COLUMN]))


class KeywordLogRegBaseline(BaselineModel):
    """Logistic regression over rubric keyword-group counts.

    Features are computed over the **full untruncated** text, which is the same
    property that motivates fusing them into the encoder (docs/DECISIONS.md D4).
    """

    name = "keyword_logreg"

    def __init__(self, *, per_keyword: bool = False, exclude_low_precision: bool = False):
        self.per_keyword = per_keyword
        self._matcher = KeywordMatcher(exclude_low_precision=exclude_low_precision)

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        texts = df[TEXT_COLUMN].tolist()
        if self.per_keyword:
            return self._matcher.per_keyword_features(texts)
        return self._matcher.group_features(texts)

    def fit(self, df, y):
        self._clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        self._clf.fit(self._features(df), y)
        return self

    def predict(self, df):
        return self._clf.predict(self._features(df))


class KeywordRuleBaseline(BaselineModel):
    """The rubric applied literally - argmax over keyword-group counts.

    No learning beyond recording a fallback label for articles where no rubric
    term fires, which is common: the ``Other`` narrative list only matches 19%
    of the articles it should.

    Args:
        task: ``"narrative"``, ``"severity"`` or ``"propaganda"`` - selects
            which rubric section to consult and how to break ties.
    """

    name = "keyword_rules"

    def __init__(self, task: str, *, exclude_low_precision: bool = False):
        if task not in {"narrative", "severity", "propaganda"}:
            raise ValueError(f"unknown task for keyword rules: {task!r}")
        self.task = task
        self._matcher = KeywordMatcher(exclude_low_precision=exclude_low_precision)
        self._prefix = {"narrative": "narr_", "severity": "sev_", "propaganda": "prop_"}[task]

    def fit(self, df, y):
        self._fallback = _most_frequent(y)
        self._dtype = np.asarray(y).dtype
        # only predict classes the training fold actually contains
        self._known = {str(v) for v in np.unique(np.asarray(y, dtype=object).astype(str))}
        return self

    def _predict_one(self, text: str):
        counts = self._matcher.counts(text)
        relevant = {
            group[len(self._prefix):]: n
            for group, n in counts.items()
            if group.startswith(self._prefix)
        }
        if self.task == "propaganda":
            return "1" if sum(relevant.values()) > 0 else "0"

        best = max(relevant, key=lambda k: relevant[k]) if relevant else None
        if best is None or relevant[best] == 0 or best not in self._known:
            return str(self._fallback)
        return best

    def predict(self, df):
        preds = [self._predict_one(t or "") for t in df[TEXT_COLUMN]]
        return np.asarray(preds).astype(self._dtype)


class HasBodyBaseline(BaselineModel):
    """Control fed only ``has_body`` - zero text.

    Exists to keep the confound visible: it reaches F1(High) ≈ 0.478 on the
    full corpus purely because headline-only articles were labelled Low.
    """

    name = "has_body"

    def fit(self, df, y):
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._clf.fit(df[["has_body"]].astype(float), y)
        return self

    def predict(self, df):
        return self._clf.predict(df[["has_body"]].astype(float))


def build_baseline(name: str, task: str) -> BaselineModel:
    """Construct a baseline by name for a given task.

    Raises:
        ValueError: if the name is not a known baseline.
    """
    if name == "majority":
        return MajorityBaseline()
    if name == "tfidf_logreg":
        return TfidfLogRegBaseline()
    if name == "keyword_logreg":
        return KeywordLogRegBaseline()
    if name == "keyword_rules":
        return KeywordRuleBaseline(task)
    if name == "has_body":
        return HasBodyBaseline()
    raise ValueError(f"unknown baseline: {name!r}")


def cross_val_predict(
    df: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    name: str,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per fold and collect out-of-fold predictions.

    Args:
        df: feature frame with ``text`` and ``has_body``.
        y: target labels aligned to ``df``.
        folds: test-fold index per row.
        name: baseline identifier.
        task: task identifier, used by the rule baseline.

    Returns:
        ``(y_true, y_pred)`` in the original row order.
    """
    y = np.asarray(y)
    preds = np.empty(len(df), dtype=object)

    for fold in np.unique(folds):
        test_mask = folds == fold
        train_df, test_df = df[~test_mask], df[test_mask]
        model = build_baseline(name, task).fit(train_df, y[~test_mask])
        preds[test_mask] = model.predict(test_df)

    return y, preds.astype(y.dtype)
