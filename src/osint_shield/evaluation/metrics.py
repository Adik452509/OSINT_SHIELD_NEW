"""Scoring, with the project's reporting rules built in.

Three rules are enforced here rather than left to each caller:

* **Macro-F1, not accuracy.** Predicting ``Security`` for every article scores
  77.4% accuracy and 0.175 macro-F1. Accuracy is reported only alongside.
* **Mean ± std across folds.** A single number hides that the rare classes swing
  by tens of points between folds; the spread is itself the result.
* **Severity is always broken out by ``has_body``.** A classifier fed nothing but
  that flag scores F1(High) ≈ 0.478, so any severity number that does not
  separate the two regimes is not interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..config import collapse_map

#: Narrative accuracy above this is a leakage alarm, not a success - the
#: annotators only agreed with each other 74.3% of the time.
LEAKAGE_ALARM_ACCURACY = 0.92


def collapse_narrative(labels) -> np.ndarray:
    """Map 5-class narrative labels onto the 3-class view.

    ``Civilian``, ``Investigation`` and ``Other`` merge into ``Other``; the
    collapsed macro-F1 is the more meaningful headline number because the
    5-class one is dominated by classes with fewer than 30 examples.
    """
    cmap = collapse_map()
    out = []
    for label in np.asarray(labels):
        key = str(label)
        if key not in cmap:
            raise KeyError(f"no collapse target for narrative class {key!r}")
        out.append(cmap[key])
    return np.asarray(out)


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 over the classes present in truth or prediction."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def binary_f1(y_true, y_pred, pos_label) -> float:
    """F1 on one class specifically - used for severity High and propaganda."""
    return float(
        f1_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
    )


def per_class_f1(y_true, y_pred) -> dict[str, float]:
    """F1 per class, keyed by label. Classes never seen score 0.0, not NaN."""
    labels = sorted({str(v) for v in np.concatenate([np.asarray(y_true, dtype=object),
                                                     np.asarray(y_pred, dtype=object)])})
    scores = f1_score(
        np.asarray(y_true, dtype=object).astype(str),
        np.asarray(y_pred, dtype=object).astype(str),
        labels=labels, average=None, zero_division=0,
    )
    return {label: float(score) for label, score in zip(labels, scores)}


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix as a labelled frame - rows are truth, columns predicted."""
    labels = sorted({str(v) for v in np.concatenate([np.asarray(y_true, dtype=object),
                                                     np.asarray(y_pred, dtype=object)])})
    matrix = confusion_matrix(
        np.asarray(y_true, dtype=object).astype(str),
        np.asarray(y_pred, dtype=object).astype(str),
        labels=labels,
    )
    return pd.DataFrame(matrix, index=labels, columns=labels)


def mean_std(values) -> tuple[float, float]:
    """Mean and population standard deviation of per-fold scores."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


@dataclass
class FoldScores:
    """Per-fold scores for one model on one task.

    Attributes:
        name: model identifier.
        task: task identifier.
        folds: one score per fold, in fold order.
        extra: anything worth carrying alongside, e.g. positives in the test
            fold - mandatory context for the propaganda number.
    """

    name: str
    task: str
    folds: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return mean_std(self.folds)[0]

    @property
    def std(self) -> float:
        return mean_std(self.folds)[1]

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "task": self.task,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "folds": [round(f, 4) for f in self.folds],
            **self.extra,
        }

    def __str__(self) -> str:
        return f"{self.mean:.3f} +/- {self.std:.3f}"


def severity_by_has_body(
    y_true, y_pred, has_body, *, pos_label: str = "High"
) -> dict[str, dict[str, float]]:
    """Severity performance split by whether the article had body text.

    The two regimes are not comparable: headline-only articles are 9.5% High
    against 34.0% for bodied ones, so a single pooled number conflates a real
    signal with a collection artifact.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    has_body = np.asarray(has_body, dtype=bool)

    out: dict[str, dict[str, float]] = {}
    for label, mask in (("has_body", has_body), ("headline_only", ~has_body)):
        if not mask.any():
            continue
        out[label] = {
            "n": int(mask.sum()),
            "f1_high": binary_f1(y_true[mask], y_pred[mask], pos_label),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "actual_high_rate": float((y_true[mask] == pos_label).mean()),
        }
    return out


def leakage_alarm(y_true, y_pred) -> tuple[bool, float]:
    """Flag narrative accuracy implausibly above the annotator ceiling.

    Returns ``(triggered, accuracy)``. A trigger means check for duplicate
    articles across folds before believing the number.
    """
    acc = float(accuracy_score(np.asarray(y_true, dtype=object).astype(str),
                               np.asarray(y_pred, dtype=object).astype(str)))
    return acc > LEAKAGE_ALARM_ACCURACY, acc
