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

from ..config import collapse_map, narrative_label_maps, severity_label_maps

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


def score_multitask(y_true: dict, y_pred: dict, *, has_body=None) -> dict:
    """Score every head the model has, using the project's reporting rules.

    Args:
        y_true: ``{task: integer labels}``.
        y_pred: ``{task: integer predictions}``. Tasks absent here are skipped.
        has_body: per-row flag; when given, severity is also broken out by it.

    Returns:
        Flat dict of scalar metrics, plus ``severity_by_has_body`` when
        ``has_body`` is supplied. ``propaganda_n_pos`` always rides alongside
        the propaganda score - it is meaningless without it.
    """
    out: dict = {}

    if "narrative" in y_pred:
        _, id_to_name = narrative_label_maps()
        t = np.asarray(y_true["narrative"], dtype=int)
        p = np.asarray(y_pred["narrative"], dtype=int)
        out["narrative_macro_f1"] = macro_f1(t, p)
        out["narrative_accuracy"] = float(accuracy_score(t, p))
        names_t = [id_to_name[i] for i in t]
        names_p = [id_to_name[i] for i in p]
        out["narrative_collapsed_macro_f1"] = macro_f1(
            collapse_narrative(names_t), collapse_narrative(names_p)
        )

    if "severity" in y_pred:
        _, id_to_sev = severity_label_maps()
        t = np.asarray(y_true["severity"], dtype=int)
        p = np.asarray(y_pred["severity"], dtype=int)
        out["severity_f1_high"] = binary_f1(t, p, pos_label=1)
        # Macro over Low and High. Unlike F1(High), a degenerate "everything is
        # High" predictor cannot score well here: at ~30% prevalence it gets
        # F1(High) ~0.46 but severity macro-F1 only ~0.23. Used for early
        # stopping (docs/DECISIONS.md D11); F1(High) stays the reported metric.
        out["severity_macro_f1"] = macro_f1(t, p)
        out["severity_accuracy"] = float(accuracy_score(t, p))
        if has_body is not None:
            out["severity_by_has_body"] = severity_by_has_body(
                [id_to_sev[i] for i in t], [id_to_sev[i] for i in p], has_body
            )

    if "propaganda" in y_pred:
        t = np.asarray(y_true["propaganda"], dtype=int)
        p = np.asarray(y_pred["propaganda"], dtype=int)
        out["propaganda_f1_pos"] = binary_f1(t, p, pos_label=1)
        out["propaganda_n_pos"] = int(t.sum())
        out["propaganda_n_pred_pos"] = int(p.sum())

    return out


def leakage_alarm(y_true, y_pred) -> tuple[bool, float]:
    """Flag narrative accuracy implausibly above the annotator ceiling.

    Returns ``(triggered, accuracy)``. A trigger means check for duplicate
    articles across folds before believing the number.
    """
    acc = float(accuracy_score(np.asarray(y_true, dtype=object).astype(str),
                               np.asarray(y_pred, dtype=object).astype(str)))
    return acc > LEAKAGE_ALARM_ACCURACY, acc
