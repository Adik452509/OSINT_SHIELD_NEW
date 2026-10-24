"""Is the model's confidence trustworthy, and where should escalation start?

Measured on development data only - the cross-validation out-of-fold
predictions of the D15 recipe - never ``test.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def reliability_table(confidence, correct, edges=DEFAULT_EDGES) -> pd.DataFrame:
    """Accuracy per confidence band. A calibrated model's ``accuracy`` tracks
    its ``mean_confidence``; ``gap`` > 0 means over-confident."""
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            rows.append({"band": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(m.sum()),
                         "share": float(m.mean()),
                         "mean_confidence": float(conf[m].mean()),
                         "accuracy": float(ok[m].mean()),
                         "gap": float(conf[m].mean() - ok[m].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """Weighted mean |confidence - accuracy| over equal-width bins."""
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(total)


def threshold_sweep(oof: pd.DataFrame, thresholds) -> pd.DataFrame:
    """What each escalation threshold would do, on out-of-fold predictions.

    Escalation = predicted High OR narrative confidence < threshold (plan §6.2).
    ``acc_confident`` vs ``acc_uncertain`` shows whether low confidence really
    marks the errors - the whole premise of confidence-based escalation.
    """
    conf = oof["conf_narrative"].to_numpy(dtype=float)
    ok = (oof["true_narrative"] == oof["pred_narrative"]).to_numpy()
    high = (oof["pred_severity"] == "High").to_numpy()
    rows = []
    for t in thresholds:
        low = conf < t
        rows.append({
            "threshold": float(t),
            "low_confidence_share": float(low.mean()),
            "escalated_share": float((low | high).mean()),
            "acc_confident": float(ok[~low].mean()) if (~low).any() else float("nan"),
            "acc_uncertain": float(ok[low].mean()) if low.any() else float("nan"),
            "errors_caught": float((~ok & low).sum() / max((~ok).sum(), 1)),
        })
    return pd.DataFrame(rows)
