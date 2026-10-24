"""Calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osint_shield.inference import expected_calibration_error, reliability_table, threshold_sweep


def test_perfectly_calibrated_predictions_have_zero_ece():
    conf = np.array([0.8] * 10)
    correct = np.array([True] * 8 + [False] * 2)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_overconfidence_shows_as_a_positive_gap():
    table = reliability_table([0.95] * 10, [True] * 5 + [False] * 5)
    assert table.loc[0, "gap"] == pytest.approx(0.45)


def test_reliability_bands_cover_every_prediction():
    conf = np.linspace(0.3, 1.0, 50)
    table = reliability_table(conf, np.ones(50, dtype=bool))
    assert table["n"].sum() == 50


def test_confidence_of_exactly_one_is_counted():
    assert reliability_table([1.0], [True])["n"].sum() == 1


@pytest.fixture
def oof():
    return pd.DataFrame({
        "conf_narrative": [0.95, 0.9, 0.65, 0.55, 0.85, 0.6],
        "true_narrative": ["Security", "Security", "Political", "Other", "Security", "Civilian"],
        "pred_narrative": ["Security", "Security", "Security", "Security", "Security", "Civilian"],
        "pred_severity": ["Low", "High", "Low", "Low", "Low", "Low"],
    })


def test_raising_the_threshold_never_escalates_fewer(oof):
    sweep = threshold_sweep(oof, [0.5, 0.6, 0.7, 0.8, 0.9])
    assert sweep["escalated_share"].is_monotonic_increasing


def test_predicted_high_escalates_regardless_of_threshold(oof):
    sweep = threshold_sweep(oof, [0.0])
    assert sweep.loc[0, "escalated_share"] == pytest.approx(1 / 6)


def test_sweep_separates_confident_and_uncertain_accuracy(oof):
    row = threshold_sweep(oof, [0.7]).iloc[0]
    assert row["acc_confident"] == pytest.approx(1.0)        # 0.95, 0.9, 0.85 all right
    assert row["acc_uncertain"] == pytest.approx(1 / 3)      # 0.65, 0.55 wrong; 0.6 right
    assert row["errors_caught"] == pytest.approx(1.0)        # both errors fall below 0.7
