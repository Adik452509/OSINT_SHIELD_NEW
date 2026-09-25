"""Metrics and baseline models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osint_shield.evaluation.baselines import (
    HasBodyBaseline,
    KeywordRuleBaseline,
    MajorityBaseline,
    TfidfLogRegBaseline,
    build_baseline,
    cross_val_predict,
)
from osint_shield.evaluation.metrics import (
    FoldScores,
    binary_f1,
    collapse_narrative,
    confusion_frame,
    leakage_alarm,
    macro_f1,
    mean_std,
    per_class_f1,
    severity_by_has_body,
)


# -------------------------------------------------------------------- metrics
def test_collapse_narrative_merges_the_three_rare_classes():
    out = collapse_narrative(
        ["Security", "Political", "Civilian", "Investigation", "Other"]
    )
    assert list(out) == ["Security", "Political", "Other", "Other", "Other"]


def test_collapse_narrative_rejects_unknown_class():
    with pytest.raises(KeyError, match="no collapse target"):
        collapse_narrative(["Cricket"])


def test_macro_f1_is_perfect_when_predictions_match():
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_macro_f1_punishes_ignoring_a_rare_class():
    """The whole reason macro-F1 is the primary metric."""
    truth = ["Security"] * 9 + ["Investigation"]
    always_majority = ["Security"] * 10
    assert macro_f1(truth, always_majority) < 0.5


def test_binary_f1_is_zero_when_the_positive_class_is_never_predicted():
    assert binary_f1(["High", "Low", "Low"], ["Low"] * 3, "High") == 0.0


def test_per_class_f1_reports_every_observed_label():
    scores = per_class_f1(["a", "b", "c"], ["a", "b", "b"])
    assert set(scores) == {"a", "b", "c"}
    assert scores["a"] == pytest.approx(1.0)
    assert scores["c"] == 0.0


def test_confusion_frame_is_square_and_labelled():
    cm = confusion_frame(["a", "b", "a"], ["a", "a", "b"])
    assert list(cm.index) == list(cm.columns) == ["a", "b"]
    assert cm.loc["a", "a"] == 1


def test_mean_std_of_empty_is_nan():
    mean, std = mean_std([])
    assert np.isnan(mean) and np.isnan(std)


def test_mean_std_values():
    mean, std = mean_std([0.4, 0.6])
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(0.1)


def test_fold_scores_summarise_and_serialise():
    s = FoldScores(name="m", task="t", folds=[0.4, 0.6], extra={"pos": 3})
    assert s.mean == pytest.approx(0.5)
    assert "0.500" in str(s)
    assert s.as_dict()["pos"] == 3


def test_leakage_alarm_triggers_only_above_the_annotator_ceiling():
    truth = ["a"] * 100
    assert leakage_alarm(truth, truth)[0] is True
    mixed = ["a"] * 80 + ["b"] * 20
    assert leakage_alarm(mixed, ["a"] * 100)[0] is False


def test_severity_split_by_has_body_reports_both_regimes():
    out = severity_by_has_body(
        y_true=["High", "Low", "Low", "High"],
        y_pred=["High", "Low", "High", "High"],
        has_body=[True, True, False, False],
    )
    assert set(out) == {"has_body", "headline_only"}
    assert out["has_body"]["n"] == 2
    assert out["headline_only"]["actual_high_rate"] == pytest.approx(0.5)


def test_severity_split_omits_an_absent_regime():
    """In body_only mode there are no headline-only rows to report."""
    out = severity_by_has_body(["High", "Low"], ["High", "Low"], [True, True])
    assert set(out) == {"has_body"}


# ------------------------------------------------------------------ baselines
@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "text": [
            "DRDO conducts missile test and deployment of armed forces",
            "Foreign minister holds bilateral talks at the summit",
            "civilians displaced humanitarian relief operations under way",
            "DRDO naval fleet border patrol counter-terrorism operation",
            "ambassador and parliament discuss foreign policy",
            "rescue and evacuation of injured civilians after earthquake relief",
        ],
        "has_body": [True, True, True, False, False, False],
    })


def test_majority_predicts_the_most_frequent_label(frame):
    y = np.array(["Security"] * 4 + ["Political"] * 2)
    model = MajorityBaseline().fit(frame, y)
    assert set(model.predict(frame)) == {"Security"}


def test_majority_preserves_label_dtype(frame):
    y = np.array([0, 0, 0, 1, 1, 0])
    preds = MajorityBaseline().fit(frame, y).predict(frame)
    assert preds.dtype == y.dtype


def test_keyword_rules_pick_the_matching_class(frame):
    y = np.array(["Security", "Political", "Civilian"] * 2)
    preds = KeywordRuleBaseline("narrative").fit(frame, y).predict(frame)
    assert preds[0] == "Security"
    assert preds[1] == "Political"
    assert preds[2] == "Civilian"


def test_keyword_rules_fall_back_when_nothing_fires():
    df = pd.DataFrame({"text": ["the weather is pleasant today"], "has_body": [True]})
    y = np.array(["Security"])
    assert KeywordRuleBaseline("narrative").fit(df, y).predict(df)[0] == "Security"


def test_keyword_rules_never_invent_an_unseen_class(frame):
    """A class absent from training must not be predicted."""
    y = np.array(["Security"] * 6)
    preds = KeywordRuleBaseline("narrative").fit(frame, y).predict(frame)
    assert set(preds) == {"Security"}


def test_keyword_rules_reject_unknown_task():
    with pytest.raises(ValueError, match="unknown task"):
        KeywordRuleBaseline("sentiment")


def test_tfidf_logreg_learns_a_separable_toy_problem(frame):
    y = np.array(["Security", "Political", "Civilian",
                  "Security", "Political", "Civilian"])
    preds = TfidfLogRegBaseline().fit(frame, y).predict(frame)
    assert (preds == y).mean() >= 0.5


def test_has_body_baseline_uses_only_the_flag(frame):
    """It must separate the regimes using no text whatsoever."""
    y = np.array(["High", "High", "High", "Low", "Low", "Low"])
    preds = HasBodyBaseline().fit(frame, y).predict(frame)
    assert list(preds) == ["High", "High", "High", "Low", "Low", "Low"]


def test_build_baseline_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown baseline"):
        build_baseline("random_forest", "narrative")


def test_cross_val_predict_covers_every_row_once(frame):
    y = np.array(["Security", "Political", "Civilian"] * 2)
    folds = np.array([0, 0, 1, 1, 2, 2])
    y_true, y_pred = cross_val_predict(frame, y, folds, "majority", "narrative")
    assert len(y_pred) == len(frame)
    assert (y_true == y).all()
    assert not pd.isna(y_pred).any()


def test_cross_val_predict_never_trains_on_the_test_fold(frame):
    """A label appearing only in the test fold must never be predicted for it."""
    y = np.array(["Security"] * 5 + ["Unseen"])
    folds = np.array([0, 0, 0, 0, 0, 1])
    _, y_pred = cross_val_predict(frame, y, folds, "majority", "narrative")
    assert y_pred[5] == "Security"
