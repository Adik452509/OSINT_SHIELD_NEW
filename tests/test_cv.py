"""Cross-validation runner: resumability, config locking, aggregation."""

from __future__ import annotations

import json

import pytest
import torch

from osint_shield.config import load_config
from osint_shield.training.cv import (
    ConfigMismatchError,
    ensure_snapshot,
    load_predictions,
    run_cv,
    run_path,
    runs_frame,
    selection_analysis,
    summarise,
)
from tests.test_training import fast_settings, silent, tiny_builder, toy_data

CPU = torch.device("cpu")


def fake_result(seed: int, fold: int, narr: float, best_epoch: int = 5) -> dict:
    return {
        "seed": seed, "fold": fold, "best_epoch": best_epoch,
        "history": [{}] * best_epoch, "seconds": 10.0, "peak_vram_gb": 2.5,
        "metrics": {
            "narrative_macro_f1": narr, "narrative_collapsed_macro_f1": narr + 0.1,
            "severity_f1_high": 0.6, "propaganda_f1_pos": 0.0,
            "narrative_accuracy": 0.8, "severity_accuracy": 0.7,
            "propaganda_n_pos": 1, "propaganda_n_pred_pos": 0,
        },
    }


# ----------------------------------------------------------------- aggregation
def test_runs_frame_flattens_one_row_per_run():
    df = runs_frame([fake_result(42, 1, 0.5), fake_result(42, 0, 0.4)])
    assert list(df["fold"]) == [0, 1]                      # sorted
    assert df.loc[0, "narrative_macro_f1"] == pytest.approx(0.4)


def test_runs_frame_rejects_empty():
    with pytest.raises(ValueError, match="no completed runs"):
        runs_frame([])


def test_summarise_separates_seed_noise_from_fold_variation():
    # fold effect is large, seed effect is zero
    results = [fake_result(s, f, 0.3 + 0.1 * f) for s in (1, 2) for f in (0, 1, 2)]
    s = summarise(runs_frame(results), max_epochs=10)["metrics"]["narrative_macro_f1"]
    assert s["mean"] == pytest.approx(0.4)
    assert s["seed_spread"] == pytest.approx(0.0)
    assert s["fold_spread"] > 0.05


def test_summarise_flags_runs_at_the_epoch_ceiling():
    results = [fake_result(42, f, 0.5, best_epoch=10) for f in range(3)]
    results.append(fake_result(42, 3, 0.5, best_epoch=4))
    be = summarise(runs_frame(results), max_epochs=10)["best_epoch"]
    assert be["share_at_ceiling"] == pytest.approx(0.75)


def test_summarise_counts_leakage_alarms():
    r = fake_result(42, 0, 0.5)
    r["metrics"]["narrative_accuracy"] = 0.97
    assert summarise(runs_frame([r]), max_epochs=10)["leakage_alarms"] == 1


def _tracked(seed, fold, curve, selected_epoch):
    """A fake run whose test-fold narrative score followed ``curve`` by epoch."""
    r = fake_result(seed, fold, curve[selected_epoch - 1], best_epoch=selected_epoch)
    r["history"] = [{"epoch": e + 1, "track_narrative_macro_f1": v,
                     "track_severity_f1_high": 0.6} for e, v in enumerate(curve)]
    return r


def test_selection_analysis_compares_selected_final_and_oracle():
    run = _tracked(42, 0, [0.1, 0.5, 0.3], selected_epoch=3)   # chose badly
    s = selection_analysis([run])["narrative_macro_f1"]
    assert s["selected"] == pytest.approx(0.3)
    assert s["final_epoch"] == pytest.approx(0.3)
    assert s["oracle"] == pytest.approx(0.5)
    assert s["curve"] == {1: 0.1, 2: 0.5, 3: 0.3}


def test_selection_curve_averages_runs_of_different_length():
    """Early-stopped runs are shorter; each epoch averages only runs that reached it."""
    a = _tracked(42, 0, [0.2, 0.4], selected_epoch=2)
    b = _tracked(42, 1, [0.4, 0.6, 0.8], selected_epoch=3)
    s = selection_analysis([a, b])["narrative_macro_f1"]
    assert s["curve"][3] == pytest.approx(0.8)
    assert s["curve_n"] == {1: 2, 2: 2, 3: 1}


def test_selection_analysis_is_none_without_tracking():
    assert selection_analysis([fake_result(42, 0, 0.5)]) is None


def test_summary_is_json_serialisable():
    s = summarise(runs_frame([fake_result(42, 0, 0.5)]), max_epochs=10)
    json.dumps(s)


# ------------------------------------------------------------- config locking
def test_snapshot_written_on_first_run(tmp_path):
    ensure_snapshot(tmp_path, {"lr": 2e-5}, resume=True)
    assert (tmp_path / "config.json").exists()


def test_resume_under_the_same_config_is_allowed(tmp_path):
    ensure_snapshot(tmp_path, {"lr": 2e-5}, resume=True)
    ensure_snapshot(tmp_path, {"lr": 2e-5}, resume=True)


def test_resume_under_a_changed_config_is_refused(tmp_path):
    """Averaging two experiments silently is the failure this prevents."""
    ensure_snapshot(tmp_path, {"lr": 2e-5}, resume=True)
    with pytest.raises(ConfigMismatchError, match="config has changed"):
        ensure_snapshot(tmp_path, {"lr": 5e-5}, resume=True)


def test_no_resume_overwrites_the_snapshot(tmp_path):
    ensure_snapshot(tmp_path, {"lr": 2e-5}, resume=True)
    ensure_snapshot(tmp_path, {"lr": 5e-5}, resume=False)
    assert "5e-05" in (tmp_path / "config.json").read_text()


# ------------------------------------------------------------- run + resume
@pytest.fixture(scope="module")
def cv_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("cv")
    results = run_cv(toy_data(), load_config("base.yaml"), folds=[0, 1], seeds=[7],
                     device=CPU, out_dir=out, model_builder=tiny_builder,
                     settings=fast_settings(), log=silent)
    return out, results


def test_run_cv_writes_one_result_and_prediction_file_per_run(cv_dir):
    out, results = cv_dir
    assert len(results) == 2
    for fold in (0, 1):
        assert run_path(out, fold, 7).exists()
        assert (out / "runs" / f"fold{fold}_seed7_pred.csv").exists()


def test_run_cv_resumes_without_retraining(cv_dir):
    out, first = cv_dir

    def must_not_train(cfg, n_kw):
        raise AssertionError("a completed run was retrained")

    again = run_cv(toy_data(), load_config("base.yaml"), folds=[0, 1], seeds=[7],
                   device=CPU, out_dir=out, model_builder=must_not_train,
                   settings=fast_settings(), log=silent)
    assert [r["fold"] for r in again] == [r["fold"] for r in first]


def test_load_predictions_reads_only_the_requested_runs(cv_dir):
    out, _ = cv_dir
    both = load_predictions(out, [(7, 0), (7, 1)])
    one = load_predictions(out, [(7, 0)])
    assert len(one) < len(both)
    assert set(one["fold"]) == {0}


def test_load_predictions_fails_when_nothing_matches(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(ValueError, match="no prediction files"):
        load_predictions(tmp_path, [(1, 0)])


def test_partial_results_still_aggregate(cv_dir):
    """An interrupted CV can still be summarised from whatever finished."""
    _, results = cv_dir
    s = summarise(runs_frame(results[:1]), max_epochs=2)
    assert s["n_runs"] == 1
    assert "narrative_macro_f1" in s["metrics"]
