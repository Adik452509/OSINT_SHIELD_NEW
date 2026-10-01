"""Training loop, datasets, early stopping and fold orchestration.

End-to-end tests drive a tiny randomly initialised BERT on CPU, so the full
fold pipeline - inner split, class weights, keyword stats, training, test
prediction - is exercised without a download or a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from osint_shield.config import load_config
from osint_shield.models import MultiTaskLoss, MultiTaskModel, class_weights
from osint_shield.training import ArticleDataset, EarlyStopping, Trainer, TrainSettings, make_collate
from osint_shield.training.fold import (
    PreparedData,
    assert_disjoint,
    inner_split,
    run_fold,
    run_overfit_check,
    select_overfit_rows,
)
from osint_shield.training.trainer import linear_warmup_decay, monitor_score, param_groups

CPU = torch.device("cpu")
HEADS = {"narrative": 5, "severity": 2, "propaganda": 2}
NARRATIVES = ["Security", "Political", "Civilian", "Other", "Investigation"]  # taxonomy ids 0-4


def silent(*_args, **_kwargs):
    return None


def tiny_encoder(seed: int = 0):
    from transformers import BertConfig, BertModel

    torch.manual_seed(seed)
    return BertModel(BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32,
                                max_position_embeddings=64))


def tiny_builder(cfg, n_keyword_features):
    return MultiTaskModel(tiny_encoder(), 16, HEADS, n_keyword_features=n_keyword_features)


def toy_data(n: int = 60, n_kw: int = 3, seed: int = 0) -> PreparedData:
    """60 articles in 5 folds; every 5-row block holds one of each narrative class."""
    rng = np.random.default_rng(seed)
    y_narr = np.arange(n) % 5
    frame = pd.DataFrame({
        "id": [f"a{i}" for i in range(n)],
        "fold": (np.arange(n) // 5) % 5,
        "dupe_group": np.arange(n),
        "narrative": [NARRATIVES[i] for i in y_narr],
        "y_narrative": y_narr,
        "y_severity": rng.integers(0, 2, n),
        "y_propaganda": 0,
        "has_body": True,
    })
    frame.loc[[3, 17, 29, 41, 58], "y_propaganda"] = 1
    ids = [list(rng.integers(5, 64, int(rng.integers(6, 12)))) for _ in range(n)]
    kw = rng.random((n, n_kw)).astype(np.float32)
    return PreparedData(frame, ids, np.zeros(n, dtype=bool), kw, pad_id=0, max_length=12)


def fast_settings(**overrides) -> TrainSettings:
    base = dict(epochs=2, patience=1, min_epochs=0, batch_size=8, grad_accum=1,
                amp=False, lr=1e-3, warmup_ratio=0.0)
    base.update(overrides)
    return TrainSettings(**base)


# ------------------------------------------------------------------ collation
def test_collate_pads_to_the_longest_item_in_the_batch():
    collate = make_collate(pad_id=0)
    batch = collate([
        {"input_ids": [5, 6, 7], "targets": {"narrative": 1}, "kw": None},
        {"input_ids": [8], "targets": {"narrative": 2}, "kw": None},
    ])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][1].tolist() == [8, 0, 0]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["targets"]["narrative"].tolist() == [1, 2]
    assert "keyword_feats" not in batch


def test_collate_stacks_keyword_features():
    collate = make_collate(pad_id=0)
    batch = collate([
        {"input_ids": [5], "targets": {"narrative": 0}, "kw": np.array([1.0, 2.0])},
        {"input_ids": [6], "targets": {"narrative": 1}, "kw": np.array([3.0, 4.0])},
    ])
    assert batch["keyword_feats"].shape == (2, 2)
    assert batch["keyword_feats"].dtype == torch.float32


def test_collate_requires_a_pad_id():
    with pytest.raises(ValueError, match="no pad token"):
        make_collate(None)


def test_dataset_rejects_misaligned_targets():
    with pytest.raises(ValueError, match="expected 2"):
        ArticleDataset([[1], [2]], {"narrative": np.array([0, 1, 2])})


# ------------------------------------------------------------- early stopping
def test_early_stopping_waits_for_patience():
    stop = EarlyStopping(patience=2)
    stop.update(0, 0.5)
    stop.update(1, 0.4)
    assert not stop.should_stop
    stop.update(2, 0.4)
    assert stop.should_stop
    assert stop.best_epoch == 0


def test_early_stopping_resets_on_improvement():
    stop = EarlyStopping(patience=2)
    for epoch, score in enumerate([0.5, 0.4, 0.6, 0.55]):
        stop.update(epoch, score)
    assert stop.best == pytest.approx(0.6)
    assert stop.bad_epochs == 1
    assert not stop.should_stop


def test_early_stopping_respects_min_epochs():
    """Flat majority-class F1 in the first epochs must not end the run."""
    stop = EarlyStopping(patience=1, min_epochs=3)
    stop.update(0, 0.17)
    stop.update(1, 0.17)
    assert not stop.should_stop          # 2 epochs seen, floor is 3
    stop.update(2, 0.17)
    assert stop.should_stop


# ------------------------------------------------------------------ schedules
def test_linear_warmup_then_decay():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = linear_warmup_decay(opt, warmup_steps=2, total_steps=10)
    lrs = []
    for _ in range(10):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.5)      # warming up
    assert lrs[2] == pytest.approx(1.0)      # peak
    assert lrs[-1] < lrs[3]                  # decaying
    assert min(lrs) >= 0.0


def test_param_groups_exclude_bias_and_norm_from_decay():
    model = MultiTaskModel(tiny_encoder(), 16, {"narrative": 5})
    decay, no_decay = param_groups(model, 0.01)
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    decayed = {id(p) for p in decay["params"]}
    for name, p in model.named_parameters():
        if name.endswith(".bias") or "norm" in name.lower():
            assert id(p) not in decayed, name


# ------------------------------------------------------------------ settings
def test_settings_read_from_config():
    s = TrainSettings.from_config(load_config("base.yaml"))
    assert s.epochs == 10
    assert s.amp_dtype == "bf16"
    assert isinstance(s.lr, float)


def test_settings_overrides_ignore_none():
    s = TrainSettings.from_config(load_config("base.yaml"), epochs=None, lr=1e-4)
    assert s.epochs == 10
    assert s.lr == pytest.approx(1e-4)


def test_settings_reject_unknown_amp_dtype():
    with pytest.raises(ValueError, match="amp_dtype"):
        TrainSettings.from_config(load_config("base.yaml"), amp_dtype="fp8")


def test_monitor_score():
    m = {"narrative_macro_f1": 0.4, "severity_f1_high": 0.6}
    assert monitor_score(m, "narrative_macro_f1") == pytest.approx(0.4)
    assert monitor_score(m, "combined") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        monitor_score(m, "val_loss")


# ---------------------------------------------------------------- inner split
def test_inner_split_keeps_duplicate_groups_together():
    rng = np.random.default_rng(0)
    labels = rng.choice(["a", "b", "c"], 80)
    groups = np.arange(80) // 2               # pairs
    tr, va = inner_split(labels, groups, 0.12, seed=0)
    assert not set(groups[tr]) & set(groups[va])
    assert len(tr) + len(va) == 80


def test_assert_disjoint_catches_a_shared_duplicate_group():
    frame = pd.DataFrame({"dupe_group": [0, 0, 1]})
    with pytest.raises(AssertionError, match="duplicate group"):
        assert_disjoint(frame, np.array([0]), np.array([1]))


def test_assert_disjoint_catches_shared_rows():
    frame = pd.DataFrame({"dupe_group": [0, 1, 2]})
    with pytest.raises(AssertionError, match="share rows"):
        assert_disjoint(frame, np.array([0, 1]), np.array([1]))


# ------------------------------------------------------------------- trainer
def test_trainer_reduces_loss_on_a_tiny_problem():
    rng = np.random.default_rng(0)
    ids = [list(rng.integers(5, 64, 8)) for _ in range(8)]
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    model = MultiTaskModel(tiny_encoder(), 16, {"narrative": 5})
    loss = MultiTaskLoss({"narrative": torch.ones(5)}, {"narrative": 1.0})
    trainer = Trainer(model, loss, fast_settings(epochs=25, batch_size=4, lr=5e-3),
                      CPU, pad_id=0, log=silent)
    history = trainer.fit(ArticleDataset(ids, {"narrative": y}))
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_trainer_predict_proba_shapes_and_normalisation():
    data = toy_data()
    model = tiny_builder(None, data.n_keyword_features)
    loss = MultiTaskLoss({t: torch.ones(n) for t, n in HEADS.items()},
                         {"narrative": 1.0, "severity": 1.0, "propaganda": 0.3})
    trainer = Trainer(model, loss, fast_settings(), CPU, pad_id=0, log=silent)
    probs = trainer.predict_proba(data.dataset(np.arange(10)))
    assert probs["narrative"].shape == (10, 5)
    assert np.allclose(probs["narrative"].sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------- fold
@pytest.fixture(scope="module")
def fold_result():
    return run_fold(toy_data(), 0, load_config("base.yaml"), seed=0, device=CPU,
                    model_builder=tiny_builder, settings=fast_settings(), log=silent)


def test_fold_test_set_is_exactly_the_requested_fold(fold_result):
    data = toy_data()
    expected = set(data.frame.loc[data.frame["fold"] == 0, "id"])
    assert set(fold_result.test_ids) == expected
    assert fold_result.n_test == len(expected)


def test_fold_train_val_test_are_disjoint(fold_result):
    train, val, test = map(set, (fold_result.train_ids, fold_result.val_ids, fold_result.test_ids))
    assert not train & val and not train & test and not val & test
    assert len(train) + len(val) + len(test) == 60


def test_fold_class_weights_come_from_the_training_rows_only(fold_result):
    """The leak this guards: weights computed on the full corpus see the test fold."""
    frame = toy_data().frame.set_index("id")
    train_y = frame.loc[fold_result.train_ids, "y_narrative"].to_numpy()
    expected = class_weights(train_y, 5).tolist()
    assert fold_result.class_weights["narrative"] == pytest.approx(expected, abs=1e-4)


def test_fold_propaganda_weight_respects_the_cap(fold_result):
    assert fold_result.class_weights["propaganda"][1] <= 10.0 + 1e-6


def test_fold_reports_every_metric_and_prediction(fold_result):
    m = fold_result.metrics
    for key in ("narrative_macro_f1", "narrative_collapsed_macro_f1", "severity_f1_high",
                "propaganda_f1_pos", "propaganda_n_pos", "severity_by_has_body"):
        assert key in m
    preds = fold_result.predictions
    assert len(preds) == fold_result.n_test
    assert set(preds["pred_narrative"]) <= set(NARRATIVES)
    assert preds["conf_narrative"].between(0, 1).all()


def test_fold_summary_is_json_serialisable(fold_result):
    import json

    json.dumps(fold_result.summary(), default=str)


# ------------------------------------------------------------------- overfit
def test_select_overfit_rows_covers_every_class():
    data = toy_data()
    idx = select_overfit_rows(data.frame, 20, seed=0)
    assert len(idx) == 20
    assert len(np.unique(idx)) == 20
    assert set(data.frame["y_narrative"].to_numpy()[idx]) == {0, 1, 2, 3, 4}


def test_overfit_check_drives_the_loss_down():
    res = run_overfit_check(toy_data(), 20, load_config("base.yaml"), seed=0, device=CPU,
                            epochs=30, lr=5e-3, model_builder=tiny_builder, log=silent)
    assert res["n_rows"] == 20
    assert res["last_loss"] < res["first_loss"]
    assert "narrative_macro_f1" in res["metrics"]
