"""Final-model training, saving and reloading."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from osint_shield.config import load_config
from osint_shield.training.final import MODEL_FILE, META_FILE, load_model, save_model, train_final
from tests.test_training import fast_settings, silent, tiny_builder, toy_data

CPU = torch.device("cpu")


def test_final_training_refuses_early_stopping():
    """There is no validation set to stop on - by design (D13)."""
    data = toy_data()
    with pytest.raises(ValueError, match="fixed schedule"):
        train_final(data, np.arange(50), load_config("base.yaml"), seed=0, device=CPU,
                    model_builder=tiny_builder, settings=fast_settings(early_stopping=True),
                    log=silent)


def test_final_training_runs_the_full_schedule():
    data = toy_data()
    _trainer, history = train_final(data, np.arange(50), load_config("base.yaml"), seed=0,
                                    device=CPU, model_builder=tiny_builder,
                                    settings=fast_settings(epochs=3, early_stopping=False),
                                    log=silent)
    assert len(history) == 3


def test_saved_model_reloads_and_predicts_identically(tmp_path):
    data = toy_data()
    trainer, _ = train_final(data, np.arange(50), load_config("base.yaml"), seed=0,
                             device=CPU, model_builder=tiny_builder,
                             settings=fast_settings(epochs=2, early_stopping=False), log=silent)
    before = trainer.predict_proba(data.dataset(np.arange(50, 60)))

    out = save_model(trainer.model, load_config("base.yaml"), tmp_path / "m", seed=0,
                     n_train=50, keyword_feature_names=["a", "b", "c"])
    assert (out / MODEL_FILE).exists() and (out / META_FILE).exists()

    model, meta = load_model(out, model_builder=tiny_builder)
    trainer.model = model
    after = trainer.predict_proba(data.dataset(np.arange(50, 60)))
    for task in before:
        assert np.allclose(before[task], after[task], atol=1e-6), task
    assert meta["n_keyword_features"] == 3
    assert meta["labels"]["narrative"]["0"] == "Security"


def test_saved_model_keeps_keyword_standardisation(tmp_path):
    """The train-fold keyword stats must survive the round trip, or inference drifts."""
    data = toy_data()
    trainer, _ = train_final(data, np.arange(50), load_config("base.yaml"), seed=0,
                             device=CPU, model_builder=tiny_builder,
                             settings=fast_settings(epochs=1, early_stopping=False), log=silent)
    out = save_model(trainer.model, load_config("base.yaml"), tmp_path / "m", seed=0,
                     n_train=50, keyword_feature_names=["a", "b", "c"])
    model, _ = load_model(out, model_builder=tiny_builder)
    assert torch.allclose(model.kw_mean, trainer.model.kw_mean)
    assert torch.allclose(model.kw_std, trainer.model.kw_std)
