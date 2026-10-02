"""Predictor: loading a saved model and turning articles into labels.

A tiny model plus a deterministic fake tokenizer, so no download and no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from osint_shield.config import load_config
from osint_shield.inference import Predictor
from osint_shield.training.final import save_model, train_final
from tests.test_training import fast_settings, silent, tiny_builder, toy_data

CPU = torch.device("cpu")


class FakeTokenizer:
    """Deterministic word -> id mapping with the attributes prepare_data reads."""

    sep_token = "</s>"
    pad_token_id = 0
    model_max_length = 64

    def __call__(self, texts, truncation=True, max_length=64, add_special_tokens=True):
        out = []
        for text in texts:
            ids = [5 + sum(map(ord, w)) % 50 for w in str(text).split()][: max_length - 2]
            out.append([1, *ids, 2])
        return {"input_ids": out}


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    # keyword fusion off on both sides: toy training features and the inference
    # features built from base.yaml would otherwise differ in width
    data = toy_data(n_kw=0)
    cfg = load_config("base.yaml")
    cfg["keywords"]["enabled"] = False
    trainer, _ = train_final(data, np.arange(50), cfg, seed=0, device=CPU,
                             model_builder=tiny_builder,
                             settings=fast_settings(epochs=1, early_stopping=False), log=silent)
    out = save_model(trainer.model, cfg, tmp_path_factory.mktemp("m"), seed=0, n_train=50,
                     keyword_feature_names=[f"k{i}" for i in range(11)])
    return Predictor.from_dir(out, device=CPU, tokenizer=FakeTokenizer(),
                              model_builder=tiny_builder)


@pytest.fixture
def articles():
    return pd.DataFrame({
        "clean_headline": ["DRDO conducts missile test", "Minister holds bilateral talks",
                           "Floods displace civilians"],
        "clean_text": ["Full body about the missile test.", None, "Rescue under way."],
    })


def test_predict_returns_a_label_and_confidence_per_task(predictor, articles):
    out = predictor.predict(articles)
    assert len(out) == 3
    for task in ("narrative", "severity", "propaganda"):
        assert task in out and f"{task}_confidence" in out
    assert set(out["severity"]) <= {"Low", "High"}
    assert set(out["narrative"]) <= {"Security", "Political", "Civilian", "Other", "Investigation"}
    assert out["narrative_confidence"].between(0, 1).all()


def test_probabilities_sum_to_one(predictor, articles):
    probs = predictor.predict_proba(articles)
    for task, p in probs.items():
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5), task


def test_prediction_is_deterministic(predictor, articles):
    a = predictor.predict_proba(articles)
    b = predictor.predict_proba(articles)
    for task in a:
        assert np.array_equal(a[task], b[task])


def test_headline_only_articles_need_no_body_column(predictor):
    """The live feed's Google-News items arrive as a headline and nothing else."""
    out = predictor.predict(pd.DataFrame({"clean_headline": ["Army chief visits border post"]}))
    assert len(out) == 1


def test_labels_are_never_read_at_inference(predictor, articles):
    """Placeholder label columns must not change predictions."""
    a = predictor.predict_proba(articles)
    b = predictor.predict_proba(articles.assign(y_narrative=4, y_severity=1))
    assert np.array_equal(a["narrative"], b["narrative"])


def test_empty_input_returns_empty(predictor):
    assert predictor.predict_proba(pd.DataFrame({"clean_headline": []})) == {}
