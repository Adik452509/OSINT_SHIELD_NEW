"""Multi-task model, heads and losses.

Uses a tiny randomly initialised BERT so nothing is downloaded and the suite
runs on CPU in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from osint_shield.config import load_config
from osint_shield.models import (
    ClassificationHead,
    MultiTaskLoss,
    MultiTaskModel,
    class_weights,
    freeze_input_embeddings,
    pool,
)

HEADS = {"narrative": 5, "severity": 2, "propaganda": 2}
HIDDEN = 16


def tiny_encoder(seed: int = 0):
    from transformers import BertConfig, BertModel

    torch.manual_seed(seed)
    return BertModel(BertConfig(vocab_size=64, hidden_size=HIDDEN, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32,
                                max_position_embeddings=64))


def toy_batch(n: int = 4, length: int = 10):
    torch.manual_seed(1)
    ids = torch.randint(5, 64, (n, length))
    mask = torch.ones_like(ids)
    mask[0, 6:] = 0                       # one padded row
    return ids, mask


# -------------------------------------------------------------- class weights
def test_class_weights_depend_only_on_the_labels_passed():
    """Two different training folds must get two different weightings."""
    fold_a = class_weights([0, 0, 0, 1], 2)
    fold_b = class_weights([0, 1, 1, 1], 2)
    assert not torch.allclose(fold_a, fold_b)


def test_class_weights_are_balanced():
    w = class_weights([0, 0, 0, 1], 2)
    # sklearn 'balanced': n / (n_classes * count)
    assert w[0] == pytest.approx(4 / (2 * 3))
    assert w[1] == pytest.approx(4 / (2 * 1))


def test_absent_class_gets_weight_one():
    w = class_weights([0, 0, 1, 1], 5)
    assert w[2:].tolist() == [1.0, 1.0, 1.0]


def test_propaganda_weight_is_capped():
    """~4 positives in ~340 rows: balanced gives ~42x; the cap must hold it at 10."""
    y = np.array([1] * 4 + [0] * 336)
    uncapped = class_weights(y, 2)
    capped = class_weights(y, 2, cap=10.0)
    assert uncapped[1] > 40
    assert capped[1] == pytest.approx(10.0)
    assert capped[0] == pytest.approx(uncapped[0])


def test_class_weights_reject_bad_input():
    with pytest.raises(ValueError, match="empty"):
        class_weights([], 2)
    with pytest.raises(ValueError, match="must lie in"):
        class_weights([0, 5], 2)


# --------------------------------------------------------------------- loss
def test_multitask_loss_is_the_weighted_sum_of_its_parts():
    logits = {"narrative": torch.randn(4, 5), "severity": torch.randn(4, 2)}
    targets = {"narrative": torch.tensor([0, 1, 2, 3]), "severity": torch.tensor([0, 1, 0, 1])}
    loss_fn = MultiTaskLoss({"narrative": torch.ones(5), "severity": torch.ones(2)},
                            {"narrative": 1.0, "severity": 0.3})
    total, parts = loss_fn(logits, targets)
    assert total.item() == pytest.approx(parts["narrative"].item() + 0.3 * parts["severity"].item(),
                                         rel=1e-5)


def test_multitask_loss_requires_weights_for_every_task():
    with pytest.raises(ValueError, match="no class weights"):
        MultiTaskLoss({"narrative": torch.ones(5)}, {"narrative": 1.0, "severity": 1.0})


# ------------------------------------------------------------------- pooling
def test_cls_pooling_takes_the_first_position():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(pool(hidden, torch.ones(2, 3), "cls"), hidden[:, 0])


def test_mean_pooling_ignores_padding():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])     # last position is padding
    mask = torch.tensor([[1, 1, 0]])
    assert pool(hidden, mask, "mean").item() == pytest.approx(2.0)


def test_unknown_pooling_raises():
    with pytest.raises(ValueError, match="unknown pooling"):
        pool(torch.zeros(1, 2, 3), torch.ones(1, 2), "max")


# --------------------------------------------------------------------- heads
def test_head_rejects_fewer_than_two_classes():
    with pytest.raises(ValueError, match="at least 2"):
        ClassificationHead(16, 1)


# --------------------------------------------------------------------- model
def test_forward_returns_one_logit_tensor_per_head():
    model = MultiTaskModel(tiny_encoder(), HIDDEN, HEADS).eval()
    ids, mask = toy_batch()
    out = model(ids, mask)
    assert set(out) == set(HEADS)
    assert out["narrative"].shape == (4, 5)
    assert out["severity"].shape == (4, 2)


def test_only_enabled_heads_are_built():
    model = MultiTaskModel(tiny_encoder(), HIDDEN, {"narrative": 5})
    assert model.tasks == ["narrative"]


def test_model_rejects_unknown_or_missing_heads():
    with pytest.raises(ValueError, match="unknown task"):
        MultiTaskModel(tiny_encoder(), HIDDEN, {"sentiment": 3})
    with pytest.raises(ValueError, match="at least one head"):
        MultiTaskModel(tiny_encoder(), HIDDEN, {})


def test_keyword_fusion_widens_the_head_input():
    model = MultiTaskModel(tiny_encoder(), HIDDEN, HEADS, n_keyword_features=11)
    assert model.heads["narrative"].linear.in_features == HIDDEN + 11


def test_keyword_fusion_requires_features():
    model = MultiTaskModel(tiny_encoder(), HIDDEN, HEADS, n_keyword_features=3).eval()
    ids, mask = toy_batch()
    with pytest.raises(ValueError, match="no keyword_feats"):
        model(ids, mask)


def test_keyword_features_actually_reach_the_heads():
    """If fusion is wired, changing only the keyword vector changes the logits."""
    model = MultiTaskModel(tiny_encoder(), HIDDEN, HEADS, n_keyword_features=3).eval()
    ids, mask = toy_batch()
    with torch.no_grad():
        a = model(ids, mask, torch.zeros(4, 3))["narrative"]
        b = model(ids, mask, torch.full((4, 3), 5.0))["narrative"]
    assert not torch.allclose(a, b)


def test_keyword_stats_handle_zero_variance():
    """A rubric group that never fires in a fold has std 0 - must not divide by it."""
    model = MultiTaskModel(tiny_encoder(), HIDDEN, HEADS, n_keyword_features=3)
    model.set_keyword_stats([0.5, 0.0, 1.0], [0.2, 0.0, 0.5])
    assert model.kw_std.tolist() == pytest.approx([0.2, 1.0, 0.5])
    ids, mask = toy_batch()
    out = model.eval()(ids, mask, torch.zeros(4, 3))
    assert torch.isfinite(out["narrative"]).all()


def test_keyword_stats_reject_wrong_shape():
    model = MultiTaskModel(tiny_encoder(), HIDDEN, HEADS, n_keyword_features=3)
    with pytest.raises(ValueError, match="expected keyword stats"):
        model.set_keyword_stats([0.0, 0.0], [1.0, 1.0])


def test_freezing_embeddings_stops_only_their_gradient():
    enc = tiny_encoder()
    n_frozen = freeze_input_embeddings(enc)
    assert n_frozen == 64 * HIDDEN                          # vocab x hidden
    assert not enc.get_input_embeddings().weight.requires_grad
    body = [p for name, p in enc.named_parameters() if "encoder.layer" in name]
    assert body and all(p.requires_grad for p in body)      # transformer body still learns


def test_from_config_honours_freeze_embeddings():
    cfg = load_config("base.yaml")
    assert cfg["model"]["freeze_embeddings"] is True
    frozen = MultiTaskModel.from_config(cfg, 0, encoder=tiny_encoder())
    assert not frozen.encoder.get_input_embeddings().weight.requires_grad

    cfg["model"]["freeze_embeddings"] = False
    trainable = MultiTaskModel.from_config(cfg, 0, encoder=tiny_encoder())
    assert trainable.encoder.get_input_embeddings().weight.requires_grad


def test_frozen_embeddings_never_reach_the_optimiser():
    """AdamW state for frozen rows is the memory the freeze exists to save."""
    from osint_shield.training.trainer import param_groups

    model = MultiTaskModel.from_config(load_config("base.yaml"), 0, encoder=tiny_encoder())
    in_optimiser = {id(p) for group in param_groups(model, 0.01) for p in group["params"]}
    assert id(model.encoder.get_input_embeddings().weight) not in in_optimiser


def test_frozen_model_still_trains_end_to_end():
    model = MultiTaskModel.from_config(load_config("base.yaml"), 0, encoder=tiny_encoder())
    ids, mask = toy_batch()
    loss = model(ids, mask)["narrative"].sum()
    loss.backward()
    assert model.encoder.get_input_embeddings().weight.grad is None
    assert model.heads["narrative"].linear.weight.grad is not None


def test_base_config_uses_memory_efficient_attention():
    assert load_config("base.yaml")["model"]["attn_implementation"] == "sdpa"


def test_keyword_stats_travel_with_the_saved_model():
    """Standardisation buffers must be in the state dict, or a reload breaks fusion."""
    model = MultiTaskModel(tiny_encoder(), HIDDEN, HEADS, n_keyword_features=3)
    model.set_keyword_stats([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    state = model.state_dict()
    assert state["kw_mean"].tolist() == [1.0, 2.0, 3.0]
    assert "kw_std" in state
