"""Config loading, inheritance and the label space."""

from __future__ import annotations

import pytest

from osint_shield.config import (
    apply_overrides,
    collapse_map,
    deep_merge,
    load_config,
    load_taxonomy,
    narrative_label_maps,
    severity_label_maps,
)
from osint_shield.paths import GOLD_CSV, KEYWORDS_YAML, ROOT, TAXONOMY_YAML


def test_repo_root_looks_right():
    assert (ROOT / "pyproject.toml").exists()
    assert (ROOT / "configs" / "base.yaml").exists()


def test_resources_present():
    assert KEYWORDS_YAML.exists()
    assert TAXONOMY_YAML.exists()


def test_gold_csv_placed():
    """The corpus is gitignored but must be present locally to train."""
    assert GOLD_CSV.exists(), "run scripts/00_setup_data.py or unzip data/raw/osint_dataset.zip"


def test_deep_merge_is_recursive_and_non_mutating():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
    over = {"a": {"y": 99}, "b": [7]}
    out = deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": 99}, "b": [7], "c": 3}
    assert base["a"]["y"] == 2, "deep_merge must not mutate its inputs"


def test_base_config_loads():
    cfg = load_config("base.yaml")
    assert cfg["split"]["n_folds"] == 5
    assert cfg["data"]["mode"] == "body_only"


@pytest.mark.parametrize(
    ("name", "model", "batch", "epochs"),
    [
        # mmBERT overrides epochs to its own curve's peak (D15)
        ("mmbert_small.yaml", "jhu-clsp/mmBERT-small", 8, 6),
        # XLM-R inherits base.yaml's 10 - more bought nothing in M6
        ("xlmr_base.yaml", "FacebookAI/xlm-roberta-base", 4, 10),
    ],
)
def test_model_configs_inherit_and_override(name, model, batch, epochs):
    cfg = load_config(name)
    assert cfg["model"]["name"] == model
    assert cfg["training"]["batch_size"] == batch
    assert cfg["training"]["epochs"] == epochs
    # inherited from base.yaml, untouched by either override
    assert cfg["training"]["lr"] == pytest.approx(2e-5)
    assert cfg["split"]["n_folds"] == 5
    assert "extends" not in cfg


def test_final_model_recipe_matches_d15():
    """The configuration M7 trains must be exactly the one D15 records."""
    cfg = load_config("mmbert_small.yaml")
    assert cfg["model"]["pooling"] == "mean"
    assert cfg["model"]["freeze_embeddings"] is True
    assert cfg["model"]["attn_implementation"] == "sdpa"
    assert cfg["tokenizer"]["max_length"] == 512
    assert cfg["keywords"]["mode"] == "group_counts"
    assert cfg["training"]["early_stopping"] is False
    assert cfg["training"]["epochs"] == 6
    assert cfg["data"]["mode"] == "body_only"


def test_overrides_parse_typed_values():
    cfg = load_config("base.yaml")
    out = apply_overrides(cfg, ["training.early_stopping=false",
                                "tokenizer.max_length=1024",
                                "model.pooling=mean"])
    assert out["training"]["early_stopping"] is False
    assert out["tokenizer"]["max_length"] == 1024
    assert out["model"]["pooling"] == "mean"
    assert cfg["tokenizer"]["max_length"] == 512, "overrides must not mutate the input"


@pytest.mark.parametrize("raw", ["5e-5", "5.0e-5", "5E-05"])
def test_override_reads_scientific_notation_as_float(raw):
    """Plain YAML would hand back the string '5e-5'."""
    out = apply_overrides(load_config("base.yaml"), [f"training.lr={raw}"])
    assert out["training"]["lr"] == pytest.approx(5e-5)
    assert isinstance(out["training"]["lr"], float)


def test_override_with_empty_value_is_an_empty_string_not_null():
    out = apply_overrides(load_config("base.yaml"), ["data.no_body_marker="])
    assert out["data"]["no_body_marker"] == ""


def test_override_typo_is_an_error_not_a_silent_no_op():
    """A misspelt key would otherwise run the reference config under an ablation's name."""
    with pytest.raises(KeyError, match="early_stoping"):
        apply_overrides(load_config("base.yaml"), ["training.early_stoping=false"])


def test_override_cannot_descend_into_a_scalar():
    with pytest.raises(KeyError):
        apply_overrides(load_config("base.yaml"), ["seed.value=3"])


def test_override_must_have_an_equals_sign():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(load_config("base.yaml"), ["training.epochs"])


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("no_such_config.yaml")


def test_narrative_ids_are_contiguous():
    """The corpus's own narrative_label column skips id 3; ours must not."""
    name_to_id, id_to_name = narrative_label_maps()
    assert sorted(id_to_name) == [0, 1, 2, 3, 4]
    assert name_to_id["Security"] == 0
    assert "Radicalization" not in name_to_id


def test_radicalization_declared_but_not_trained():
    tax = load_taxonomy()
    assert tax["narrative"]["declared_but_absent"] == ["Radicalization"]


def test_severity_positive_class_is_high():
    name_to_id, _ = severity_label_maps()
    assert name_to_id == {"Low": 0, "High": 1}
    assert load_taxonomy()["severity"]["positive_class"] == "High"


def test_collapse_map_covers_every_class_and_yields_three():
    cmap = collapse_map()
    names = set(narrative_label_maps()[0])
    assert names <= set(cmap), "every trained class needs a collapse target"
    assert set(cmap.values()) == {"Security", "Political", "Other"}


def test_propaganda_is_trained_but_diagnostic():
    """Kept in the objective by explicit decision; never reported as a result."""
    cfg = load_config("base.yaml")
    prop = cfg["heads"]["propaganda"]
    assert prop["enabled"] is True
    assert prop["report_as"] == "diagnostic"
    assert prop["loss_weight"] < cfg["heads"]["narrative"]["loss_weight"]
    assert prop["class_weight_cap"] == 10.0


def test_recruitment_subtype_has_no_positives():
    subs = {s["name"]: s for s in load_taxonomy()["propaganda"]["subtypes"]}
    assert subs["recruitment"]["n"] == 0
    assert all(not s["trainable"] for s in subs.values())
