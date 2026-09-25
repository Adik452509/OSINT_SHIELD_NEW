"""Data layer: loading, duplicate detection, fold construction.

Tests that need the corpus are marked ``corpus`` and skip if it is absent, so
the suite still runs on a machine without the gitignored data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osint_shield.data.dedupe import (
    add_duplicate_columns,
    components_from_similarity,
    duplicate_summary,
    normalise_headline,
    similarity_groups,
    strip_source_suffix,
)
from osint_shield.data.loaders import apply_data_mode, build_text, load_gold
from osint_shield.data.splits import (
    PROPAGANDA_STRATUM,
    assign_folds,
    check_no_group_spans_folds,
    resolve_holdout,
    stratification_labels,
)
from osint_shield.paths import GOLD_CSV

corpus = pytest.mark.skipif(not GOLD_CSV.exists(), reason="corpus not present")


# ----------------------------------------------------------------- normalise
def test_strip_source_suffix():
    assert strip_source_suffix("IAF chief visits France - News On AIR") == "IAF chief visits France"
    assert strip_source_suffix("Navy drill | The Hindu") == "Navy drill"


def test_strip_source_suffix_leaves_plain_headlines():
    plain = "DRDO conducts salvo launch of two Pralay missiles"
    assert strip_source_suffix(plain) == plain


def test_normalise_headline_lowercases_and_strips_punctuation():
    assert normalise_headline("IAF Chief visits France! - News18") == "iaf chief visits france"


def test_normalise_headline_collapses_whitespace():
    assert normalise_headline("Navy    drill\tbegins") == "navy drill begins"


# ---------------------------------------------------------------- similarity
def test_identical_headlines_group_together():
    groups = similarity_groups(["Airmen fly with Indian air force counterparts"] * 2)
    assert groups[0] == groups[1]


def test_case_and_suffix_variants_group_together():
    groups = similarity_groups([
        "IAF chief visits France amid India's biggest Rafale expansion plan",
        "IAF Chief visits France amid India's biggest Rafale expansion plan - NDTV",
    ])
    assert groups[0] == groups[1]


def test_unrelated_headlines_stay_apart():
    groups = similarity_groups([
        "DRDO conducts salvo launch of two Pralay missiles",
        "Tourism board announces new film festival in Goa",
    ])
    assert groups[0] != groups[1]


def test_similarity_groups_handles_degenerate_input():
    assert len(similarity_groups([])) == 0
    assert list(similarity_groups(["only one"])) == [0]


def test_components_chain_through_an_intermediate():
    """A~B and B~C merges A with C even though the direct A-C edge is missing.

    Tested on an explicit matrix rather than real text, because TF-IDF weights
    depend on the whole set passed in - a 3-document call cannot reproduce a
    corpus-level similarity.
    """
    sim = np.array([
        [1.00, 0.90, 0.50],
        [0.90, 1.00, 0.90],
        [0.50, 0.90, 1.00],
    ])
    assert len(set(components_from_similarity(sim, 0.85))) == 1
    # raise the bar above every edge and they fall apart into singletons
    assert len(set(components_from_similarity(sim, 0.95))) == 3


def test_components_ignore_the_diagonal():
    """Self-similarity of 1.0 must not merge unrelated rows."""
    sim = np.eye(3)
    assert len(set(components_from_similarity(sim, 0.85))) == 3


def test_add_duplicate_columns_flags_only_repeats():
    df = pd.DataFrame({"clean_headline": ["Same story", "Same story", "Other story"]})
    out = add_duplicate_columns(df)
    assert list(out["is_exact_dup"]) == [False, True, False]
    assert out.loc[0, "exact_group"] == out.loc[1, "exact_group"]


def test_duplicate_summary_counts_redundancy():
    df = pd.DataFrame({"clean_headline": ["A story here", "A story here", "Totally other text"]})
    s = duplicate_summary(add_duplicate_columns(df))
    assert s["n_rows"] == 3
    assert s["n_exact_dups"] == 1
    assert s["n_redundant_rows"] == s["n_rows"] - s["n_groups"]


# -------------------------------------------------------------------- splits
def _toy(n=60, seed=0):
    """Synthetic corpus: adjacent rows share a duplicate group, last rows are held out."""
    rng = np.random.default_rng(seed)
    n_test = min(10, n // 4)
    df = pd.DataFrame({
        "id": [f"a{i}" for i in range(n)],
        "narrative": rng.choice(["Security", "Political", "Civilian"], n, p=[0.7, 0.2, 0.1]),
        "severity": rng.choice(["Low", "High"], n),
        "y_propaganda": 0,
        "has_body": True,
        "dupe_group": np.arange(n) // 2,      # pairs share a group
        "split_orig": ["train"] * (n - n_test) + ["test"] * n_test,
    })
    df.loc[[i for i in (0, 7, 19, 33, 41) if i < n], "y_propaganda"] = 1
    return df


def test_stratification_moves_rare_positives_to_own_stratum():
    df = _toy()
    labels = stratification_labels(df)
    assert (labels[df["y_propaganda"] == 1] == PROPAGANDA_STRATUM).all()
    assert PROPAGANDA_STRATUM not in labels[df["y_propaganda"] == 0]


def test_resolve_holdout_separates_test_split():
    pool, holdout, _leaked = resolve_holdout(_toy(), exclude_leaks=False)
    assert set(holdout["split_orig"]) == {"test"}
    assert "test" not in set(pool["split_orig"])
    assert len(pool) + len(holdout) == 60


def test_resolve_holdout_removes_rows_duplicating_a_holdout_article():
    df = _toy()
    # force a train row into the same duplicate group as a held-out row
    df.loc[0, "dupe_group"] = df.loc[55, "dupe_group"]
    pool, holdout, leaked = resolve_holdout(df, exclude_leaks=True)
    assert len(leaked) >= 1
    assert "a0" in set(leaked["id"])
    assert not set(pool["dupe_group"]) & set(holdout["dupe_group"])


def test_resolve_holdout_accounts_for_every_row():
    """pool + holdout + leaked must reconstruct the input exactly."""
    df = _toy()
    df.loc[0, "dupe_group"] = df.loc[55, "dupe_group"]
    pool, holdout, leaked = resolve_holdout(df, exclude_leaks=True)
    assert len(pool) + len(holdout) + len(leaked) == len(df)
    assert set(pool["id"]) | set(holdout["id"]) | set(leaked["id"]) == set(df["id"])


def test_resolve_holdout_reports_no_leaks_when_disabled():
    pool, holdout, leaked = resolve_holdout(_toy(), exclude_leaks=False)
    assert len(leaked) == 0
    assert len(pool) + len(holdout) == 60


def test_assign_folds_covers_every_row_exactly_once():
    df = _toy()
    folds = assign_folds(df, n_folds=5, seed=42)
    assert len(folds) == len(df)
    assert set(folds) == {0, 1, 2, 3, 4}


def test_no_duplicate_group_spans_two_folds():
    """The property the whole module exists to guarantee."""
    df = _toy()
    df["fold"] = assign_folds(df, n_folds=5, seed=42)
    assert check_no_group_spans_folds(df) == []


def test_assign_folds_is_deterministic_for_a_seed():
    df = _toy()
    assert (assign_folds(df, seed=42) == assign_folds(df, seed=42)).all()


def test_assign_folds_rejects_too_few_rows():
    with pytest.raises(ValueError, match="cannot build"):
        assign_folds(_toy(n=3), n_folds=5)


# -------------------------------------------------------------------- loaders
def test_apply_data_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown data mode"):
        apply_data_mode(pd.DataFrame({"has_body": [True]}), "everything")


def test_build_text_marks_headline_only_rows():
    df = pd.DataFrame({
        "clean_headline": ["Missile test conducted", "Navy drill begins"],
        "clean_text": ["Full body text here.", None],
        "has_body": [True, False],
    })
    texts = build_text(df, sep="</s>")
    assert texts[0] == "Missile test conducted </s> Full body text here."
    assert texts[1] == "[NO_BODY] Navy drill begins"


def test_build_text_uses_the_models_own_separator():
    df = pd.DataFrame({"clean_headline": ["H"], "clean_text": ["B"], "has_body": [True]})
    assert "<eos>" in build_text(df, sep="<eos>")[0]


# ------------------------------------------------------- corpus-backed checks
@corpus
def test_corpus_loads_with_expected_shape():
    df = load_gold()
    assert len(df) == 1064
    assert df["id"].is_unique
    assert int(df["y_propaganda"].sum()) == 7


@corpus
def test_corpus_narrative_counts():
    counts = load_gold()["narrative"].value_counts().to_dict()
    assert counts == {"Security": 824, "Political": 155, "Civilian": 41,
                      "Other": 26, "Investigation": 18}


@corpus
def test_corpus_label_ids_are_contiguous_and_radicalization_absent():
    df = load_gold()
    assert sorted(df["y_narrative"].unique()) == [0, 1, 2, 3, 4]
    assert "Radicalization" not in set(df["narrative"])


@corpus
def test_corpus_has_body_matches_measured_share():
    df = load_gold()
    assert int(df["has_body"].sum()) == 574          # 1064 - 490 empty bodies
    assert int((~df["has_body"]).sum()) == 490


@corpus
def test_corpus_splits_are_disjoint_and_complete():
    df = load_gold()
    assert df["split_orig"].value_counts().to_dict() == {"train": 744, "val": 160, "test": 160}


@corpus
def test_corpus_exact_duplicate_count():
    df = add_duplicate_columns(load_gold())
    assert int(df["is_exact_dup"].sum()) == 11


@corpus
def test_corpus_largest_cluster_is_the_pralay_story():
    """The biggest real leak: six near-identical articles about one missile test.

    Several of its pairs score below the 0.85 threshold and only group by
    chaining - so this asserts transitivity on real data, which a synthetic
    matrix cannot.
    """
    df = add_duplicate_columns(load_gold())
    sizes = df["dupe_group"].value_counts()
    assert sizes.max() == 6
    biggest = df[df["dupe_group"] == sizes.idxmax()]
    assert biggest["clean_headline"].str.contains("Pralay", case=False).all()


@corpus
def test_corpus_near_dup_clusters_match_analysis():
    summary = duplicate_summary(add_duplicate_columns(load_gold()))
    assert summary["n_groups"] == 1038
    assert summary["n_multi_row_groups"] == 22
    assert summary["n_redundant_rows"] == 1064 - 1038


@corpus
def test_corpus_body_only_mode_count():
    """571 after exact dedupe - the number the measured baselines used."""
    df = add_duplicate_columns(load_gold())
    df = df[~df["is_exact_dup"]]
    assert len(apply_data_mode(df, "body_only")) == 571


@corpus
def test_corpus_folds_have_no_leakage_and_spread_propaganda():
    df = add_duplicate_columns(load_gold())
    df = df[~df["is_exact_dup"]].reset_index(drop=True)
    pool, holdout, _leaked = resolve_holdout(df, exclude_leaks=True)
    pool = apply_data_mode(pool, "body_only").reset_index(drop=True)
    pool["fold"] = assign_folds(pool, n_folds=5, seed=42)

    assert check_no_group_spans_folds(pool) == []
    assert not set(pool["id"]) & set(holdout["id"])
    per_fold = pool.groupby("fold")["y_propaganda"].sum()
    assert per_fold.sum() == int(pool["y_propaganda"].sum())
