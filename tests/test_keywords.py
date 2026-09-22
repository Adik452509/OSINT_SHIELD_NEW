"""Rubric matching and fusion features."""

from __future__ import annotations

import numpy as np
import pytest

from osint_shield.keywords import KeywordMatcher, build_features, load_rubric


@pytest.fixture(scope="module")
def matcher() -> KeywordMatcher:
    return KeywordMatcher()


def test_rubric_loads_all_eleven_groups():
    r = load_rubric()
    assert len(r.group_names) == 11  # 2 severity + 4 propaganda + 5 narrative
    assert "narr_Security" in r.groups
    assert "sev_High" in r.groups
    assert "prop_recruitment" in r.groups


def test_radicalization_has_no_keyword_group():
    """Absent from the PDF and from the corpus - it must not appear here."""
    assert "narr_Radicalization" not in load_rubric().groups


def test_matches_are_case_insensitive(matcher):
    assert "narr_Security" in matcher.hits("drdo conducts a MISSILE TEST")


def test_hits_reports_the_matched_terms(matcher):
    hits = matcher.hits("DRDO conducts missile test near the border")
    assert set(hits["narr_Security"]) == {"DRDO", "missile test"}


def test_word_boundaries_prevent_substring_matches(matcher):
    """'aid' must not fire inside 'said', 'FIR' must not fire inside 'FIRST'."""
    assert matcher.hits("the minister said nothing") == {}
    assert "narr_Investigation" not in matcher.hits("FIRST responders arrived")


def test_hyphenated_terms_match(matcher):
    """'counter-terrorism' ends in a word char but contains punctuation."""
    assert "narr_Security" in matcher.hits("a counter-terrorism operation")


def test_no_match_returns_empty(matcher):
    assert matcher.hits("the weather is pleasant today") == {}
    assert matcher.hits("") == {}


def test_counts_covers_every_group_and_counts_repeats(matcher):
    counts = matcher.counts("An airstrike. Another airstrike. A third airstrike.")
    assert len(counts) == 11
    assert counts["narr_Security"] == 3


def test_group_features_shape_and_column_order(matcher):
    texts = ["DRDO missile test", "civilians displaced by flood risk", "nothing here"]
    mat = matcher.group_features(texts)
    assert mat.shape == (3, 11)
    assert mat.dtype == np.float32
    names = matcher.group_feature_names
    assert mat[0, names.index("grp_narr_Security")] > 0
    assert mat[1, names.index("grp_narr_Civilian")] > 0
    assert mat[2].sum() == 0


def test_log1p_transform_damps_repetition(matcher):
    once = matcher.group_features(["airstrike"], transform="log1p")
    thrice = matcher.group_features(["airstrike airstrike airstrike"], transform="log1p")
    i = matcher.group_feature_names.index("grp_narr_Security")
    assert thrice[0, i] > once[0, i]
    assert thrice[0, i] < 3 * once[0, i], "log1p must compress, not scale linearly"


def test_binary_transform_is_zero_or_one(matcher):
    mat = matcher.group_features(["airstrike airstrike"], transform="binary")
    assert set(np.unique(mat)) <= {0.0, 1.0}


def test_unknown_transform_raises(matcher):
    with pytest.raises(ValueError, match="unknown transform"):
        matcher.group_features(["x"], transform="sqrt")


def test_low_precision_terms_can_be_excluded():
    keep = KeywordMatcher(exclude_low_precision=False)
    drop = KeywordMatcher(exclude_low_precision=True)
    text = "activists say they were targeted"
    assert "prop_victimhood" in keep.hits(text)
    assert "prop_victimhood" not in drop.hits(text)


def test_build_features_modes_agree_on_row_count():
    texts = ["DRDO missile test", "nothing"]
    grp, gnames = build_features(texts, mode="group_counts")
    per, pnames = build_features(texts, mode="per_keyword")
    assert grp.shape == (2, len(gnames)) == (2, 11)
    assert per.shape == (2, len(pnames))
    assert per.shape[1] > grp.shape[1]


def test_build_features_off_is_zero_width():
    """'off' must still return a concatenable matrix."""
    mat, names = build_features(["anything"], mode="off")
    assert mat.shape == (1, 0)
    assert names == []


def test_build_features_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown keyword mode"):
        build_features(["x"], mode="magic")


def test_marker_injection_wraps_matches(matcher):
    out = matcher.mark("DRDO tested a missile test")
    assert "<Security> DRDO </Security>" in out


def test_features_see_past_the_truncation_point(matcher):
    """The reason these features exist: they read the whole document."""
    tail = "word " * 4000 + "ceasefire violated"
    mat = matcher.group_features([tail])
    assert mat[0, matcher.group_feature_names.index("grp_sev_High")] > 0
