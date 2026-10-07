"""The blind review sheet: buckets, budget, blindness."""

from __future__ import annotations

import pandas as pd
import pytest

from osint_shield.keywords.mining import assign_buckets, blind_sheet


@pytest.fixture
def corpus() -> pd.DataFrame:
    rows = [("o1", "Six brave sons honoured", None, 1),
            ("o2", "Neutral headline", "nothing loaded here", 1),
            ("r1", "Opposition plays vote bank politics", None, 0),
            ("r2", "Activists say they were silenced", "body", 0)]
    rows += [(f"n{i}", f"Routine report number {i}", "plain text", 0) for i in range(30)]
    return pd.DataFrame(rows, columns=["id", "clean_headline", "clean_text", "y_propaganda"])


def test_originals_always_included(corpus):
    chosen = assign_buckets(corpus, budget=10)
    assert {"o1", "o2"} <= set(chosen["id"])
    assert set(chosen.loc[chosen["id"].isin(["o1", "o2"]), "bucket"]) == {"original"}


def test_original_beats_rubric_when_both_apply(corpus):
    """o1 contains a rubric term but is an original - it keeps the first bucket."""
    chosen = assign_buckets(corpus, budget=10)
    assert chosen.set_index("id").loc["o1", "bucket"] == "original"


def test_rubric_hits_are_included(corpus):
    chosen = assign_buckets(corpus, budget=10).set_index("id")
    assert chosen.loc["r1", "bucket"] == "rubric"
    assert chosen.loc["r2", "bucket"] == "rubric"


def test_random_fills_exactly_to_budget(corpus):
    chosen = assign_buckets(corpus, budget=10)
    assert len(chosen) == 10
    assert (chosen["bucket"] == "random").sum() == 6


def test_random_never_takes_an_already_bucketed_article(corpus):
    chosen = assign_buckets(corpus, budget=20)
    assert chosen["id"].is_unique


def test_random_sample_is_reproducible(corpus):
    a = set(assign_buckets(corpus, budget=12, seed=3)["id"])
    b = set(assign_buckets(corpus, budget=12, seed=3)["id"])
    assert a == b


def test_budget_too_small_for_fixed_buckets_raises(corpus):
    with pytest.raises(ValueError, match="over the budget"):
        assign_buckets(corpus, budget=2)


def test_sheet_reveals_nothing_about_buckets(corpus):
    sheet, key = blind_sheet(assign_buckets(corpus, budget=10))
    for col in ("bucket", "rubric_terms", "y_propaganda"):
        assert col not in sheet.columns
    assert {"label", "types", "sure", "evidence", "notes"} <= set(sheet.columns)
    assert (sheet["label"] == "").all()


def test_sheet_is_shuffled_and_key_lines_up(corpus):
    chosen = assign_buckets(corpus, budget=10)
    sheet, key = blind_sheet(chosen, seed=1)
    assert list(sheet["id"]) != list(chosen["id"])          # order changed
    assert list(sheet["id"]) == list(key["id"])             # key matches row by row
    assert list(sheet["row"]) == list(key["row"])


def test_headline_only_articles_are_marked(corpus):
    sheet, _ = blind_sheet(assign_buckets(corpus, budget=10))
    assert (sheet.loc[sheet["id"] == "o1", "article"] == "(headline only)").all()


def test_long_articles_are_clipped_on_a_word_boundary():
    df = pd.DataFrame({"id": ["a"], "clean_headline": ["H"], "clean_text": ["word " * 3000],
                       "y_propaganda": [1]})
    sheet, _ = blind_sheet(assign_buckets(df, budget=1), body_chars=100)
    assert sheet.loc[0, "article"].endswith("[...]")
    assert len(sheet.loc[0, "article"]) < 120
