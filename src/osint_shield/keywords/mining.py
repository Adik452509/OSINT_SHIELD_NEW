"""Building the blind propaganda review sheet (D18, D19).

Three buckets, filled to a fixed budget:

``original``  the 7 articles the first annotation called propaganda - re-judged
              under the own-voice rule
``rubric``    every article on which any rubric propaganda term fires
``random``    a seeded random sample of everything else - the only way to
              estimate how much propaganda the rubric misses

Rows are shuffled and carry no hint of their bucket; the key is written to a
separate file so the reviewer is not pulled towards anyone's earlier verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .matcher import KeywordMatcher

BUCKETS = ("original", "rubric", "random")


def article_text(row) -> str:
    body = row["clean_text"] if isinstance(row.get("clean_text"), str) else ""
    return f"{row['clean_headline']} {body}"


def rubric_hits(df: pd.DataFrame, matcher: KeywordMatcher | None = None) -> pd.Series:
    """Which rubric propaganda terms fire on each article (empty string if none)."""
    m = matcher or KeywordMatcher()

    def hits(row) -> str:
        found = {g: t for g, t in m.hits(article_text(row)).items() if g.startswith("prop_")}
        return "; ".join(f"{g[5:]}: {', '.join(t)}" for g, t in found.items())

    return df.apply(hits, axis=1)


def assign_buckets(df: pd.DataFrame, *, budget: int, seed: int = 0) -> pd.DataFrame:
    """Pick the review set and record why each article is in it.

    Priority: ``original`` first, then ``rubric``, then ``random`` fills the
    remaining budget from articles in neither. An article in two buckets keeps
    the first one.

    Returns:
        The chosen rows with ``bucket`` and ``rubric_terms`` columns, original order.

    Raises:
        ValueError: if the original and rubric buckets alone exceed the budget.
    """
    out = df.copy()
    out["rubric_terms"] = rubric_hits(out)
    out["bucket"] = None
    out.loc[out["y_propaganda"] == 1, "bucket"] = "original"
    out.loc[out["bucket"].isna() & (out["rubric_terms"] != ""), "bucket"] = "rubric"

    fixed = int(out["bucket"].notna().sum())
    if fixed > budget:
        raise ValueError(f"original + rubric buckets hold {fixed} articles, over the budget "
                         f"of {budget} - raise the budget or tighten the rubric")
    pool = out.index[out["bucket"].isna()]
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool, size=min(budget - fixed, len(pool)), replace=False)
    out.loc[pick, "bucket"] = "random"
    return out[out["bucket"].notna()]


def blind_sheet(chosen: pd.DataFrame, *, seed: int = 0,
                body_chars: int = 4000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle the chosen rows into a sheet with no bucket information.

    Returns:
        ``(sheet, key)``. ``sheet`` holds the text and empty columns for the
        reviewer; ``key`` maps each row to its bucket and is never shown.
    """
    order = chosen.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def clip(text) -> str:
        text = text if isinstance(text, str) else ""
        return text if len(text) <= body_chars else text[:body_chars].rsplit(" ", 1)[0] + " [...]"

    sheet = pd.DataFrame({
        "row": range(1, len(order) + 1),
        "id": order["id"].to_numpy(),
        "headline": order["clean_headline"].to_numpy(),
        "article": [clip(t) if isinstance(t, str) else "(headline only)"
                    for t in order["clean_text"]],
        "label": "", "types": "", "sure": "", "evidence": "", "notes": "",
    })
    key_cols = ["id", "bucket", "rubric_terms", "y_propaganda"]
    key = order[key_cols].assign(row=range(1, len(order) + 1))
    if "not_blind" in order:
        key["not_blind"] = order["not_blind"].to_numpy()
    return sheet, key[["row"] + [c for c in key.columns if c != "row"]]
