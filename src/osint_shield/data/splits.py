"""Cross-validation fold construction.

Three constraints have to hold simultaneously, and they pull against each
other:

1. **``test.csv`` stays untouched.** The plan asks for CV over all 1,064
   articles *and* an untouched ``test.csv``, but ``test.csv`` is a subset of
   the 1,064 — so the CV pool is ``train ∪ val`` only (docs/DECISIONS.md D1).
2. **Duplicates never span folds.** Near-duplicate clusters are assigned whole
   (D5). A cluster that reaches into ``test.csv`` is dropped from the CV pool
   entirely, or the final hold-out is compromised too.
3. **The 7 propaganda positives are spread across folds.** Stratifying on
   narrative alone can leave a fold with zero positives, making its
   positive-class F1 undefined rather than merely bad.

Constraint 3 is met by replacing the stratification label of the propaganda
positives with a sentinel, so they form their own stratum of 7 and
:class:`~sklearn.model_selection.StratifiedGroupKFold` spreads them ~1.4 per
fold. They are only 7 rows of ~500, so removing them from narrative
stratification costs nothing measurable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

#: Stratification label given to rare positives so they get their own stratum.
PROPAGANDA_STRATUM = "__PROP__"


def resolve_holdout(
    df: pd.DataFrame,
    *,
    holdout_split: str = "test",
    group_col: str = "dupe_group",
    exclude_leaks: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Separate the final hold-out from the cross-validation pool.

    Args:
        df: corpus with ``split_orig`` and a duplicate-group column.
        holdout_split: which original split is held out.
        group_col: duplicate cluster column.
        exclude_leaks: drop CV rows whose duplicate cluster also contains a
            hold-out row.

    Returns:
        ``(cv_pool, holdout, leaked)`` - ``leaked`` holds the rows removed from
        the pool, so callers can report which articles were dropped and
        reconcile counts after a data-mode filter.
    """
    is_holdout = df["split_orig"] == holdout_split
    holdout = df[is_holdout].copy()
    pool = df[~is_holdout].copy()
    leaked = pool.iloc[:0].copy()

    if exclude_leaks and len(holdout):
        contaminated = set(holdout[group_col])
        mask = pool[group_col].isin(contaminated)
        leaked = pool[mask].copy()
        pool = pool[~mask].copy()

    return (pool.reset_index(drop=True),
            holdout.reset_index(drop=True),
            leaked.reset_index(drop=True))


def stratification_labels(
    df: pd.DataFrame,
    *,
    stratify_col: str = "narrative",
    rare_positive_col: str | None = "y_propaganda",
) -> np.ndarray:
    """Build the per-row stratification label.

    Rows flagged in ``rare_positive_col`` are moved into their own stratum so
    they are spread across folds rather than clumping.
    """
    labels = df[stratify_col].astype(str).to_numpy()
    if rare_positive_col and rare_positive_col in df.columns:
        labels = np.where(df[rare_positive_col].to_numpy() == 1, PROPAGANDA_STRATUM, labels)
    return labels


def assign_folds(
    df: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    stratify_col: str = "narrative",
    group_col: str = "dupe_group",
    rare_positive_col: str | None = "y_propaganda",
) -> np.ndarray:
    """Assign each row a test-fold index in ``[0, n_folds)``.

    Uses :class:`StratifiedGroupKFold`, which keeps every member of a group in
    the same fold while balancing the stratification label as far as the
    grouping allows.

    Returns:
        Integer array of fold assignments, aligned to ``df``'s row order.

    Raises:
        ValueError: if the pool is too small, or a class has fewer members
            than there are folds in a way that makes stratification impossible.
    """
    if len(df) < n_folds:
        raise ValueError(f"cannot build {n_folds} folds from {len(df)} rows")

    y = stratification_labels(
        df, stratify_col=stratify_col, rare_positive_col=rare_positive_col
    )
    groups = df[group_col].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(splitter.split(df, y, groups)):
        folds[test_idx] = fold_idx

    if (folds < 0).any():
        raise ValueError("StratifiedGroupKFold left rows unassigned")
    return folds


def fold_summary(df: pd.DataFrame, *, fold_col: str = "fold") -> pd.DataFrame:
    """Per-fold counts: size, narrative distribution, severity and propaganda."""
    rows = []
    for fold, group in df.groupby(fold_col):
        row = {"fold": int(fold), "n": len(group)}
        row.update(group["narrative"].value_counts().to_dict())
        row["High"] = int((group["severity"] == "High").sum())
        row["prop_pos"] = int(group["y_propaganda"].sum())
        row["has_body"] = int(group["has_body"].sum())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0).astype({"fold": int, "n": int}).set_index("fold")


def check_no_group_spans_folds(df: pd.DataFrame, *, group_col: str = "dupe_group",
                               fold_col: str = "fold") -> list[int]:
    """Return the ids of any duplicate groups split across folds.

    An empty list is the assertion this module exists to satisfy.
    """
    spans = df.groupby(group_col)[fold_col].nunique()
    return spans[spans > 1].index.tolist()
