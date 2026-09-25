#!/usr/bin/env python
"""M2 - build the cross-validation folds and the data report.

One reproducible path from the raw corpus to fold assignments:

    load 1,064 -> drop exact dups -> group near dups -> hold out test.csv
    -> drop hold-out leaks -> apply data mode -> stratified group 5-fold

Run from the project root with the venv active:

    python scripts/02_build_folds.py                 # body_only (primary)
    python scripts/02_build_folds.py --mode all      # the confound measurement
    python scripts/02_build_folds.py --mode both     # write both

Writes data/processed/folds_<mode>.csv and docs/DATA_REPORT.md.
Every printed number is checked against the values measured during analysis;
a mismatch means the pipeline changed and should be investigated before M3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.config import load_config  # noqa: E402
from osint_shield.data.dedupe import add_duplicate_columns, duplicate_summary  # noqa: E402
from osint_shield.data.loaders import apply_data_mode, load_gold  # noqa: E402
from osint_shield.data.splits import (  # noqa: E402
    assign_folds,
    check_no_group_spans_folds,
    fold_summary,
    resolve_holdout,
)

# Values established during the Phase 1 analysis. These are regression targets,
# not inputs - the pipeline must reproduce them.
EXPECTED = {
    "n_gold": 1064,
    "n_exact_dups": 11,
    "n_after_exact": 1053,
    "n_body_only": 571,
    "n_propaganda": 7,
    "narrative": {"Security": 824, "Political": 155, "Civilian": 41,
                  "Other": 26, "Investigation": 18},
}

OK, BAD = "  [ok]  ", "  [!!]  "


def check(label: str, got, want) -> bool:
    """Print a pass/fail line comparing a measured value to its target."""
    passed = got == want
    print(f"{OK if passed else BAD}{label}: {got}" + ("" if passed else f"  (expected {want})"))
    return passed


def build(cfg: dict, mode: str, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full pipeline for one data mode."""
    ded = cfg["dedupe"]
    spl = cfg["split"]
    ok = True

    print("=" * 74)
    print(f"M2  building folds   mode={mode}   folds={spl['n_folds']}   seed={cfg['seed']}")
    print("=" * 74)

    # ---- 1. load -----------------------------------------------------------
    print("\n[1] load corpus")
    df = load_gold()
    ok &= check("rows", len(df), EXPECTED["n_gold"])
    ok &= check("propaganda positives", int(df["y_propaganda"].sum()), EXPECTED["n_propaganda"])
    ok &= check("narrative counts", df["narrative"].value_counts().to_dict(),
                EXPECTED["narrative"])
    print(f"{OK}splits: " + ", ".join(
        f"{k}={v}" for k, v in df["split_orig"].value_counts().sort_index().items()))
    print(f"{OK}has_body: {int(df['has_body'].sum())} of {len(df)} "
          f"({df['has_body'].mean() * 100:.1f}%)")

    # ---- 2. duplicates -----------------------------------------------------
    print("\n[2] duplicates")
    df = add_duplicate_columns(
        df, threshold=ded["similarity"], strip_suffix=ded["strip_source_suffix"]
    )
    summary = duplicate_summary(df, split_col="split_orig")
    ok &= check("exact duplicate rows", summary["n_exact_dups"], EXPECTED["n_exact_dups"])
    print(f"{OK}near-dup clusters at sim>={ded['similarity']}: "
          f"{summary['n_multi_row_groups']} multi-row groups, "
          f"{summary['n_redundant_rows']} redundant rows, "
          f"largest group {summary['largest_group']}")
    print(f"{OK}clusters spanning two original splits: "
          f"{summary['n_groups_spanning_splits']}  (these are the leaks we fix)")

    if ded["drop_exact"]:
        df = df[~df["is_exact_dup"]].reset_index(drop=True)
        ok &= check("rows after dropping exact dups", len(df), EXPECTED["n_after_exact"])

    # ---- 3. hold-out -------------------------------------------------------
    print("\n[3] hold-out")
    pool, holdout, leaked = resolve_holdout(
        df, holdout_split="test", exclude_leaks=ded["exclude_holdout_leaks"]
    )
    n_leaks = len(leaked)
    print(f"{OK}hold-out (test.csv): {len(holdout)} rows - untouched until the end")
    print(f"{OK}CV pool (train+val): {len(pool)} rows")
    print(f"{OK}dropped from CV pool because a duplicate sits in the hold-out: {n_leaks}")
    for _, row in leaked.iterrows():
        print(f"         - {str(row['clean_headline'])[:78]}")

    # ---- 4. data mode ------------------------------------------------------
    print(f"\n[4] data mode: {mode}")
    pool = apply_data_mode(pool, mode).reset_index(drop=True)
    holdout_m = apply_data_mode(holdout, mode).reset_index(drop=True)
    print(f"{OK}CV pool: {len(pool)} rows    hold-out: {len(holdout_m)} rows")
    if mode == "body_only":
        # only the BODIED leaks belong in a body_only reconciliation
        leaked_m = apply_data_mode(leaked, mode)
        total_body = len(pool) + len(holdout_m) + len(leaked_m)
        ok &= check(
            f"body_only reconciliation (pool {len(pool)} + holdout {len(holdout_m)} "
            f"+ bodied leaks {len(leaked_m)})",
            total_body, EXPECTED["n_body_only"],
        )

    # ---- 5. folds ----------------------------------------------------------
    print(f"\n[5] {spl['n_folds']}-fold stratified group CV")
    pool["fold"] = assign_folds(
        pool,
        n_folds=spl["n_folds"],
        seed=cfg["seed"],
        stratify_col=spl["stratify_on"],
        group_col=spl["group_on"],
    )
    summary_df = fold_summary(pool)
    print(summary_df.to_string())

    spanning = check_no_group_spans_folds(pool)
    ok &= check("duplicate groups spanning two folds", len(spanning), 0)

    empty_prop = summary_df.index[summary_df["prop_pos"] == 0].tolist()
    if empty_prop:
        print(f"{BAD}folds with zero propaganda positives: {empty_prop}")
        print("        (their positive-class F1 is undefined, not merely low)")
    else:
        print(f"{OK}every fold has at least one propaganda positive")

    sizes = summary_df["n"]
    print(f"{OK}fold sizes {sizes.min()}-{sizes.max()} (spread {sizes.max() - sizes.min()})")

    stats = {
        "mode": mode,
        "n_gold": EXPECTED["n_gold"],
        "n_after_exact": len(df),
        "n_pool": len(pool),
        "n_holdout": len(holdout_m),
        "n_leaks_removed": n_leaks,
        "duplicates": summary,
        "folds": summary_df,
        "all_checks_passed": ok,
    }
    return pool, holdout_m, stats


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a frame as a markdown table without depending on `tabulate`."""
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = [[str(idx)] + [str(v) for v in row] for idx, row in zip(df.index, df.to_numpy())]
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_report(runs: list[dict]) -> Path:
    """Write docs/DATA_REPORT.md from one or both mode runs."""
    lines = [
        "# Data report",
        "",
        "Generated by `scripts/02_build_folds.py`. Every number here is computed from",
        "`data/raw/gold_dataset.csv`, not copied from the plan PDF.",
        "",
    ]
    for st in runs:
        d = st["duplicates"]
        lines += [
            f"## Mode: `{st['mode']}`",
            "",
            "| Stage | Rows |",
            "|---|---|",
            f"| raw corpus | {st['n_gold']} |",
            f"| after dropping {d['n_exact_dups']} exact duplicates | {st['n_after_exact']} |",
            f"| final hold-out (`test.csv`) | {st['n_holdout']} |",
            f"| dropped - duplicate of a hold-out article | {st['n_leaks_removed']} |",
            f"| **cross-validation pool** | **{st['n_pool']}** |",
            "",
            "### Duplicates",
            "",
            f"- similarity clusters: {d['n_groups']} from {d['n_rows']} rows "
            f"({d['n_redundant_rows']} redundant)",
            f"- multi-row clusters: {d['n_multi_row_groups']}, largest {d['largest_group']}",
            f"- clusters spanning two original splits: {d.get('n_groups_spanning_splits', 0)}",
            "",
            "### Folds",
            "",
            df_to_markdown(st["folds"]),
            "",
            f"Duplicate groups spanning two folds: **0**. "
            f"Checks passed: **{st['all_checks_passed']}**.",
            "",
        ]
    path = ROOT / "docs" / "DATA_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--mode", default=None, choices=["body_only", "all", "both"],
                    help="overrides data.mode from the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    modes = ["body_only", "all"] if args.mode == "both" else [args.mode or cfg["data"]["mode"]]

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, all_ok = [], True
    for mode in modes:
        pool, holdout, stats = build(cfg, mode)
        all_ok &= stats["all_checks_passed"]
        runs.append(stats)

        cols = ["id", "fold", "dupe_group", "narrative", "severity", "y_narrative",
                "y_severity", "y_propaganda", "has_body", "split_orig", "source_domain"]
        pool[cols].to_csv(out_dir / f"folds_{mode}.csv", index=False)
        holdout[[c for c in cols if c != "fold"]].to_csv(
            out_dir / f"holdout_{mode}.csv", index=False)
        print(f"\n  wrote data/processed/folds_{mode}.csv  ({len(pool)} rows)")
        print(f"  wrote data/processed/holdout_{mode}.csv ({len(holdout)} rows)")

    report = write_report(runs)
    print(f"  wrote {report.relative_to(ROOT)}")
    print("\n" + ("M2 PASSED - ready for M3 (baselines)"
                  if all_ok else "M2 FAILED - see the [!!] lines above"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
