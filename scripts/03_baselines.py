#!/usr/bin/env python
"""M3 - baselines. Run before any transformer.

Produces the bar every later result is compared against, under **two fold
protocols**:

``clean``
    The M2 group-aware folds, ``test.csv`` held out. The honest bar.
``leaky``
    Flat ``StratifiedKFold`` over the whole deduped corpus, duplicates free to
    land in different folds. Reproduces the reference numbers measured during
    analysis, so it doubles as a pipeline sanity check.

The gap between them is not noise - it is how much of the original score was
duplicate leakage, and it belongs in the write-up.

    python scripts/03_baselines.py
    python scripts/03_baselines.py --mode body_only --protocol clean

Takes about a minute. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.config import load_config  # noqa: E402
from osint_shield.data.dedupe import add_duplicate_columns  # noqa: E402
from osint_shield.data.loaders import apply_data_mode, build_text, load_gold  # noqa: E402
from osint_shield.data.splits import assign_folds, resolve_holdout  # noqa: E402
from osint_shield.evaluation.baselines import cross_val_predict  # noqa: E402
from osint_shield.evaluation.metrics import (  # noqa: E402
    FoldScores,
    binary_f1,
    collapse_narrative,
    confusion_frame,
    leakage_alarm,
    macro_f1,
    per_class_f1,
    severity_by_has_body,
)

MODELS = ["majority", "keyword_rules", "keyword_logreg", "tfidf_logreg", "has_body"]

TASKS = {
    "narrative_5":    {"col": "narrative",    "metric": "macro"},
    "narrative_3":    {"col": "narrative",    "metric": "macro", "collapse": True},
    "severity_high":  {"col": "severity",     "metric": "binary", "pos_label": "High"},
    "propaganda_pos": {"col": "y_propaganda", "metric": "binary", "pos_label": 1},
}

#: Measured during Phase 1 with the leaky protocol. Regression targets.
REFERENCE_LEAKY = {
    ("body_only", "narrative_5"): 0.473,
    ("body_only", "narrative_3"): 0.627,
    ("body_only", "severity_high"): 0.686,
    ("body_only", "propaganda_pos"): 0.000,
    ("all", "narrative_5"): 0.487,
    ("all", "severity_high"): 0.631,
}
TOLERANCE = 0.03


def prepare(cfg: dict, mode: str, protocol: str) -> pd.DataFrame:
    """Load the corpus and attach a ``fold`` column for the chosen protocol."""
    df = add_duplicate_columns(
        load_gold(),
        threshold=cfg["dedupe"]["similarity"],
        strip_suffix=cfg["dedupe"]["strip_source_suffix"],
    )
    df = df[~df["is_exact_dup"]].reset_index(drop=True)

    if protocol == "clean":
        pool, _holdout, _leaked = resolve_holdout(
            df, holdout_split="test", exclude_leaks=cfg["dedupe"]["exclude_holdout_leaks"]
        )
        pool = apply_data_mode(pool, mode).reset_index(drop=True)
        pool["fold"] = assign_folds(
            pool, n_folds=cfg["split"]["n_folds"], seed=cfg["seed"],
            stratify_col=cfg["split"]["stratify_on"], group_col=cfg["split"]["group_on"],
        )
        out = pool
    elif protocol == "leaky":
        pool = apply_data_mode(df, mode).reset_index(drop=True)
        skf = StratifiedKFold(cfg["split"]["n_folds"], shuffle=True, random_state=cfg["seed"])
        folds = np.empty(len(pool), dtype=int)
        for i, (_, test_idx) in enumerate(skf.split(pool, pool["narrative"])):
            folds[test_idx] = i
        pool["fold"] = folds
        out = pool
    else:
        raise ValueError(f"unknown protocol: {protocol!r}")

    out["text"] = build_text(out, sep=" ", no_body_marker=cfg["data"]["no_body_marker"])
    return out


def score_task(df: pd.DataFrame, task: str, model: str) -> FoldScores | None:
    """Run one model on one task and return its per-fold scores."""
    spec = TASKS[task]
    y = df[spec["col"]].to_numpy()
    folds = df["fold"].to_numpy()

    # a constant feature carries no information - reported as skipped, not as 0
    if model == "has_body" and df["has_body"].nunique() < 2:
        return None

    y_true, y_pred = cross_val_predict(df, y, folds, model, _task_family(spec["col"]))

    if spec.get("collapse"):
        y_true, y_pred = collapse_narrative(y_true), collapse_narrative(y_pred)

    scores = FoldScores(name=model, task=task)
    for fold in np.unique(folds):
        m = folds == fold
        if spec["metric"] == "macro":
            scores.folds.append(macro_f1(y_true[m], y_pred[m]))
        else:
            scores.folds.append(binary_f1(y_true[m], y_pred[m], spec["pos_label"]))

    if task == "propaganda_pos":
        scores.extra["pos_in_test_folds"] = [
            int((y_true[folds == f] == 1).sum()) for f in np.unique(folds)
        ]
    if task == "narrative_5":
        triggered, acc = leakage_alarm(y_true, y_pred)
        scores.extra["accuracy"] = round(acc, 4)
        scores.extra["leakage_alarm"] = triggered
    if task == "severity_high":
        scores.extra["by_has_body"] = severity_by_has_body(
            y_true, y_pred, df["has_body"].to_numpy()
        )
    scores.extra["_y_true"] = y_true
    scores.extra["_y_pred"] = y_pred
    return scores


def _task_family(col: str) -> str:
    return {"narrative": "narrative", "severity": "severity",
            "y_propaganda": "propaganda"}[col]


def run(cfg: dict, mode: str, protocol: str) -> dict:
    df = prepare(cfg, mode, protocol)
    print("\n" + "=" * 78)
    print(f"  mode={mode}   protocol={protocol}   n={len(df)}   "
          f"folds={df['fold'].nunique()}")
    print("=" * 78)

    results: dict[str, dict] = {}
    for task in TASKS:
        print(f"\n  {task}")
        results[task] = {}
        for model in MODELS:
            scores = score_task(df, task, model)
            if scores is None:
                print(f"    {model:<16} skipped (has_body is constant in this mode)")
                continue
            flag = ""
            if scores.extra.get("leakage_alarm"):
                flag = "  <-- LEAKAGE ALARM: accuracy above the 74.3% annotator ceiling"
            if model == "tfidf_logreg":
                flag = flag or "   <-- the bar"
            print(f"    {model:<16} {scores}{flag}")
            payload = scores.as_dict()
            payload.pop("_y_true", None)
            payload.pop("_y_pred", None)
            results[task][model] = payload

            if task == "propaganda_pos" and model == "tfidf_logreg":
                pos = scores.extra["pos_in_test_folds"]
                print(f"    {'':<16} positives per test fold: {pos} "
                      f"(total {sum(pos)}) - diagnostic only")
            if task == "severity_high" and model == "tfidf_logreg":
                for regime, stats in scores.extra["by_has_body"].items():
                    print(f"    {'':<16}   {regime:<14} n={stats['n']:4d}  "
                          f"F1(High)={stats['f1_high']:.3f}  "
                          f"actual High rate={stats['actual_high_rate']:.3f}")

        # reference check against the Phase 1 measurements
        ref = REFERENCE_LEAKY.get((mode, task))
        if protocol == "leaky" and ref is not None and "tfidf_logreg" in results[task]:
            got = results[task]["tfidf_logreg"]["mean"]
            delta = abs(got - ref)
            mark = "ok" if delta <= TOLERANCE else "!!"
            print(f"    [{mark}] reproduces Phase 1 measurement {ref:.3f} "
                  f"(got {got:.3f}, delta {delta:.3f})")

    return {"n": len(df), "tasks": results}


def print_confusion(cfg: dict) -> None:
    """Pooled 3-class confusion for the strongest baseline - the error structure."""
    df = prepare(cfg, "body_only", "clean")
    scores = score_task(df, "narrative_3", "tfidf_logreg")
    print("\n" + "=" * 78)
    print("  pooled 3-class narrative confusion - tfidf_logreg, body_only, clean")
    print("  (rows = truth, columns = predicted)")
    print("=" * 78)
    print(confusion_frame(scores.extra["_y_true"], scores.extra["_y_pred"]).to_string())
    print("\n  per-class F1:")
    for label, f1 in per_class_f1(scores.extra["_y_true"], scores.extra["_y_pred"]).items():
        print(f"    {label:<12} {f1:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--mode", default="both", choices=["body_only", "all", "both"])
    ap.add_argument("--protocol", default="both", choices=["clean", "leaky", "both"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    modes = ["body_only", "all"] if args.mode == "both" else [args.mode]
    protocols = ["leaky", "clean"] if args.protocol == "both" else [args.protocol]

    print("=" * 78)
    print("M3  baselines")
    print("=" * 78)

    out: dict = {}
    for mode in modes:
        for protocol in protocols:
            out[f"{mode}|{protocol}"] = run(cfg, mode, protocol)

    print_confusion(cfg)

    # headline comparison: what did leakage buy?
    print("\n" + "=" * 78)
    print("  leakage cost - tfidf_logreg mean, leaky vs clean")
    print("=" * 78)
    print(f"  {'mode':<12}{'task':<18}{'leaky':>8}{'clean':>8}{'delta':>8}")
    for mode in modes:
        for task in TASKS:
            try:
                lk = out[f"{mode}|leaky"]["tasks"][task]["tfidf_logreg"]["mean"]
                cl = out[f"{mode}|clean"]["tasks"][task]["tfidf_logreg"]["mean"]
            except KeyError:
                continue
            print(f"  {mode:<12}{task:<18}{lk:>8.3f}{cl:>8.3f}{cl - lk:>+8.3f}")

    out_dir = ROOT / "runs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline_results.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  written to {path.relative_to(ROOT)}")
    print("\n  A fine-tuned encoder must beat the tfidf_logreg CLEAN row by a clear")
    print("  margin. If it does not, look for a bug before looking for a better model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
