"""Cross-validation: every fold x every seed, resumable, then aggregated.

A full run is 15 trainings and ~30 minutes on the RTX 3050, so it is built to
survive interruption:

* each (fold, seed) result is written the moment it finishes - predictions
  first, then the result JSON atomically, so a JSON on disk always means a
  complete run;
* rerunning skips every (fold, seed) whose JSON exists;
* the config is snapshotted on first run, and a resume under a *different*
  config is refused rather than silently mixing two experiments.

Folds are fixed across seeds. Seeds vary initialisation and data order only,
so seed spread measures training noise and per-fold results can be paired
against a baseline scored on the same folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.metrics import LEAKAGE_ALARM_ACCURACY
from .fold import PreparedData, run_fold
from .trainer import TrainSettings

#: Per-run metrics carried into the aggregate.
METRIC_KEYS = (
    "narrative_macro_f1",
    "narrative_collapsed_macro_f1",
    "severity_f1_high",
    "propaganda_f1_pos",
    "narrative_accuracy",
    "severity_accuracy",
)


class ConfigMismatchError(RuntimeError):
    """Raised when resuming a CV run under a different config than it began with."""


def run_path(out_dir: Path, fold: int, seed: int) -> Path:
    return Path(out_dir) / "runs" / f"fold{fold}_seed{seed}.json"


def predictions_path(out_dir: Path, fold: int, seed: int) -> Path:
    return Path(out_dir) / "runs" / f"fold{fold}_seed{seed}_pred.csv"


def ensure_snapshot(out_dir: Path, cfg: dict, *, resume: bool) -> None:
    """Write the config snapshot, or verify it matches when resuming.

    Raises:
        ConfigMismatchError: resuming, and the config differs from the one the
            existing runs were trained under.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    current = json.dumps(cfg, sort_keys=True, indent=2, default=str)
    if resume and path.exists() and path.read_text(encoding="utf-8") != current:
        raise ConfigMismatchError(
            f"the config has changed since the runs in {out_dir} were trained.\n"
            "Resuming would average results from two different setups.\n"
            "Either restore the original config, or start over with --no-resume."
        )
    path.write_text(current, encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def run_cv(
    data: PreparedData,
    cfg: dict,
    *,
    folds: list[int],
    seeds: list[int],
    device,
    out_dir: Path,
    model_builder=None,
    settings: TrainSettings | None = None,
    resume: bool = True,
    log=print,
) -> list[dict]:
    """Train every (seed, fold) pair, skipping any already on disk.

    Returns:
        One summary dict per (seed, fold), in plan order.
    """
    out_dir = Path(out_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    plan = [(seed, fold) for seed in seeds for fold in folds]
    results: list[dict] = []
    durations: list[float] = []

    for i, (seed, fold) in enumerate(plan, start=1):
        path = run_path(out_dir, fold, seed)
        if resume and path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
            log(f"\n  [{i}/{len(plan)}] seed {seed} fold {fold} - done earlier, loaded")
            continue

        remaining = len(plan) - i + 1
        eta = (f"   (~{np.mean(durations) * remaining / 60:.0f} min left)"
               if durations else "")
        log(f"\n  [{i}/{len(plan)}] seed {seed} fold {fold}{eta}")

        res = run_fold(data, fold, cfg, seed=seed, device=device,
                       model_builder=model_builder, settings=settings, log=log)
        # predictions first: a result JSON on disk must imply a complete run
        res.predictions.to_csv(predictions_path(out_dir, fold, seed), index=False)
        summary = res.summary()
        _write_json_atomic(path, summary)

        durations.append(res.seconds)
        m = res.metrics
        log(f"      -> narrative {m['narrative_macro_f1']:.3f}  "
            f"3-class {m['narrative_collapsed_macro_f1']:.3f}  "
            f"severity {m['severity_f1_high']:.3f}  "
            f"best epoch {res.best_epoch}/{len(res.history)}  ({res.seconds:.0f}s)")
        results.append(summary)

    return results


def runs_frame(results: list[dict]) -> pd.DataFrame:
    """One row per (seed, fold) with flattened metrics and run diagnostics."""
    rows = []
    for r in results:
        m = r["metrics"]
        row = {
            "seed": int(r["seed"]),
            "fold": int(r["fold"]),
            "best_epoch": int(r["best_epoch"]),
            "epochs_run": len(r["history"]),
            "seconds": float(r["seconds"]),
            "peak_vram_gb": float(r["peak_vram_gb"]),
            "propaganda_n_pos": int(m.get("propaganda_n_pos", 0)),
            "propaganda_n_pred_pos": int(m.get("propaganda_n_pred_pos", 0)),
        }
        row.update({k: float(m[k]) for k in METRIC_KEYS if k in m})
        rows.append(row)
    if not rows:
        raise ValueError("no completed runs to aggregate")
    return pd.DataFrame(rows).sort_values(["seed", "fold"]).reset_index(drop=True)


def summarise(df: pd.DataFrame, *, max_epochs: int) -> dict:
    """Mean ± std per metric, plus the spreads that make the mean interpretable.

    ``seed_spread`` is the std of per-seed means - training noise.
    ``fold_spread`` is the std of per-fold means - data-split variation.
    """
    out: dict = {"n_runs": int(len(df)), "metrics": {}}
    for key in METRIC_KEYS:
        if key not in df:
            continue
        values = df[key].astype(float)
        seed_means = df.groupby("seed")[key].mean()
        fold_means = df.groupby("fold")[key].mean()
        out["metrics"][key] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std(ddof=0)), 4),
            "seed_spread": round(float(seed_means.std(ddof=0)), 4),
            "fold_spread": round(float(fold_means.std(ddof=0)), 4),
            "seed_means": {int(k): round(float(v), 4) for k, v in seed_means.items()},
            "fold_means": {int(k): round(float(v), 4) for k, v in fold_means.items()},
        }

    best = df["best_epoch"].astype(int)
    out["best_epoch"] = {
        "distribution": {int(k): int(v) for k, v in best.value_counts().sort_index().items()},
        "median": float(best.median()),
        "share_at_ceiling": round(float((best >= max_epochs).mean()), 3),
    }
    out["propaganda_pos_seen_per_seed"] = {
        int(k): int(v) for k, v in df.groupby("seed")["propaganda_n_pos"].sum().items()
    }
    out["leakage_alarms"] = int((df["narrative_accuracy"] > LEAKAGE_ALARM_ACCURACY).sum())
    out["seconds_total"] = round(float(df["seconds"].sum()), 1)
    out["peak_vram_gb_max"] = round(float(df["peak_vram_gb"].max()), 2)
    return out


def selection_analysis(results: list[dict]) -> dict | None:
    """How well did early stopping choose? Needs ``track_test_fold`` runs.

    For each run, compares the test score at the **selected** checkpoint with
    the score at the **final** epoch and the **best possible** epoch (the
    oracle - unreachable in practice, it peeks at the test fold). Also returns
    the mean test-fold learning curve by epoch.

    A selected score well below the oracle means the inner-val signal is too
    noisy to choose by; selected below final means early stopping is actively
    harmful and a fixed schedule would do better.

    Returns ``None`` when no run carries tracked metrics.
    """
    tasks = {"narrative_macro_f1": "narrative_macro_f1",
             "severity_f1_high": "severity_f1_high"}
    tracked = [r for r in results
               if r.get("history") and f"track_{next(iter(tasks))}" in r["history"][0]]
    if not tracked:
        return None

    out: dict = {"n_runs": len(tracked)}
    for key in tasks:
        tkey = f"track_{key}"
        selected = [float(r["metrics"][key]) for r in tracked]
        final = [float(r["history"][-1][tkey]) for r in tracked]
        oracle = [max(float(h[tkey]) for h in r["history"]) for r in tracked]
        curve: dict[int, list[float]] = {}
        for r in tracked:
            for h in r["history"]:
                curve.setdefault(int(h["epoch"]), []).append(float(h[tkey]))
        out[key] = {
            "selected": round(float(np.mean(selected)), 4),
            "final_epoch": round(float(np.mean(final)), 4),
            "oracle": round(float(np.mean(oracle)), 4),
            "curve": {e: round(float(np.mean(v)), 4) for e, v in sorted(curve.items())},
            "curve_n": {e: len(v) for e, v in sorted(curve.items())},
        }
    return out


def load_predictions(out_dir: Path, plan: list[tuple[int, int]]) -> pd.DataFrame:
    """Concatenate out-of-fold predictions for exactly the given (seed, fold) pairs.

    Reading by plan rather than globbing the directory keeps stale files from an
    earlier, differently-scoped run out of the aggregate.
    """
    frames = []
    for seed, fold in plan:
        path = predictions_path(out_dir, fold, seed)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError(f"no prediction files found in {out_dir}")
    return pd.concat(frames, ignore_index=True)
