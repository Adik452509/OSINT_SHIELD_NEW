#!/usr/bin/env python
"""M5 - full cross-validation: 5 folds x 3 seeds.

The first number that can honestly be compared with the TF-IDF bar.

    python scripts/05_train_cv.py                       # keyword fusion ON (group_counts)
    python scripts/05_train_cv.py --keywords off        # the control arm
    python scripts/05_train_cv.py --compare             # fusion vs off, side by side

~30 minutes per arm for mmBERT-small. Close browsers first - the GPU also drives
the display. Safe to interrupt: rerun the same command and it resumes from the
last completed (fold, seed).

TF-IDF is recomputed here on exactly the same five folds, so each fold is a
like-for-like comparison rather than a comparison against an average.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Must run before anything imports transformers - HF_HOME is read at import time.
from osint_shield.runtime import configure_environment  # noqa: E402

configure_environment()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from osint_shield.config import apply_overrides, load_config  # noqa: E402
from osint_shield.data.loaders import build_text, load_fold_frame  # noqa: E402
from osint_shield.evaluation.baselines import cross_val_predict  # noqa: E402
from osint_shield.evaluation.metrics import (  # noqa: E402
    binary_f1,
    collapse_narrative,
    confusion_frame,
    macro_f1,
    per_class_f1,
)
from osint_shield.models import load_tokenizer  # noqa: E402
from osint_shield.runtime import get_device, gpu_memory, is_cuda_oom  # noqa: E402
from osint_shield.training import prepare_data  # noqa: E402
from osint_shield.training.cv import (  # noqa: E402
    ConfigMismatchError,
    ensure_snapshot,
    load_predictions,
    run_cv,
    runs_frame,
    selection_analysis,
    summarise,
)

RUNS_DIR = ROOT / "runs" / "m5"
KEYWORD_ARMS = ["group_counts", "off", "per_keyword", "markers"]

#: task -> (column, scorer, metric key in the encoder results)
TASKS = {
    "narrative 5-class": ("narrative", "macro", "narrative_macro_f1"),
    "narrative 3-class": ("narrative", "macro3", "narrative_collapsed_macro_f1"),
    "severity F1(High)": ("severity", "binary", "severity_f1_high"),
}
OK, BAD, INFO = "  [ok]  ", "  [!!]  ", "  [..]  "


# --------------------------------------------------------------------- TF-IDF
def tfidf_per_fold(df: pd.DataFrame, cfg: dict) -> dict[str, dict[int, float]]:
    """Score TF-IDF + LogReg on exactly the encoder's folds."""
    frame = df.assign(text=build_text(df, sep=" ",
                                      no_body_marker=cfg["data"]["no_body_marker"]))
    folds = frame["fold"].to_numpy()
    out: dict[str, dict[int, float]] = {}
    for task, (col, scorer, _) in TASKS.items():
        family = "narrative" if col == "narrative" else "severity"
        y_true, y_pred = cross_val_predict(frame, frame[col].to_numpy(), folds,
                                           "tfidf_logreg", family)
        if scorer == "macro3":
            y_true, y_pred = collapse_narrative(y_true), collapse_narrative(y_pred)
        scores = {}
        for f in np.unique(folds):
            m = folds == f
            scores[int(f)] = (binary_f1(y_true[m], y_pred[m], "High") if scorer == "binary"
                              else macro_f1(y_true[m], y_pred[m]))
        out[task] = scores
    return out


def verdict(deltas: list[float]) -> str:
    """Read a set of per-fold paired deltas conservatively. Five folds is not many."""
    mean, wins = float(np.mean(deltas)), sum(d > 0 for d in deltas)
    if abs(mean) < 0.02:
        return "matches the bar"
    if mean > 0 and wins >= 4:
        return "BEATS the bar"
    if mean < 0 and wins <= 1:
        return "BELOW the bar"
    return "inconclusive"


# -------------------------------------------------------------------- reports
def report(df_runs: pd.DataFrame, summary: dict, tfidf: dict, oof: pd.DataFrame,
           cfg: dict) -> list[str]:
    """Print the M5 report and return it as markdown lines for report.md."""
    md: list[str] = []
    n = summary["n_runs"]
    max_epochs = cfg["training"]["epochs"]

    print("\n" + "=" * 86)
    print(f"  RESULTS - {n} runs ({df_runs['seed'].nunique()} seeds x "
          f"{df_runs['fold'].nunique()} folds)")
    print("=" * 86)
    header = (f"  {'task':<20}{'encoder':>16}{'TF-IDF':>10}{'paired Δ':>11}"
              f"{'folds won':>11}   verdict")
    print(header)
    md += ["| task | encoder (mean ± std) | TF-IDF | paired Δ | folds won | verdict |",
           "|---|---|---|---|---|---|"]

    for task, (_, _, key) in TASKS.items():
        s = summary["metrics"][key]
        fold_means = s["fold_means"]
        tf = tfidf[task]
        deltas = [fold_means[f] - tf[f] for f in sorted(tf) if f in fold_means]
        wins = sum(d > 0 for d in deltas)
        v = verdict(deltas)
        enc = f"{s['mean']:.3f} ± {s['std']:.3f}"
        print(f"  {task:<20}{enc:>16}{np.mean(list(tf.values())):>10.3f}"
              f"{np.mean(deltas):>+11.3f}{f'{wins}/{len(deltas)}':>11}   {v}")
        md.append(f"| {task} | {enc} | {np.mean(list(tf.values())):.3f} | "
                  f"{np.mean(deltas):+.3f} | {wins}/{len(deltas)} | {v} |")

    p = summary["metrics"]["propaganda_f1_pos"]
    seen = summary["propaganda_pos_seen_per_seed"]
    print(f"  {'propaganda F1(pos)':<20}{p['mean']:>9.3f} ± {p['std']:.3f}"
          f"   diagnostic only - positives in test folds per seed: {seen}")
    md.append(f"| propaganda F1(pos) | {p['mean']:.3f} ± {p['std']:.3f} | 0.000 | — | — | "
              f"diagnostic ({seen} positives per seed) |")

    print("\n  spread  (std of per-seed means = training noise; per-fold = split variation)")
    for key, label in [("narrative_macro_f1", "narrative 5-class"),
                       ("severity_f1_high", "severity F1(High)")]:
        s = summary["metrics"][key]
        print(f"    {label:<20} seed {s['seed_spread']:.3f}   fold {s['fold_spread']:.3f}   "
              f"seed means {s['seed_means']}")

    be = summary["best_epoch"]
    if cfg["training"].get("early_stopping", True):
        print(f"\n  best epoch  distribution {be['distribution']}   median {be['median']:.0f}   "
              f"at the {max_epochs}-epoch ceiling: {be['share_at_ceiling'] * 100:.0f}%")
        if be["share_at_ceiling"] >= 0.5:
            print(f"{BAD}most runs peaked at the final epoch - D8 trigger: sweep LR/epochs")
        md += ["", f"Best epoch distribution {be['distribution']}, "
                   f"{be['share_at_ceiling'] * 100:.0f}% at the ceiling."]
    else:
        print(f"\n  fixed schedule: every run trained {max_epochs} epochs and kept the last")
        md += ["", f"Fixed schedule: {max_epochs} epochs, final epoch kept."]

    # pooled error structure across every run
    t3, p3 = collapse_narrative(oof["true_narrative"]), collapse_narrative(oof["pred_narrative"])
    print("\n  pooled 3-class confusion, all runs  (rows = truth)")
    print("  " + confusion_frame(t3, p3).to_string().replace("\n", "\n  "))
    print("\n  per-class F1, 5-class, pooled:")
    for label, f1 in per_class_f1(oof["true_narrative"], oof["pred_narrative"]).items():
        print(f"    {label:<14} {f1:.3f}")

    alarms = summary["leakage_alarms"]
    print(f"\n{OK if alarms == 0 else BAD}leakage alarms (narrative accuracy > 0.92): {alarms}")
    print(f"{INFO}total training time {summary['seconds_total'] / 60:.1f} min   "
          f"peak VRAM {summary['peak_vram_gb_max']:.2f} GB")
    return md


def print_selection(sel: dict | None, max_epochs: int, early_stopping: bool = True) -> None:
    """Report the test-fold learning curve, and how well early stopping chose."""
    if sel is None:
        return
    if not early_stopping:
        curve = sel["narrative_macro_f1"]["curve"]
        best_e = max(curve, key=curve.get)
        print(f"\n  test-fold learning curve, fixed schedule  ({sel['n_runs']} runs, all epochs)")
        print("  (diagnostic only - the CV is a development set now; test.csv stays held out)")
        print("    mean test narrative macro-F1 by epoch:")
        print("      " + "  ".join(f"{e}:{v:.3f}" for e, v in curve.items()))
        print(f"    peak at epoch {best_e} ({curve[best_e]:.3f}); the kept final epoch scores "
              f"{curve[max(curve)]:.3f}")
        return
    print(f"\n  checkpoint selection, measured on the test folds  ({sel['n_runs']} runs)")
    print("  (diagnostic only - the CV is a development set now; test.csv stays held out)")
    for key, label in [("narrative_macro_f1", "narrative 5-class"),
                       ("severity_f1_high", "severity F1(High)")]:
        s = sel[key]
        print(f"    {label:<20} selected {s['selected']:.3f}   final epoch "
              f"{s['final_epoch']:.3f}   best possible {s['oracle']:.3f}")
    curve = sel["narrative_macro_f1"]["curve"]
    n = sel["narrative_macro_f1"]["curve_n"]
    print("    mean test narrative macro-F1 by epoch:")
    print("      " + "  ".join(f"{e}:{v:.3f}" for e, v in curve.items()))
    print("      runs reaching each epoch: " + " ".join(f"{e}:{n[e]}" for e in curve))
    s = sel["narrative_macro_f1"]
    if s["selected"] < s["final_epoch"] - 0.01:
        print(f"{BAD}early stopping chose WORSE than simply taking the final epoch")
    if s["oracle"] - s["selected"] > 0.05:
        print(f"{INFO}selection leaves {s['oracle'] - s['selected']:.3f} on the table - "
              "the 47-row inner-val signal is noisy")


PAIRED_KEYS = [("narrative_macro_f1", "narrative 5-class"),
               ("narrative_collapsed_macro_f1", "narrative 3-class"),
               ("severity_f1_high", "severity F1(High)")]


def paired_table(frames: dict[str, pd.DataFrame], baseline: str, title: str) -> None:
    """Means per run set, then each set paired against ``baseline`` by (seed, fold)."""
    width = max(14, max(len(k) for k in frames) + 2)
    print("=" * 86)
    print(f"  {title}")
    print("=" * 86)
    print(f"  {'run':<{width}}" + "".join(f"{label:>22}" for _, label in PAIRED_KEYS))
    for name, df in frames.items():
        cells = "".join(f"{df[k].mean():>14.3f} ± {df[k].std(ddof=0):.3f}"
                        for k, _ in PAIRED_KEYS)
        print(f"  {name:<{width}}{cells}")

    base = frames[baseline].set_index(["seed", "fold"])
    print(f"\n  paired against '{baseline}'  (same fold, same seed - one variable differs)")
    for name, df in frames.items():
        if name == baseline:
            continue
        joined = df.set_index(["seed", "fold"]).join(base, rsuffix="_base", how="inner")
        print(f"  {name}  ({len(joined)} paired runs)")
        for k, label in PAIRED_KEYS:
            d = joined[k] - joined[f"{k}_base"]
            print(f"    {label:<22} Δ {d.mean():+.3f} ± {d.std(ddof=0):.3f}   "
                  f"better in {(d > 0).sum()}/{len(d)} runs")


def compare_arms(model_short: str, mode: str, tag: str = "") -> int:
    """Keyword fusion vs no fusion: paired per (fold, seed), same folds, same seeds."""
    suffix = f"_{tag}" if tag else ""
    frames = {}
    for arm in KEYWORD_ARMS:
        path = RUNS_DIR / f"{model_short}_{mode}_kw-{arm}{suffix}" / "cv.csv"
        if path.exists():
            frames[arm] = pd.read_csv(path)
    if len(frames) < 2 or "off" not in frames:
        print(f"need the 'off' arm and at least one other under {RUNS_DIR}; "
              f"found {sorted(frames)}")
        return 1
    paired_table(frames, "off", f"keyword fusion arms - {model_short}, {mode}")
    return 0


def compare_tags(model_short: str, mode: str, arm: str, base: str, tag: str) -> int:
    """An ablation run against its reference run - the M6 comparison.

    ``base`` is either a tag of the same model/mode/arm, or the full folder
    name of any run under ``runs/m5`` - which is how a different model (the
    XLM-R bake-off) is paired against the mmBERT reference.
    """
    this = RUNS_DIR / f"{model_short}_{mode}_kw-{arm}_{tag}"
    base_dir = RUNS_DIR / base
    if not (base_dir / "cv.csv").exists():
        base_dir = RUNS_DIR / f"{model_short}_{mode}_kw-{arm}_{base}"
    frames = {}
    for label, folder in ((base, base_dir), (f"{model_short} {tag}", this)):
        path = folder / "cv.csv"
        if not path.exists():
            print(f"missing {path.relative_to(ROOT)} - has that run finished?")
            return 1
        frames[label] = pd.read_csv(path)
    paired_table(frames, base, f"'{model_short} {tag}' vs reference '{base}' - {mode}")
    return 0


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="mmbert_small.yaml")
    ap.add_argument("--mode", choices=["body_only", "all"], default=None)
    ap.add_argument("--keywords", choices=KEYWORD_ARMS, default=None,
                    help="keyword fusion arm (default: from config)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="retrain every run even if results exist")
    ap.add_argument("--compare", action="store_true",
                    help="compare completed keyword arms instead of training")
    ap.add_argument("--tag", default="",
                    help="suffix for the output folder, so a new configuration never "
                         "overwrites or resumes into an earlier one (e.g. --tag v2)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override one config value, e.g. --set training.early_stopping=false "
                         "(repeatable; unknown keys are an error)")
    ap.add_argument("--vs", default=None, metavar="BASE",
                    help="compare the run named by --tag against a reference: a tag of the "
                         "same model, or the full folder name of any run under runs/m5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        cfg = apply_overrides(cfg, args.overrides)
    except (KeyError, ValueError) as exc:
        print(f"{BAD}{exc}")
        return 1
    mode = args.mode or cfg["data"]["mode"]
    model_short = cfg["model"]["name"].split("/")[-1]
    arm = args.keywords or (cfg["keywords"]["mode"] if cfg["keywords"]["enabled"] else "off")
    if args.compare:
        return compare_arms(model_short, mode, args.tag)
    if args.vs:
        if not args.tag:
            print(f"{BAD}--vs needs --tag to name the run being compared")
            return 1
        return compare_tags(model_short, mode, arm, args.vs, args.tag)
    cfg["keywords"]["mode"] = arm
    cfg["keywords"]["enabled"] = arm != "off"
    seeds = args.seeds or cfg["seeds"]
    folds = args.folds or list(range(cfg["split"]["n_folds"]))
    run_name = f"{model_short}_{mode}_kw-{arm}" + (f"_{args.tag}" if args.tag else "")
    out_dir = RUNS_DIR / run_name

    print("=" * 86)
    print(f"M5  {cfg['model']['name']}   mode={mode}   keywords={arm}   "
          f"pooling={cfg['model']['pooling']}" + (f"   tag={args.tag}" if args.tag else ""))
    print(f"    seeds {seeds}   folds {folds}   -> {len(seeds) * len(folds)} runs")
    if args.overrides:
        print(f"    overrides: {'  '.join(args.overrides)}")
    print(f"    early stopping {'on' if cfg['training'].get('early_stopping', True) else 'OFF (fixed schedule)'}"
          f"   max_length {cfg['tokenizer']['max_length']}   "
          f"embeddings {'frozen' if cfg['model'].get('freeze_embeddings') else 'trainable'}")
    print(f"    monitor {cfg['training']['monitor']}   epochs <= {cfg['training']['epochs']}   "
          f"lr {cfg['training']['lr']}   out {out_dir.relative_to(ROOT)}")
    print("=" * 86)

    try:
        ensure_snapshot(out_dir, cfg, resume=not args.no_resume)
    except ConfigMismatchError as exc:
        print(f"\n{BAD}{exc}")
        return 1

    device = get_device()
    mem = gpu_memory()
    if mem is None:
        print(f"{BAD}no GPU - 15 runs on CPU would take many hours. Check M1.")
        return 1
    free, total = mem
    print(f"{OK if free >= 3.5 else BAD}free VRAM {free:.2f} of {total:.2f} GB"
          + ("" if free >= 3.5 else "  - close browsers and other GPU apps first"))

    tok = load_tokenizer(cfg["model"]["name"])
    df = load_fold_frame(mode)
    data = prepare_data(df, tok, cfg)
    print(f"{OK}{len(df)} articles   {data.truncated.mean() * 100:.1f}% truncated at "
          f"{data.max_length}   keyword features {data.n_keyword_features}")

    try:
        results = run_cv(data, cfg, folds=folds, seeds=seeds, device=device,
                         out_dir=out_dir, resume=not args.no_resume)
    except KeyboardInterrupt:
        print("\n\n  interrupted - completed runs are saved. Rerun the same command to resume.")
        return 130
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        print(f"\n{BAD}CUDA out of memory. Completed runs are saved.")
        print("        Close browsers / Copilot / Xbox app, then rerun the same command.")
        return 1

    df_runs = runs_frame(results)
    summary = summarise(df_runs, max_epochs=cfg["training"]["epochs"])
    oof = load_predictions(out_dir, [(s, f) for s in seeds for f in folds])
    tfidf = tfidf_per_fold(df, cfg)
    md = report(df_runs, summary, tfidf, oof, cfg)
    selection = selection_analysis(results)
    print_selection(selection, cfg["training"]["epochs"],
                    cfg["training"].get("early_stopping", True))

    df_runs.to_csv(out_dir / "cv.csv", index=False)
    oof.to_csv(out_dir / "oof.csv", index=False)
    summary["tfidf_per_fold"] = tfidf
    summary["selection"] = selection
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    (out_dir / "report.md").write_text(
        f"# M5 · {cfg['model']['name']} · {mode} · keywords={arm}\n\n" + "\n".join(md) + "\n",
        encoding="utf-8")
    print(f"\n  written to {out_dir.relative_to(ROOT)}  (cv.csv, oof.csv, summary.json, report.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
