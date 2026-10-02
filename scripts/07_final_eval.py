#!/usr/bin/env python
"""M7 - train the final model and score it ONCE on the held-out articles.

Everything before this was development: ~8 configurations were compared on the
same 5 folds, so those numbers are optimistic. `test.csv` has been untouched
since Phase 1 (D1). This script produces the honest number.

Fixed in advance, before any held-out result exists:
  * headline number  = mean +/- std over 3 seeds, on the 94 articles WITH body text
  * deployed model   = seed 42 (the config's own seed) - choosing the best seed
                       after seeing scores would itself be test-set tuning
  * headline-only    = the 65 held-out articles without a body, fed in as the
                       plain headline (as the live feed will) and reported
                       SEPARATELY - the model never trained on such inputs

Runs once. A second run reprints the stored result instead of re-scoring;
--force re-scores, and doing so after changing anything is test-set tuning.

    python scripts/07_final_eval.py

~6 minutes on the RTX 3050. Writes runs/m7/ and docs/FINAL_RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.runtime import configure_environment  # noqa: E402

configure_environment()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from osint_shield.config import (  # noqa: E402
    load_config,
    narrative_label_maps,
    severity_label_maps,
)
from osint_shield.data.loaders import build_text, load_fold_frame, load_gold  # noqa: E402
from osint_shield.evaluation.baselines import build_baseline  # noqa: E402
from osint_shield.evaluation.metrics import (  # noqa: E402
    collapse_narrative,
    confusion_frame,
    per_class_f1,
    score_multitask,
)
from osint_shield.keywords import build_features  # noqa: E402
from osint_shield.models import load_tokenizer  # noqa: E402
from osint_shield.paths import DATA_PROCESSED, GOLD_CSV  # noqa: E402
from osint_shield.runtime import get_device, gpu_memory  # noqa: E402
from osint_shield.training.fold import (  # noqa: E402
    _free_memory,
    assert_disjoint,
    predictions_frame,
    prepare_data,
)
from osint_shield.training.final import save_model, train_final  # noqa: E402

OUT = ROOT / "runs" / "m7"
RESULTS = OUT / "final_results.json"
REPORT_MD = ROOT / "docs" / "FINAL_RESULTS.md"

#: The cross-validation estimate for the chosen recipe (D15) - the number the
#: held-out result is read against. Expected to be optimistic (winner's curse).
CV_ESTIMATE = {"narrative_macro_f1": 0.524, "narrative_collapsed_macro_f1": 0.646,
               "severity_f1_high": 0.692}
METRICS = [("narrative_macro_f1", "narrative 5-class"),
           ("narrative_collapsed_macro_f1", "narrative 3-class"),
           ("severity_f1_high", "severity F1(High)")]
SUBSETS = ["bodied", "headline_only", "all"]
BASELINES = ["majority", "keyword_rules", "tfidf_logreg"]
OK, BAD, INFO = "  [ok]  ", "  [!!]  ", "  [..]  "


# ---------------------------------------------------------------------- data
def load_holdout() -> pd.DataFrame:
    """The held-out articles, with text and the annotators' own signals."""
    hold = pd.read_csv(DATA_PROCESSED / "holdout_all.csv")
    meta = load_gold()[["id", "clean_headline", "clean_text", "narrative_confidence",
                        "severity_confidence", "agree_narrative", "agree_severity",
                        "label_status"]]
    llama = pd.read_csv(GOLD_CSV, usecols=["id", "llama_narrative", "llama_severity"])
    return (hold.merge(meta, on="id", validate="one_to_one")
                .merge(llama, on="id", validate="one_to_one"))


def subset_masks(frame: pd.DataFrame, idx: np.ndarray) -> dict[str, np.ndarray]:
    has_body = frame["has_body"].to_numpy()[idx].astype(bool)
    return {"bodied": has_body, "headline_only": ~has_body,
            "all": np.ones(len(idx), dtype=bool)}


def score_subsets(y_true: dict, y_pred: dict, masks: dict) -> dict:
    return {name: score_multitask({t: v[m] for t, v in y_true.items()},
                                  {t: v[m] for t, v in y_pred.items()})
            for name, m in masks.items() if m.any()}


# ------------------------------------------------------------------ baselines
def baseline_scores(frame, train_idx, hold_idx, masks) -> dict:
    """Majority, the keyword rubric, and TF-IDF - trained on the same 476 articles."""
    df = frame.assign(text=build_text(frame, sep=" ", no_body_marker=""))
    narr_to_id, _ = narrative_label_maps()
    sev_to_id, _ = severity_label_maps()
    targets = {"narrative": ("narrative", "narrative"), "severity": ("severity", "severity"),
               "propaganda": ("y_propaganda", "propaganda")}
    y_true = {t: frame[c].to_numpy()[hold_idx] for t, c in
              {"narrative": "y_narrative", "severity": "y_severity",
               "propaganda": "y_propaganda"}.items()}
    out = {}
    for name in BASELINES:
        preds = {}
        for task, (col, family) in targets.items():
            model = build_baseline(name, family).fit(df.iloc[train_idx],
                                                     df[col].to_numpy()[train_idx])
            raw = np.asarray(model.predict(df.iloc[hold_idx]))
            if task == "narrative":
                raw = np.array([narr_to_id[str(v)] for v in raw])
            elif task == "severity":
                raw = np.array([sev_to_id[str(v)] for v in raw])
            preds[task] = raw.astype(int)
        out[name] = score_subsets(y_true, preds, masks)
    return out


# ------------------------------------------------------------- error analysis
def error_analysis(preds: pd.DataFrame, hold: pd.DataFrame) -> dict:
    """Do the model's errors fall where the annotators themselves disagreed?"""
    df = preds.merge(hold[["id", "agree_narrative", "agree_severity", "label_status",
                           "narrative_confidence", "llama_narrative", "llama_severity"]],
                     on="id", how="left")
    df["narr_ok"] = df["true_narrative"] == df["pred_narrative"]
    df["sev_ok"] = df["true_severity"] == df["pred_severity"]
    df["conf_bin"] = pd.cut(df["narrative_confidence"], [0, 0.699, 0.849, 1.0],
                            labels=["< 0.70", "0.70-0.85", ">= 0.85"])

    out: dict = {"n": int(len(df))}
    out["narrative_acc_by_agreement"] = {
        ("annotators agreed" if int(k) == 1 else "annotators disagreed"):
            {"n": int(len(g)), "accuracy": round(float(g["narr_ok"].mean()), 3)}
        for k, g in df.groupby("agree_narrative")}
    out["severity_acc_by_agreement"] = {
        ("annotators agreed" if int(k) == 1 else "annotators disagreed"):
            {"n": int(len(g)), "accuracy": round(float(g["sev_ok"].mean()), 3)}
        for k, g in df.groupby("agree_severity")}
    out["narrative_acc_by_label_status"] = {
        str(k): {"n": int(len(g)), "accuracy": round(float(g["narr_ok"].mean()), 3)}
        for k, g in df.groupby("label_status")}
    out["narrative_acc_by_annotator_confidence"] = {
        str(k): {"n": int(len(g)), "accuracy": round(float(g["narr_ok"].mean()), 3)}
        for k, g in df.groupby("conf_bin", observed=True)}

    errors = df[~df["narr_ok"]]
    out["narrative_errors"] = int(len(errors))
    if len(errors):
        out["share_of_errors_on_disagreed_articles"] = round(
            float((errors["agree_narrative"] == 0).mean()), 3)
        out["share_of_errors_matching_llama"] = round(
            float((errors["pred_narrative"] == errors["llama_narrative"]).mean()), 3)
    out["base_rate_disagreed"] = round(float((df["agree_narrative"] == 0).mean()), 3)
    return out


# ------------------------------------------------------------------ reporting
def fmt(mean, std=None):
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {std:.3f}"


def print_report(res: dict) -> list[str]:
    """Print the M7 report; return it as markdown lines."""
    md = ["# Final results — held-out evaluation (M7)", "",
          f"Evaluated once on {res['evaluated_at']}. Model: `{res['model']}`, recipe D15. "
          f"Trained on {res['n_train']} articles; scored on the {res['n_holdout']} held-out "
          "articles of `test.csv`, untouched since Phase 1.", ""]
    enc, base = res["encoder"], res["baselines"]

    for subset, title in [("bodied", "HELD-OUT, articles with body text — THE HEADLINE NUMBER"),
                          ("headline_only", "held-out, headline-only articles — reported separately"),
                          ("all", "held-out, all articles")]:
        if subset not in enc["mean"]:
            continue
        n = res["subset_n"][subset]
        print("\n" + "=" * 92)
        print(f"  {title}   (n={n})")
        print("=" * 92)
        print(f"  {'task':<20}{'encoder, 3 seeds':>20}{'seed 42':>10}"
              f"{'TF-IDF':>9}{'rubric':>9}{'majority':>10}"
              + (f"{'CV est.':>10}" if subset == "bodied" else ""))
        md += [f"## {title} (n = {n})", "",
               "| task | encoder (3 seeds) | seed 42 (deployed) | TF-IDF | keyword rubric | "
               "majority |" + (" CV estimate |" if subset == "bodied" else ""),
               "|---|---|---|---|---|---|" + ("---|" if subset == "bodied" else "")]
        for key, label in METRICS:
            m, s = enc["mean"][subset][key], enc["std"][subset][key]
            d = enc["deployed"][subset][key]
            row = [fmt(m, s), fmt(d)] + [fmt(base[b][subset][key]) for b in
                                         ("tfidf_logreg", "keyword_rules", "majority")]
            cv = CV_ESTIMATE[key] if subset == "bodied" else None
            print(f"  {label:<20}{row[0]:>20}{row[1]:>10}{row[2]:>9}{row[3]:>9}{row[4]:>10}"
                  + (f"{cv:>10.3f}" if cv is not None else ""))
            md.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |"
                      + (f" {cv:.3f} |" if cv is not None else ""))
        p = enc["mean"][subset]["propaganda_f1_pos"]
        npos = enc["deployed"][subset]["propaganda_n_pos"]
        print(f"  {'propaganda F1(pos)':<20}{p:>20.3f}   diagnostic only - {npos} positive(s) "
              "in this subset")
        md += [f"| propaganda F1(pos) | {p:.3f} | — | — | — | — | "
               + ("— |" if subset == "bodied" else "") + "", "",
               f"Propaganda is diagnostic only: {npos} positive(s) in this subset.", ""]

    gap = enc["mean"]["bodied"]["narrative_macro_f1"] - CV_ESTIMATE["narrative_macro_f1"]
    print(f"\n{INFO}held-out vs CV estimate, narrative 5-class: {gap:+.3f} "
          "(CV was selected from ~8 configs - some drop is expected)")
    md += [f"Held-out minus CV estimate, narrative 5-class: **{gap:+.3f}** "
           "(the CV figure was the best of ~8 configurations on the same folds).", ""]

    ea = res["error_analysis"]
    print("\n" + "=" * 92)
    print(f"  error analysis — deployed model (seed 42), held-out bodied articles (n={ea['n']})")
    print("=" * 92)
    md += ["## Error analysis — deployed model, bodied held-out articles", ""]
    for key, title in [("narrative_acc_by_agreement", "narrative accuracy by Claude/Llama agreement"),
                       ("severity_acc_by_agreement", "severity accuracy by Claude/Llama agreement"),
                       ("narrative_acc_by_annotator_confidence", "narrative accuracy by annotator confidence"),
                       ("narrative_acc_by_label_status", "narrative accuracy by label status")]:
        print(f"\n  {title}")
        md += [f"**{title}**", "", "| group | n | accuracy |", "|---|---|---|"]
        for g, v in ea[key].items():
            print(f"    {g:<22} n={v['n']:<4} accuracy {v['accuracy']:.3f}")
            md.append(f"| {g} | {v['n']} | {v['accuracy']:.3f} |")
        md.append("")
    if ea.get("narrative_errors"):
        print(f"\n  {ea['narrative_errors']} narrative errors:")
        print(f"    {ea['share_of_errors_on_disagreed_articles'] * 100:.0f}% fall on articles the "
              f"annotators disagreed on (base rate {ea['base_rate_disagreed'] * 100:.0f}%)")
        print(f"    {ea['share_of_errors_matching_llama'] * 100:.0f}% match Llama's label - the "
              "model sided with the other annotator")
        md += [f"Of {ea['narrative_errors']} narrative errors, "
               f"**{ea['share_of_errors_on_disagreed_articles'] * 100:.0f}%** fall on articles the "
               f"annotators disagreed on (base rate {ea['base_rate_disagreed'] * 100:.0f}%), and "
               f"**{ea['share_of_errors_matching_llama'] * 100:.0f}%** match Llama's label.", ""]

    print("\n  pooled 3-class confusion, deployed model, bodied  (rows = truth)")
    print("  " + pd.DataFrame(res["confusion_3class"]).to_string().replace("\n", "\n  "))
    print("\n  per-class F1, 5-class, deployed model, bodied:")
    for k, v in res["per_class_f1"].items():
        print(f"    {k:<14} {v:.3f}")
    md += ["Per-class F1, 5-class (deployed model, bodied): "
           + ", ".join(f"{k} {v:.3f}" for k, v in res["per_class_f1"].items()) + "."]
    return md


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="mmbert_small.yaml")
    ap.add_argument("--force", action="store_true",
                    help="re-score the held-out set (test-set tuning if anything changed)")
    args = ap.parse_args()

    if RESULTS.exists() and not args.force:
        print(f"The held-out set has already been scored ({RESULTS.relative_to(ROOT)}).")
        print("Reprinting that result - nothing is re-evaluated.\n")
        print_report(json.loads(RESULTS.read_text(encoding="utf-8")))
        return 0
    if args.force and RESULTS.exists():
        print(f"{BAD}--force: re-scoring a held-out set that has already been scored.")
        print("        If anything changed since the first run, this is test-set tuning and")
        print("        the FIRST result is the one to report.\n")

    # Headline-only held-out articles are fed in as the plain headline, the way
    # the live feed will deliver them; the [NO_BODY] marker was never in training.
    # Set directly: an empty `--set` value would YAML-parse to None, not "".
    cfg = load_config(args.config)
    cfg["data"]["no_body_marker"] = ""
    seeds, deploy_seed = list(cfg["seeds"]), int(cfg["seed"])

    print("=" * 92)
    print(f"M7  final model {cfg['model']['name']}   recipe D15   seeds {seeds}   "
          f"deployed seed {deploy_seed}")
    print("=" * 92)
    device = get_device()
    mem = gpu_memory()
    if mem is None:
        print(f"{BAD}no GPU")
        return 1
    print(f"{OK if mem[0] >= 3.5 else BAD}free VRAM {mem[0]:.2f} of {mem[1]:.2f} GB")

    train = load_fold_frame("body_only").assign(role="train")
    hold = load_holdout().assign(role="holdout")
    frame = pd.concat([train, hold], ignore_index=True)
    frame["fold"] = np.where(frame["role"] == "holdout", 1, 0)
    train_idx = np.where(frame["role"] == "train")[0]
    hold_idx = np.where(frame["role"] == "holdout")[0]

    assert_disjoint(frame, train_idx, hold_idx)       # rows AND duplicate groups
    masks = subset_masks(frame, hold_idx)
    print(f"{OK}train {len(train_idx)}  |  held-out {len(hold_idx)}: "
          f"{int(masks['bodied'].sum())} with body, {int(masks['headline_only'].sum())} "
          "headline-only  |  no shared rows or duplicate groups")

    tok = load_tokenizer(cfg["model"]["name"])
    data = prepare_data(frame, tok, cfg)
    kw_names = build_features(["x"], mode=cfg["keywords"]["mode"])[1]
    y_true = {t: frame[c].to_numpy()[hold_idx] for t, c in
              {"narrative": "y_narrative", "severity": "y_severity",
               "propaganda": "y_propaganda"}.items()}

    per_seed, preds_deployed = {}, None
    for seed in seeds:
        print(f"\n  training on all {len(train_idx)} articles - seed {seed}")
        trainer, _history = train_final(data, train_idx, cfg, seed=seed, device=device)
        probs = trainer.predict_proba(data.dataset(hold_idx))
        y_pred = {t: p.argmax(axis=-1) for t, p in probs.items()}
        per_seed[seed] = score_subsets(y_true, y_pred, masks)
        m = per_seed[seed]["bodied"]
        print(f"      -> held-out bodied: narrative {m['narrative_macro_f1']:.3f}  "
              f"3-class {m['narrative_collapsed_macro_f1']:.3f}  "
              f"severity {m['severity_f1_high']:.3f}")
        if seed == deploy_seed:
            path = save_model(trainer.model, cfg, OUT / f"model_seed{seed}", seed=seed,
                              n_train=len(train_idx), keyword_feature_names=kw_names)
            print(f"      saved deployed model to {path.relative_to(ROOT)}")
            preds_deployed = predictions_frame(frame, hold_idx, probs, fold=-1, seed=seed)
        del trainer
        _free_memory()

    print("\n  scoring baselines on the same split ...")
    baselines = baseline_scores(frame, train_idx, hold_idx, masks)

    scalar = [k for k, _ in METRICS] + ["propaganda_f1_pos", "propaganda_n_pos"]
    encoder = {"mean": {}, "std": {}, "deployed": per_seed[deploy_seed], "per_seed": per_seed}
    for subset in SUBSETS:
        if subset not in per_seed[deploy_seed]:
            continue
        encoder["mean"][subset] = {k: float(np.mean([per_seed[s][subset][k] for s in seeds]))
                                   for k in scalar}
        encoder["std"][subset] = {k: float(np.std([per_seed[s][subset][k] for s in seeds]))
                                  for k in scalar}

    bodied_preds = preds_deployed[preds_deployed["has_body"].astype(bool)]
    cm = confusion_frame(collapse_narrative(bodied_preds["true_narrative"]),
                         collapse_narrative(bodied_preds["pred_narrative"]))
    # plain ints: json would silently turn numpy ints into strings
    cm_dict = {str(c): {str(r): int(cm.loc[r, c]) for r in cm.index} for c in cm.columns}
    res = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M"),
        "model": cfg["model"]["name"],
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(hold_idx)),
        "subset_n": {k: int(v.sum()) for k, v in masks.items()},
        "encoder": encoder,
        "baselines": baselines,
        "error_analysis": error_analysis(bodied_preds, hold),
        "confusion_3class": cm_dict,
        "per_class_f1": per_class_f1(bodied_preds["true_narrative"],
                                     bodied_preds["pred_narrative"]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    preds_deployed.merge(hold[["id", "clean_headline", "narrative_confidence",
                               "agree_narrative", "label_status", "llama_narrative"]],
                         on="id").to_csv(OUT / "holdout_predictions.csv", index=False)
    RESULTS.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    md = print_report(res)
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  written: {RESULTS.relative_to(ROOT)}, runs/m7/holdout_predictions.csv, "
          f"{REPORT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
