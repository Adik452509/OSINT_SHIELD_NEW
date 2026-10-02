#!/usr/bin/env python
"""M7b - how should the live pipeline classify articles that arrive with no body?

Google-News items reach us as a headline only. The model trained on bodied
articles. On the 65 held-out headline-only articles the keyword rubric looked
stronger (0.528 vs 0.276 narrative) - but too few to decide (CI [-0.41, +0.14]).

This answers it on development data, never test.csv: the ~415 headline-only
articles in the development pool, which the bodied-only model never trained on.
Any whose duplicate group also appears in the 476 training articles is dropped
first - a syndicated headline of a story the model trained on would flatter it.

Decision rule, pre-registered in docs/PROGRESS.md before this script ran:
  the encoder is the default; for headline-only articles a task switches to an
  alternative only if it beats the encoder with paired-bootstrap P >= 0.90 on the
  primary metric (narrative 5-class, not worse on 3-class / severity F1(High)).

    python scripts/07c_headline_routing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.runtime import configure_environment  # noqa: E402

configure_environment()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from osint_shield.data.loaders import build_text, load_fold_frame  # noqa: E402
from osint_shield.evaluation.baselines import build_baseline  # noqa: E402
from osint_shield.evaluation.metrics import binary_f1, collapse_narrative, macro_f1  # noqa: E402
from osint_shield.inference import Predictor  # noqa: E402
from osint_shield.keywords import KeywordMatcher  # noqa: E402

MODEL_DIR = ROOT / "runs" / "m7" / "model_seed42"
HOLDOUT_PRED = ROOT / "runs" / "m7" / "holdout_predictions.csv"
OUT = ROOT / "runs" / "m7b"
B, SEED = 2000, 0
P_SWITCH = 0.90          # pre-registered
ESCALATE_BELOW = 0.70    # plan §6.2 starting threshold


def narr_scores(y, p, idx):
    return {"narrative_5": macro_f1(y[idx], p[idx]),
            "narrative_3": macro_f1(collapse_narrative(y[idx]), collapse_narrative(p[idx]))}


def main() -> int:
    if not MODEL_DIR.exists():
        print("deployed model not found - run scripts/07_final_eval.py first")
        return 1

    train = load_fold_frame("body_only")
    pool = load_fold_frame("all")
    ho = pool[~pool["has_body"].astype(bool)]
    n_before = len(ho)
    overlap = ho["dupe_group"].isin(set(train["dupe_group"]))
    ho = ho[~overlap].reset_index(drop=True)

    print("=" * 84)
    print("M7b  headline-only routing study  (development data - test.csv not used)")
    print("=" * 84)
    print(f"  headline-only development articles: {n_before}; dropped {int(overlap.sum())} that "
          f"duplicate a training story -> n = {len(ho)}")
    print(f"  narrative: {ho['narrative'].value_counts().to_dict()}")
    print(f"  severity High: {int((ho['severity'] == 'High').sum())} "
          f"({(ho['severity'] == 'High').mean() * 100:.1f}%)")

    # ---- encoder: the deployed model, fed the plain headline ---------------
    predictor = Predictor.from_dir(MODEL_DIR)
    print(f"\n  deployed model on {predictor.device}; classifying {len(ho)} headlines ...")
    enc = predictor.predict(ho)
    enc_n, enc_s = enc["narrative"].to_numpy(), enc["severity"].to_numpy()

    # ---- baselines fit on the same 476 bodied articles ---------------------
    tr = train.assign(text=build_text(train, sep=" ", no_body_marker=""))
    hd = ho.assign(text=build_text(ho, sep=" ", no_body_marker=""))

    def fit_pred(name: str, col: str, family: str) -> np.ndarray:
        model = build_baseline(name, family).fit(tr, tr[col].to_numpy())
        return np.asarray(model.predict(hd)).astype(str)

    tfidf_n, tfidf_s = fit_pred("tfidf_logreg", "narrative", "narrative"), \
        fit_pred("tfidf_logreg", "severity", "severity")
    rub_n, rub_s = fit_pred("keyword_rules", "narrative", "narrative"), \
        fit_pred("keyword_rules", "severity", "severity")

    matcher = KeywordMatcher()
    fires = np.array([any(n > 0 for g, n in matcher.counts(t).items() if g.startswith("narr_"))
                      for t in hd["text"]])
    hybrid_n = np.where(fires, rub_n, enc_n)
    print(f"  a narrative rubric keyword fires on {fires.mean() * 100:.0f}% of headlines")

    yn, ys = ho["narrative"].to_numpy(), ho["severity"].to_numpy()
    narr = {"encoder": enc_n, "tfidf": tfidf_n, "rubric": rub_n, "hybrid": hybrid_n}
    sev = {"encoder": enc_s, "tfidf": tfidf_s, "rubric": rub_s}

    # ---- point estimates + paired bootstrap against the encoder ------------
    full = np.arange(len(ho))
    point_n = {k: narr_scores(yn, p, full) for k, p in narr.items()}
    point_s = {k: binary_f1(ys, p, "High") for k, p in sev.items()}
    rng = np.random.default_rng(SEED)
    wins_n = {k: 0 for k in narr}
    wins_s = {k: 0 for k in sev}
    for _ in range(B):
        idx = rng.integers(0, len(ho), len(ho))
        base_n = macro_f1(yn[idx], enc_n[idx])
        base_s = binary_f1(ys[idx], enc_s[idx], "High")
        for k, p in narr.items():
            wins_n[k] += macro_f1(yn[idx], p[idx]) > base_n
        for k, p in sev.items():
            wins_s[k] += binary_f1(ys[idx], p[idx], "High") > base_s
    p_n = {k: v / B for k, v in wins_n.items()}
    p_s = {k: v / B for k, v in wins_s.items()}

    print("\n" + "=" * 84)
    print(f"  narrative  (n={len(ho)})")
    print(f"  {'method':<10}{'5-class':>10}{'3-class':>10}{'P(beats encoder)':>20}")
    for k in narr:
        print(f"  {k:<10}{point_n[k]['narrative_5']:>10.3f}{point_n[k]['narrative_3']:>10.3f}"
              f"{'—' if k == 'encoder' else f'{p_n[k]:.2f}':>20}")
    print(f"\n  severity F1(High)")
    for k in sev:
        print(f"  {k:<10}{point_s[k]:>10.3f}{'—' if k == 'encoder' else f'{p_s[k]:.2f}':>30}")

    # ---- the pre-registered decision ----------------------------------------
    qual_n = [k for k in narr if k != "encoder" and p_n[k] >= P_SWITCH
              and point_n[k]["narrative_3"] >= point_n["encoder"]["narrative_3"]]
    choice_n = max(qual_n, key=lambda k: point_n[k]["narrative_5"]) if qual_n else "encoder"
    qual_s = [k for k in sev if k != "encoder" and p_s[k] >= P_SWITCH]
    choice_s = max(qual_s, key=lambda k: point_s[k]) if qual_s else "encoder"

    print("\n" + "=" * 84)
    print("  DECISION (pre-registered rule: switch only if P(beats encoder) >= 0.90)")
    print("=" * 84)
    print(f"  headline-only narrative -> {choice_n.upper()}")
    print(f"  headline-only severity  -> {choice_s.upper()}")

    # ---- confidence: how often would headline-only items escalate? ---------
    conf_ho = enc["narrative_confidence"].to_numpy()
    esc_ho = (conf_ho < ESCALATE_BELOW) | (enc_s == "High")
    print(f"\n  encoder narrative confidence, headline-only: mean {conf_ho.mean():.3f}; "
          f"{(conf_ho < ESCALATE_BELOW).mean() * 100:.0f}% below {ESCALATE_BELOW}")
    print(f"  would escalate to the LLM (low confidence OR High severity): "
          f"{esc_ho.mean() * 100:.0f}%")
    summary_bodied = None
    if HOLDOUT_PRED.exists():
        hp = pd.read_csv(HOLDOUT_PRED)
        hb = hp[hp["has_body"].astype(bool)]
        conf_b = hb["conf_narrative"].to_numpy()
        esc_b = (conf_b < ESCALATE_BELOW) | (hb["pred_severity"].to_numpy() == "High")
        summary_bodied = {"mean_confidence": round(float(conf_b.mean()), 3),
                          "share_low_confidence": round(float((conf_b < ESCALATE_BELOW).mean()), 3),
                          "share_escalated": round(float(esc_b.mean()), 3)}
        print(f"  for comparison, bodied held-out: mean {conf_b.mean():.3f}; "
              f"{(conf_b < ESCALATE_BELOW).mean() * 100:.0f}% below {ESCALATE_BELOW}; "
              f"{esc_b.mean() * 100:.0f}% would escalate")

    OUT.mkdir(parents=True, exist_ok=True)
    ho.assign(encoder=enc_n, encoder_conf=conf_ho, tfidf=tfidf_n, rubric=rub_n,
              hybrid=hybrid_n, rubric_fires=fires, encoder_severity=enc_s)[
        ["id", "clean_headline", "narrative", "encoder", "encoder_conf", "rubric", "hybrid",
         "tfidf", "rubric_fires", "severity", "encoder_severity"]
    ].to_csv(OUT / "headline_only_predictions.csv", index=False)
    result = {
        "n": len(ho), "n_dropped_duplicates": int(overlap.sum()),
        "narrative": {k: {**point_n[k], "p_beats_encoder": None if k == "encoder" else p_n[k]}
                      for k in narr},
        "severity_f1_high": {k: {"point": point_s[k],
                                 "p_beats_encoder": None if k == "encoder" else p_s[k]}
                             for k in sev},
        "rubric_fire_rate": round(float(fires.mean()), 3),
        "decision": {"narrative": choice_n, "severity": choice_s},
        "escalation": {"headline_only": {
            "mean_confidence": round(float(conf_ho.mean()), 3),
            "share_low_confidence": round(float((conf_ho < ESCALATE_BELOW).mean()), 3),
            "share_escalated": round(float(esc_ho.mean()), 3)},
            "bodied_holdout": summary_bodied},
    }
    (OUT / "routing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  written: runs/m7b/routing.json, runs/m7b/headline_only_predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
