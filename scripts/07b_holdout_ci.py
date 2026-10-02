#!/usr/bin/env python
"""M7b - confidence intervals for the held-out comparison.

Paired bootstrap over the held-out articles: is the encoder's gap to TF-IDF and
to the keyword rubric real, or within sampling noise?

Does NOT re-score the encoder - it reads the deployed model's saved held-out
predictions (runs/m7/holdout_predictions.csv). TF-IDF and the rubric are
deterministic and refit exactly as 07_final_eval.py did. So this quantifies the
uncertainty of the one evaluation; it is not a second look at the test set.

    python scripts/07b_holdout_ci.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from osint_shield.data.loaders import build_text, load_fold_frame, load_gold  # noqa: E402
from osint_shield.evaluation.baselines import build_baseline  # noqa: E402
from osint_shield.evaluation.metrics import binary_f1, collapse_narrative, macro_f1  # noqa: E402
from osint_shield.paths import DATA_PROCESSED  # noqa: E402

PRED = ROOT / "runs" / "m7" / "holdout_predictions.csv"
OUT = ROOT / "runs" / "m7" / "bootstrap_ci.json"
B, SEED = 4000, 0


def main() -> int:
    if not PRED.exists():
        print("run scripts/07_final_eval.py first")
        return 1
    enc = pd.read_csv(PRED).set_index("id")
    hold = pd.read_csv(DATA_PROCESSED / "holdout_all.csv").merge(
        load_gold()[["id", "clean_headline", "clean_text"]], on="id")
    train = load_fold_frame("body_only")
    tr = train.assign(text=build_text(train, sep=" ", no_body_marker=""))
    ho = hold.assign(text=build_text(hold, sep=" ", no_body_marker=""))

    def baseline(name: str, col: str, family: str, df: pd.DataFrame) -> np.ndarray:
        model = build_baseline(name, family).fit(tr, tr[col].to_numpy())
        return np.asarray(model.predict(df)).astype(str)

    results: dict = {}
    for subset, mask in [("bodied", ho["has_body"].astype(bool)),
                         ("headline_only", ~ho["has_body"].astype(bool))]:
        h = ho[mask.to_numpy()].reset_index(drop=True)
        e = enc.loc[h["id"]]
        preds = {
            "encoder": (e["pred_narrative"].to_numpy(), e["pred_severity"].to_numpy()),
            "tfidf": (baseline("tfidf_logreg", "narrative", "narrative", h),
                      baseline("tfidf_logreg", "severity", "severity", h)),
            "rubric": (baseline("keyword_rules", "narrative", "narrative", h),
                       baseline("keyword_rules", "severity", "severity", h)),
        }
        yn, ys = h["narrative"].to_numpy(), h["severity"].to_numpy()

        def score(p, idx):
            return {
                "narrative_5": macro_f1(yn[idx], p[0][idx]),
                "narrative_3": macro_f1(collapse_narrative(yn[idx]), collapse_narrative(p[0][idx])),
                "severity_high": binary_f1(ys[idx], p[1][idx], "High"),
            }

        full = np.arange(len(h))
        point = {k: score(p, full) for k, p in preds.items()}
        rng = np.random.default_rng(SEED)
        boot = {k: {m: [] for m in point["encoder"]} for k in preds}
        for _ in range(B):
            idx = rng.integers(0, len(h), len(h))
            for k, p in preds.items():
                for m, v in score(p, idx).items():
                    boot[k][m].append(v)

        print(f"\n===== {subset}  n={len(h)}   classes {h['narrative'].value_counts().to_dict()}"
              f"   High {(h['severity'] == 'High').sum()}")
        res: dict = {"n": len(h)}
        for m in point["encoder"]:
            res[m] = {}
            print(f"\n  {m}")
            for k in preds:
                arr = np.array(boot[k][m])
                lo, hi = np.percentile(arr, [2.5, 97.5])
                res[m][k] = {"point": round(point[k][m], 4), "ci": [round(lo, 4), round(hi, 4)]}
                print(f"    {k:<8} {point[k][m]:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
            for other in ("tfidf", "rubric"):
                d = np.array(boot["encoder"][m]) - np.array(boot[other][m])
                lo, hi = np.percentile(d, [2.5, 97.5])
                res[m][f"encoder_minus_{other}"] = {
                    "point": round(point["encoder"][m] - point[other][m], 4),
                    "ci": [round(lo, 4), round(hi, 4)],
                    "p_encoder_better": round(float(np.mean(d > 0)), 3)}
                print(f"    encoder - {other:<7} {point['encoder'][m] - point[other][m]:+.3f}"
                      f"   95% CI [{lo:+.3f}, {hi:+.3f}]   P(encoder better) {np.mean(d > 0):.2f}")
        results[subset] = res

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
