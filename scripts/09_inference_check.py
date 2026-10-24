#!/usr/bin/env python
"""M9 - check the inference service before serving it.

  1. Calibration   is the deployed recipe's confidence trustworthy, and what does
                   each escalation threshold cost? Measured on development
                   (cross-validation out-of-fold) predictions - never test.csv.
  2. Latency       single-article and batched timings on the deployed model,
                   against the plan's ~20 ms per article (§6.1).
  3. Consistency   the same article alone and inside a batch must get the same labels.
  4. Examples      what the output records actually look like.

    python scripts/09_inference_check.py

~1 minute. Close browsers first - the GPU also drives the display.
"""

from __future__ import annotations

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

from osint_shield.config import load_config  # noqa: E402
from osint_shield.data.loaders import load_fold_frame  # noqa: E402
from osint_shield.inference import (  # noqa: E402
    ArticleClassifier,
    expected_calibration_error,
    reliability_table,
    threshold_sweep,
)
from osint_shield.paths import resolve  # noqa: E402

OUT = ROOT / "runs" / "m9"
TARGET_MS = 20.0
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
OK, BAD, INFO = "  [ok]  ", "  [!!]  ", "  [..]  "


def section(title: str) -> None:
    print("\n" + "=" * 84)
    print(f"  {title}")
    print("=" * 84)


def main() -> int:
    cfg = load_config("serve.yaml")
    report: dict = {}

    # ---- 1. calibration on development predictions -------------------------
    oof = pd.read_csv(resolve(cfg["calibration_oof"]))
    section(f"1. CALIBRATION - {len(oof)} out-of-fold predictions (D15 recipe, development data)")
    ok_n = oof["true_narrative"] == oof["pred_narrative"]
    sev_conf = np.maximum(oof["p_high"], 1 - oof["p_high"])
    ok_s = oof["true_severity"] == oof["pred_severity"]
    for name, conf, ok in (("narrative", oof["conf_narrative"], ok_n),
                           ("severity", sev_conf, ok_s)):
        table = reliability_table(conf, ok)
        ece = expected_calibration_error(conf, ok)
        print(f"\n  {name}: expected calibration error {ece:.3f}  "
              "(0 = confidence matches accuracy; gap > 0 = over-confident)")
        print("  " + table.round(3).to_string(index=False).replace("\n", "\n  "))
        report[f"{name}_ece"] = ece
        report[f"{name}_reliability"] = table.to_dict("records")

    sweep = threshold_sweep(oof, THRESHOLDS)
    print("\n  escalation threshold trade-off  (escalate = predicted High OR confidence < t)")
    print("  " + sweep.round(3).to_string(index=False).replace("\n", "\n  "))
    high_share = float((oof["pred_severity"] == "High").mean())
    print(f"\n{INFO}predicted High alone escalates {high_share * 100:.0f}% - "
          "a floor no threshold can go below")
    report["threshold_sweep"] = sweep.to_dict("records")
    report["predicted_high_share"] = high_share

    # ---- 2-4. the deployed model --------------------------------------------
    section("2. LATENCY - deployed model")
    clf = ArticleClassifier.from_config(cfg)
    print(f"{INFO}model {clf.model_version} on {clf.predictor.device}   "
          f"escalate_below {clf.routing.escalate_below}")
    dev = load_fold_frame("body_only").sample(64, random_state=0)
    articles = [{"id": r["id"], "headline": r["clean_headline"], "body": r["clean_text"]}
                for _, r in dev.iterrows()]
    articles += [{"id": f"{a['id']}_h", "headline": a["headline"]} for a in articles[:16]]

    clf.classify(articles[:8])                      # warm-up: CUDA kernels, tokenizer
    single = []
    for a in articles[:64]:
        t0 = time.perf_counter()
        clf.classify([a])
        single.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    batch_records = clf.classify(articles)
    batch_ms = (time.perf_counter() - t0) * 1000 / len(articles)
    p50, p95 = np.percentile(single, [50, 95])
    print(f"{OK if p50 <= TARGET_MS else INFO}one article per call: p50 {p50:.1f} ms, "
          f"p95 {p95:.1f} ms   (plan target ~{TARGET_MS:.0f} ms)")
    print(f"{OK if batch_ms <= TARGET_MS else INFO}batched ({len(articles)} at once): "
          f"{batch_ms:.1f} ms per article")
    report["latency_ms"] = {"single_p50": p50, "single_p95": p95, "batched_per_article": batch_ms}

    # ---- 3. consistency ----------------------------------------------------
    section("3. CONSISTENCY - same article alone vs inside a batch")
    single_records = [clf.classify([a])[0] for a in articles]
    mismatches, max_diff = 0, 0.0
    for s, b in zip(single_records, batch_records):
        for task in ("narrative", "severity"):
            if s[task]["label"] != b[task]["label"]:
                mismatches += 1
            if s[task]["confidence"] is not None and b[task]["confidence"] is not None:
                max_diff = max(max_diff, abs(s[task]["confidence"] - b[task]["confidence"]))
    print(f"{OK if mismatches == 0 else BAD}label mismatches: {mismatches} of "
          f"{2 * len(articles)}   largest confidence difference {max_diff:.4f}")
    print("        (tiny differences are bf16 rounding from different padding - labels must agree)")
    report["consistency"] = {"label_mismatches": mismatches, "max_confidence_diff": max_diff}

    # ---- routing on this sample ---------------------------------------------
    bodied = [r for r in batch_records if r["has_body"]]
    headline = [r for r in batch_records if not r["has_body"]]
    esc = np.mean([r["escalated"] for r in bodied])
    print(f"\n{INFO}on this sample: {esc * 100:.0f}% of bodied articles escalate; "
          f"{sum(r['narrative']['source'] == 'rubric' for r in headline)} of {len(headline)} "
          "headline-only narratives decided by the rubric; all headline-only need review: "
          f"{all(r['needs_review'] for r in headline)}")

    # ---- 4. examples ----------------------------------------------------------
    section("4. EXAMPLE RECORDS")
    picks = [next((r for r in bodied if not r["escalated"]), None),
             next((r for r in bodied if r["escalated"]), None),
             headline[0] if headline else None]
    for r in [p for p in picks if p]:
        shown = {**r, "headline": r["headline"][:70]}
        print(json.dumps(shown, indent=2, ensure_ascii=False))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inference_check.json").write_text(json.dumps(report, indent=2, default=float),
                                              encoding="utf-8")
    print(f"\n  written to {(OUT / 'inference_check.json').relative_to(ROOT)}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
