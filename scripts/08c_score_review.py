#!/usr/bin/env python
"""M8.5 - what the reviewer's labels say about the rubric and the corpus (D18, D19).

Joins the labelled sheet to its bucket key (safe now that labelling is done) and
reports, with 95% Wilson intervals:

  * originals  - how many of the 7 first-pass positives survive the own-voice rule
  * rubric     - precision: of the articles a rubric term fires on, how many are propaganda
  * random     - prevalence among the articles NEITHER flagged, which estimates how
                 many positives the rubric misses across the whole corpus

Also writes data/resources/propaganda_v2_labels.csv (ids and labels only, no
article text - safe to commit).

    python scripts/08c_score_review.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from osint_shield.data.loaders import load_gold  # noqa: E402
from osint_shield.keywords.mining import rubric_hits  # noqa: E402
from osint_shield.paths import DATA_PROCESSED, RESOURCES  # noqa: E402

LABELLED = [DATA_PROCESSED / "propaganda_review_labeled.csv",
            DATA_PROCESSED / "propaganda_review.csv"]
KEY = DATA_PROCESSED / "propaganda_review_KEY.csv"
LABELS_OUT = RESOURCES / "propaganda_v2_labels.csv"
OUT = ROOT / "runs" / "m8" / "review_scores.json"
TYPES = {"glorification", "justification", "victimhood", "recruitment"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval - well behaved at 0/n and for small n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def read_sheet() -> pd.DataFrame:
    for path in LABELLED:
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        if (df["label"].str.strip() != "").any():
            print(f"  labels read from {path.relative_to(ROOT)}")
            return df
    raise SystemExit("no labelled sheet found in data/processed/")


def normalise_types(raw: str) -> str:
    found = [t.strip().lower() for t in str(raw).replace(";", ",").split(",")]
    return ",".join(sorted(t for t in found if t in TYPES))


def rate_line(name: str, sub: pd.DataFrame) -> dict:
    n, k = len(sub), int(sub["yes"].sum())
    ks = int((sub["yes"] & sub["is_sure"]).sum())
    lo, hi = wilson(k, n)
    print(f"  {name:<10} n={n:<4} yes {k:<3} ({k / n * 100 if n else 0:5.1f}%, 95% CI "
          f"{lo * 100:4.1f}-{hi * 100:4.1f}%)   of which sure: {ks}")
    return {"n": n, "yes": k, "yes_sure": ks, "rate": k / n if n else 0.0, "ci": [lo, hi]}


def main() -> int:
    sheet = read_sheet()
    key = pd.read_csv(KEY, encoding="utf-8-sig")
    df = sheet.merge(key[["id", "bucket", "rubric_terms", "y_propaganda", "not_blind"]],
                     on="id", how="left", validate="one_to_one")
    if df["bucket"].isna().any():
        raise SystemExit("some sheet rows are missing from the key - was the sheet rebuilt?")

    df["label"] = df["label"].str.strip().str.lower()
    bad = sorted(set(df["label"]) - {"yes", "no"})
    if bad:
        raise SystemExit(f"labels must be yes/no; found {bad}")
    df["yes"] = df["label"] == "yes"
    df["is_sure"] = df["sure"].str.strip().str.lower() == "sure"
    df["types_clean"] = df["types"].map(normalise_types)
    df["not_blind"] = df["not_blind"].astype(str).str.lower() == "true"

    print("=" * 84)
    print(f"M8.5  scoring the review - {len(df)} articles, {int(df['yes'].sum())} labelled yes")
    print("=" * 84)
    res: dict = {"n_reviewed": len(df), "n_yes": int(df["yes"].sum()),
                 "n_yes_sure": int((df["yes"] & df["is_sure"]).sum())}

    print("\n  by bucket:")
    res["buckets"] = {b: rate_line(b, df[df["bucket"] == b])
                      for b in ("original", "rubric", "random")}

    # ---- the originals under the own-voice rule ----------------------------
    orig = df[df["bucket"] == "original"]
    print(f"\n  original positives that survive the own-voice rule: "
          f"{int(orig['yes'].sum())} of {len(orig)}")
    for _, r in orig[~orig["yes"]].iterrows():
        print(f"    now 'no': {r['headline'][:80]}")

    # ---- rubric precision, all and blind-only ------------------------------
    rub = df[df["bucket"] == "rubric"]
    flagged = df[df["rubric_terms"].fillna("") != ""]           # includes originals it fires on
    k, n = int(flagged["yes"].sum()), len(flagged)
    lo, hi = wilson(k, n)
    print(f"\n  rubric precision (every reviewed article a term fires on): "
          f"{k}/{n} = {k / n * 100:.1f}%  (95% CI {lo * 100:.1f}-{hi * 100:.1f}%)")
    blind = flagged[~flagged["not_blind"]]
    kb, nb = int(blind["yes"].sum()), len(blind)
    print(f"    blind rows only (calibration articles excluded): {kb}/{nb}")
    res["rubric_precision"] = {"yes": k, "n": n, "ci": [lo, hi], "blind_yes": kb, "blind_n": nb}

    # ---- prevalence among unflagged articles -> what the rubric misses -----
    gold = load_gold()
    gold["terms"] = rubric_hits(gold)
    n_unflagged = int(((gold["terms"] == "") & (gold["y_propaganda"] == 0)).sum())
    rnd = res["buckets"]["random"]
    est, est_lo, est_hi = (rnd["rate"] * n_unflagged, rnd["ci"][0] * n_unflagged,
                           rnd["ci"][1] * n_unflagged)
    n_flagged_corpus = int((gold["terms"] != "").sum())
    found_by_rubric = res["rubric_precision"]["yes"]
    print(f"\n  corpus: {len(gold)} articles; a rubric term fires on {n_flagged_corpus}; "
          f"{n_unflagged} have no rubric term and no original label")
    print(f"  propaganda the rubric MISSES, estimated from the random sample:")
    print(f"    {rnd['rate'] * 100:.1f}% of {n_unflagged} = ~{est:.0f} articles "
          f"(95% CI {est_lo:.0f}-{est_hi:.0f})")
    if found_by_rubric + est > 0:
        recall = found_by_rubric / (found_by_rubric + est)
        print(f"    -> rubric recall roughly {recall * 100:.0f}% "
              f"(it finds {found_by_rubric}; ~{est:.0f} more exist that it cannot see)")
        res["rubric_recall_estimate"] = recall
    res["missed_estimate"] = {"rate": rnd["rate"], "n_unflagged": n_unflagged,
                              "count": est, "ci": [est_lo, est_hi]}

    # ---- types and certainty -----------------------------------------------
    types = df.loc[df["yes"], "types_clean"].str.split(",").explode().value_counts().to_dict()
    print(f"\n  techniques among the yes labels: {types}")
    print(f"  yes labels marked sure: {res['n_yes_sure']} of {res['n_yes']}")
    res["types"] = types

    # ---- the new label set: ids and labels only, no article text -----------
    RESOURCES.mkdir(parents=True, exist_ok=True)
    out = df[["id", "label", "types_clean", "sure", "bucket"]].rename(
        columns={"types_clean": "types"})
    out["sure"] = out["sure"].str.strip().str.lower()
    out.sort_values("id").to_csv(LABELS_OUT, index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  written: {LABELS_OUT.relative_to(ROOT)} (commit this - ids and labels only), "
          f"{OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
