#!/usr/bin/env python
"""M8.4 - build the blind propaganda review sheet (D18, D19).

    python scripts/08b_build_review_sheet.py

Writes two files to data/processed/ (gitignored - they hold article text):

  propaganda_review.csv      YOUR sheet: open it in Excel and fill label / types
                             / sure / evidence / notes, using docs/PROPAGANDA_GUIDE.md
  propaganda_review_KEY.csv  why each row is in the sheet - do NOT open it until
                             you have finished labelling

Refuses to overwrite a sheet you have started filling in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from osint_shield.data.loaders import load_gold  # noqa: E402
from osint_shield.keywords import KeywordMatcher  # noqa: E402
from osint_shield.keywords.mining import assign_buckets, blind_sheet  # noqa: E402
from osint_shield.paths import DATA_PROCESSED  # noqa: E402

SHEET = DATA_PROCESSED / "propaganda_review.csv"
KEY = DATA_PROCESSED / "propaganda_review_KEY.csv"
CALIBRATION = ROOT / "runs" / "m8" / "screen_partial.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite a sheet already started")
    args = ap.parse_args()

    if SHEET.exists() and not args.force:
        started = pd.read_csv(SHEET, encoding="utf-8-sig", dtype=str).fillna("")
        if (started["label"].str.strip() != "").any():
            print(f"  [!!] {SHEET.name} already has labels in it - not overwriting your work.")
            print("       (--force overwrites it; you would lose those labels)")
            return 1

    df = load_gold().reset_index(drop=True)
    chosen = assign_buckets(df, budget=args.budget, seed=args.seed)
    # the calibration articles were shown to the reviewer with verdicts - mark them
    seen = set(pd.read_csv(CALIBRATION)["id"]) if CALIBRATION.exists() else set()
    chosen = chosen.assign(not_blind=chosen["id"].isin(seen))
    sheet, key = blind_sheet(chosen, seed=args.seed)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel shows curly quotes and Devanagari correctly
    sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")
    key.to_csv(KEY, index=False, encoding="utf-8-sig")

    counts = key["bucket"].value_counts().to_dict()
    print("=" * 76)
    print(f"M8.4  blind review sheet - {len(sheet)} articles")
    print("=" * 76)
    print(f"  buckets (hidden from the sheet): {counts}")
    print(f"  rubric terms used: all, including low-precision ones")
    print(f"  not blind (seen during calibration): {int(key['not_blind'].sum())}")
    print(f"  with body text: {int((sheet['article'] != '(headline only)').sum())}, "
          f"headline only: {int((sheet['article'] == '(headline only)').sum())}")
    print(f"\n  YOUR SHEET:  {SHEET.relative_to(ROOT)}")
    print(f"  key (don't open yet): {KEY.relative_to(ROOT)}")
    print("\n  Fill in, per row:  label = yes/no   types = glorification, justification,")
    print("  victimhood, recruitment   sure = sure/unsure   evidence = exact words   notes")
    print("  Rules: docs/PROPAGANDA_GUIDE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
