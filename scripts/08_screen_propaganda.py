#!/usr/bin/env python
"""M8.3 - Llama screens every article for propaganda (D18).

Uses the definitions in docs/PROPAGANDA_GUIDE.md - not the keyword list, whose
propaganda terms largely reproduce the 7 original positives (D18) - so it can
find kinds of propaganda the rubric never named. Its output only nominates
articles for human review; it does not create labels.

    python scripts/08_screen_propaganda.py --only check    # 7 originals + rubric hits (~1 min)
    python scripts/08_screen_propaganda.py                 # all 1,064 (~15-30 min)

`--only check` is the calibration test: articles where a working screener
should say yes to at least some. It prints every answer with its reasoning.

Resumable: each answer is appended to runs/m8/screen.jsonl as it arrives, and a
rerun skips articles already screened. Close browsers first - Llama shares the
GPU with the display. Do not train a model at the same time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from osint_shield.data.loaders import load_gold  # noqa: E402
from osint_shield.keywords import KeywordMatcher  # noqa: E402
from osint_shield.llm import (  # noqa: E402
    PROMPT_VERSION,
    PROPAGANDA_SCHEMA,
    PROPAGANDA_SYSTEM,
    OllamaClient,
    OllamaError,
    evidence_found,
    propaganda_prompt,
)

OUT = ROOT / "runs" / "m8"
JSONL = OUT / "screen.jsonl"
CSV = OUT / "screen.csv"
CSV_PARTIAL = OUT / "screen_partial.csv"


def load_done() -> dict[str, dict]:
    """Answers already on disk from the CURRENT prompt version - older ones are ignored."""
    if not JSONL.exists():
        return {}
    done = {}
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("prompt_version") == PROMPT_VERSION:
                done[rec["id"]] = rec          # later lines override earlier ones
    return done


def check_subset(df: pd.DataFrame) -> pd.DataFrame:
    """The 7 original positives plus every strict-rubric hit: where a working
    screener should answer yes to at least some."""
    strict = KeywordMatcher(exclude_low_precision=True)
    hit = df.apply(lambda r: any(
        g.startswith("prop_") for g in strict.hits(
            f"{r['clean_headline']} {r['clean_text'] if isinstance(r['clean_text'], str) else ''}")),
        axis=1)
    return df[hit | (df["y_propaganda"] == 1)]


def show_answers(df: pd.DataFrame, done: dict) -> None:
    """Print each answer with its reasoning - for eyeballing a calibration run."""
    print("\n" + "=" * 80)
    print("  ANSWERS  (o = one of the 7 original positives)")
    print("=" * 80)
    for _, r in df.iterrows():
        rec = done.get(r["id"], {})
        mark = "o" if r["y_propaganda"] == 1 else " "
        verdict = ("YES" if rec.get("propaganda") else "no ") + f" s{rec.get('strength', '?')}"
        print(f"\n  {mark} [{verdict}] {str(r['clean_headline'])[:90]}")
        print(f"        analysis: {str(rec.get('analysis', ''))[:180]}")
        if rec.get("propaganda"):
            found = "found" if rec.get("evidence_found") else "NOT in text"
            print(f"        evidence ({found}): {str(rec.get('evidence', ''))[:120]}")


def keyword_hits(text: str, matcher: KeywordMatcher) -> str:
    hits = {g: t for g, t in matcher.hits(text).items() if g.startswith("prop_")}
    return "; ".join(f"{g[5:]}: {', '.join(t)}" for g, t in hits.items())


def summarise(df: pd.DataFrame) -> None:
    flagged = df["llama_propaganda"].fillna(False).astype(bool)
    print("\n" + "=" * 80)
    print(f"  SCREEN SUMMARY - {len(df)} articles")
    print("=" * 80)
    print(f"  flagged as propaganda by Llama: {int(flagged.sum())} "
          f"({flagged.mean() * 100:.1f}%)")
    print("  strength of flagged:      " +
          str(df.loc[flagged, "llama_strength"].value_counts().sort_index().to_dict()))
    if flagged.any():
        ok = df.loc[flagged, "evidence_found"].mean()
        print(f"  flagged with a quote actually found in the text: {ok * 100:.0f}% "
              "(the rest are paraphrased or invented)")
    errors = int(df["error"].notna().sum())
    if errors:
        print(f"  errors (no valid answer): {errors} - rerun to retry them")
    orig = df["y_propaganda"] == 1
    print(f"\n  the 7 ORIGINAL positives: Llama flags {int((flagged & orig).sum())}")
    print("    (diagnostic only - under the own-voice rule some originals may not qualify)")
    kw_strict = df["kw_strict"].fillna("") != ""
    print(f"  rubric hits (low-precision terms removed): {int(kw_strict.sum())}; "
          f"also flagged by Llama: {int((kw_strict & flagged).sum())}")
    print(f"  flagged by Llama but NOT by the rubric: {int((flagged & ~kw_strict).sum())}"
          "  <- what the keyword list could never find")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="screen only the first N (smoke test)")
    ap.add_argument("--only", choices=["check"], default=None,
                    help="'check' = the 7 originals + strict rubric hits, answers printed")
    ap.add_argument("--model", default=None, help="Ollama model tag (default llama3.2:3b)")
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--retry-errors", action="store_true", help="re-screen articles that errored")
    args = ap.parse_args()

    client = OllamaClient(model=args.model)
    try:
        available = client.models()
    except OllamaError as exc:
        print(f"  [!!] {exc}\n       start Ollama (open the app, or run `ollama serve`)")
        return 1
    if client.model not in available:
        print(f"  [!!] {client.model} is not pulled. Available: {available}")
        return 1

    df = load_gold().reset_index(drop=True)
    if args.only == "check":
        df = check_subset(df)
    if args.limit:
        df = df.head(args.limit)
    partial = bool(args.only or args.limit)
    done = load_done()
    if args.retry_errors:
        done = {k: v for k, v in done.items() if not v.get("error")}
    todo = df[~df["id"].isin(done)]

    print("=" * 80)
    print(f"M8.3  Llama propaganda screen   model {client.model}   prompt {PROMPT_VERSION}")
    print(f"      articles {len(df)}   already done {len(df) - len(todo)}   to do {len(todo)}")
    print("=" * 80)

    OUT.mkdir(parents=True, exist_ok=True)
    t_start, times = time.time(), []
    try:
        with JSONL.open("a", encoding="utf-8") as fh:
            for i, (_, row) in enumerate(todo.iterrows(), start=1):
                body = row["clean_text"] if isinstance(row["clean_text"], str) else ""
                t0 = time.time()
                rec = {"id": row["id"], "prompt_version": PROMPT_VERSION}
                try:
                    ans = client.generate_json(
                        propaganda_prompt(row["clean_headline"], body, max_chars=args.max_chars),
                        PROPAGANDA_SCHEMA, system=PROPAGANDA_SYSTEM)
                    rec.update({
                        "analysis": str(ans.get("analysis", "") or ""),
                        "own_voice": bool(ans.get("own_voice", False)),
                        "propaganda": bool(ans.get("propaganda", False)),
                        "types": [t for t in ans.get("types", []) if isinstance(t, str)],
                        "strength": int(ans.get("strength", 0) or 0),
                        "evidence": str(ans.get("evidence", "") or ""),
                    })
                    rec["evidence_found"] = evidence_found(rec["evidence"],
                                                           row["clean_headline"], body)
                except OllamaError as exc:
                    if "cannot reach" in str(exc):
                        raise
                    rec["error"] = str(exc)[:200]
                rec["seconds"] = round(time.time() - t0, 2)
                times.append(rec["seconds"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[rec["id"]] = rec
                if i % 10 == 0 or i == len(todo):
                    left = (len(todo) - i) * (sum(times) / len(times)) / 60
                    n_flag = sum(1 for r in done.values() if r.get("propaganda"))
                    print(f"  {i:4d}/{len(todo)}   {sum(times) / len(times):.1f}s/article   "
                          f"~{left:.0f} min left   flagged so far {n_flag}")
    except KeyboardInterrupt:
        print("\n  interrupted - every answer so far is saved. Rerun to continue.")
        return 130
    except OllamaError as exc:
        print(f"\n  [!!] {exc}\n       answers so far are saved. Restart Ollama and rerun.")
        return 1

    strict, full = KeywordMatcher(exclude_low_precision=True), KeywordMatcher()
    rows = []
    for _, row in df.iterrows():
        rec = done.get(row["id"], {})
        text = f"{row['clean_headline']} {row['clean_text'] if isinstance(row['clean_text'], str) else ''}"
        rows.append({
            "id": row["id"], "split": row["split_orig"], "has_body": row["has_body"],
            "headline": row["clean_headline"], "y_propaganda": int(row["y_propaganda"]),
            "llama_propaganda": rec.get("propaganda"), "llama_strength": rec.get("strength"),
            "llama_types": ",".join(rec.get("types", [])), "llama_evidence": rec.get("evidence"),
            "llama_own_voice": rec.get("own_voice"), "llama_analysis": rec.get("analysis"),
            "evidence_found": rec.get("evidence_found"), "error": rec.get("error"),
            "kw_all": keyword_hits(text, full), "kw_strict": keyword_hits(text, strict),
        })
    out = pd.DataFrame(rows)
    target = CSV_PARTIAL if partial else CSV       # a subset never overwrites the full table
    out.to_csv(target, index=False)
    if args.only == "check":
        show_answers(df, done)
    summarise(out)
    print(f"\n  screened in {(time.time() - t_start) / 60:.1f} min this run   ->   "
          f"{target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
