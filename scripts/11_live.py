#!/usr/bin/env python
"""M10 - the live pipeline.

    python scripts/11_live.py ingest              # poll every source once
    python scripts/11_live.py ingest --source navalnews
    python scripts/11_live.py stats               # what the database holds
    python scripts/11_live.py sample              # eyeball the relevance gate

M10a: ingestion only - articles are collected, deduplicated, gated and stored in
data/live/osint_live.db. Classification and the 15-minute loop arrive in M10b.
Settings: configs/live.yaml.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.config import load_config  # noqa: E402
from osint_shield.paths import resolve  # noqa: E402


def _store(cfg):
    from osint_shield.storage import LiveStore

    return LiveStore(resolve(cfg["live"]["db_path"]))


def cmd_ingest(cfg, args) -> int:
    from osint_shield.ingestion import HttpFetcher, RelevanceGate, load_sources, run_ingest

    live = cfg["live"]
    sources = load_sources(live)
    if args.source:
        sources = [s for s in sources if s.name == args.source]
        if not sources:
            print(f"  [!!] no source named {args.source!r} in configs/live.yaml")
            return 1

    store, gate = _store(cfg), RelevanceGate()
    print(f"  polling {len(sources)} sources -> {store.path.relative_to(ROOT)}\n")
    print(f"  {'source':34s} {'fetched':>7s} {'new':>5s} {'relevant':>8s} {'w/ body':>7s} "
          f"{'dup url':>7s} {'dup hl':>6s}")

    def show(rep):
        if rep.error:
            print(f"  {rep.source:34s} ERROR  {rep.error[:70]}")
            return
        up = f"  (+{rep.upgraded} bodies attached)" if rep.upgraded else ""
        print(f"  {rep.source:34s} {rep.fetched:7d} {rep.new:5d} {rep.relevant:8d} "
              f"{rep.with_body:7d} {rep.dup_url:7d} {rep.dup_headline:6d}{up}")

    ing = live.get("ingest", {})
    http = live.get("http", {})
    with HttpFetcher(timeout=float(http.get("timeout_s", 20)),
                     user_agent=http.get("user_agent", "Mozilla/5.0")) as fetch:
        report = run_ingest(store, sources, fetch, gate,
                            min_body_chars=int(ing.get("min_body_chars", 1000)),
                            max_items=int(ing.get("max_items_per_source", 200)),
                            blocked_domains=live.get("blocked_domains") or (),
                            on_source=show)

    t = report.totals()
    print(f"\n  {'TOTAL':34s} {t['fetched']:7d} {t['new']:5d} {t['relevant']:8d} "
          f"{t['with_body']:7d} {t['dup_url']:7d} {t['dup_headline']:6d}")
    if t["new"]:
        print(f"\n  of {t['new']} new articles, {t['relevant']} passed the relevance gate "
              f"({t['relevant'] / t['new']:.0%}); {t['with_body']} of those came with full text.")
    if t["blocked"]:
        print(f"  {t['blocked']} item(s) from blocked domains skipped.")
    if t["failed_sources"]:
        print(f"  [!] {t['failed_sources']} source(s) failed - see ERROR rows above.")
    return 0


def cmd_stats(cfg, args) -> int:
    s = _store(cfg).stats()
    print(f"  articles stored     {s['articles']}")
    print(f"  relevant            {s['relevant']}")
    print(f"    with full text    {s['relevant_with_body']}")
    print(f"    waiting for body  {s['pending_fetch']}")
    print(f"  by status           {s['by_status']}")
    print(f"  polls so far        {s['runs']}  (last finished {s['last_run']} UTC)\n")
    print(f"  {'source':34s} {'stored':>6s} {'relevant':>8s} {'w/ body':>7s}")
    for r in s["by_source"]:
        print(f"  {r['source']:34s} {r['n']:6d} {r['relevant'] or 0:8d} {r['with_body'] or 0:7d}")
    return 0


def cmd_sample(cfg, args) -> int:
    """Random admitted and rejected headlines from gated sources - judge the gate by eye."""
    store = _store(cfg)
    con = sqlite3.connect(store.path)
    for relevant, title in ((1, "ADMITTED"), (0, "REJECTED")):
        rows = con.execute(
            "SELECT source, headline, gate_terms FROM articles a WHERE relevant = ? AND "
            "(source_kind = 'google_news' OR source NOT IN ('defence_in','navalnews','19fortyfive')) "
            "ORDER BY RANDOM() LIMIT ?", (relevant, args.n)).fetchall()
        print(f"\n  --- {title} ({len(rows)} random) ---")
        for source, headline, terms in rows:
            why = f"   <- {terms}" if relevant else ""
            print(f"  [{source[:14]:14s}] {headline[:90]}{why}")
    con.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="live.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="poll the sources once")
    p.add_argument("--source", help="poll only this source (name from live.yaml)")
    sub.add_parser("stats", help="summarise the database")
    p = sub.add_parser("sample", help="random admitted / rejected headlines")
    p.add_argument("-n", type=int, default=15)
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        return {"ingest": cmd_ingest, "stats": cmd_stats, "sample": cmd_sample}[args.cmd](cfg, args)
    except ImportError as exc:
        print(f"  [!!] {exc}\n       install the ingestion extras:  pip install -e \".[ingest]\"")
        return 1


if __name__ == "__main__":
    sys.exit(main())
