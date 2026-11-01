#!/usr/bin/env python
"""M10 - the live pipeline.

    python scripts/11_live.py run --once          # one full cycle, then stop
    python scripts/11_live.py run                 # a cycle every 15 min; Ctrl+C stops
    python scripts/11_live.py stats               # what the database holds
    python scripts/11_live.py show                # latest labelled articles
    python scripts/11_live.py show --high         # ... only High severity
    python scripts/11_live.py ingest              # collect only (M10a)
    python scripts/11_live.py sample              # eyeball the relevance gate

A cycle = collect -> fetch article text -> classify -> Llama explains escalated
articles. Data: data/live/osint_live.db. Settings: configs/live.yaml.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.runtime import configure_environment  # noqa: E402

configure_environment()

from osint_shield.config import load_config  # noqa: E402
from osint_shield.paths import resolve  # noqa: E402


def _store(cfg):
    from osint_shield.storage import LiveStore

    return LiveStore(resolve(cfg["live"]["db_path"]))


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ------------------------------------------------------------------------- run
def _print_cycle(rep: dict) -> None:
    if rep.get("skipped"):
        print(f"    skipped: {rep['skipped']}")
        return
    ing, f = rep.get("ingest") or {}, rep.get("fetch") or {}
    c, x = rep.get("classify") or {}, rep.get("explain") or {}
    t = ing.get("totals") or {}
    print(f"    collected  {t.get('new', 0)} new, {t.get('relevant', 0)} relevant "
          f"({t.get('fetched', 0)} feed items read)")
    if ing.get("failed_sources"):
        print(f"               sources that failed: {', '.join(ing['failed_sources'])}")
    print(f"    text       {f.get('pages_ok', 0)} pages fetched, {f.get('pages_failed', 0)} failed, "
          f"{f.get('pages_skipped', 0)} skipped (site blocks us); "
          f"Google News {f.get('gnews_decoded', 0)} decoded, {f.get('gnews_failed', 0)} failed, "
          f"{f.get('duplicates', 0)} were duplicates; {f.get('expired', 0)} gave up waiting")
    if f.get("gnews_paused_until"):
        print(f"               Google News decoding paused until {f['gnews_paused_until']} UTC")
    print(f"    classified {c.get('classified', 0)} ({c.get('with_body', 0)} with text): "
          f"{c.get('high', 0)} High, {c.get('escalated', 0)} escalated, "
          f"{c.get('needs_review', 0)} for review")
    note = f"  [{x['skipped']}]" if x.get("skipped") else ""
    each = f", {x['seconds_each']} s each" if x.get("seconds_each") else ""
    print(f"    explained  {x.get('ok', 0)} ok, {x.get('unverified', 0)} without verified quotes, "
          f"{x.get('failed', 0)} failed{each}; {x.get('quotes_dropped', 0)} invented quotes "
          f"dropped{note}")
    for err in rep.get("errors", []):
        print(f"    [!!] {err}")
    print(f"    took {rep.get('seconds', 0)} s")


def cmd_run(cfg, args) -> int:
    from osint_shield.live import LivePipeline

    every = float(args.every or cfg["live"].get("poll_minutes", 15))
    pipe = LivePipeline.from_config(cfg, log=lambda m: print(f"  [{_stamp()}] {m}", flush=True))
    print(f"  database: {pipe.store.path.relative_to(ROOT)}")
    if not args.once:
        print(f"  a cycle every {every:g} min - Ctrl+C to stop\n")
    try:
        while True:
            start = time.monotonic()
            print(f"  === cycle started {_stamp()}")
            _print_cycle(pipe.run_cycle())
            if args.once:
                break
            wait = every * 60 - (time.monotonic() - start)
            if wait > 0:
                print(f"  next cycle in {wait / 60:.1f} min\n", flush=True)
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        pipe.close()
    return 0


# ---------------------------------------------------------------------- ingest
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


# ----------------------------------------------------------------------- stats
def cmd_stats(cfg, args) -> int:
    s = _store(cfg).stats()
    print(f"  articles stored      {s['articles']}   (relevant {s['relevant']})")
    print(f"  article status       {s['by_status']}")
    print(f"  text (relevant)      {s['by_fetch_status']}   from {s['body_source']}")
    print(f"  classified           {s['predictions']}: {s['by_narrative']}")
    print(f"    High severity      {s['high_severity']}")
    print(f"    escalated          {s['escalated']}")
    print(f"    for review         {s['needs_review']}")
    print(f"  explanations         {s['explanations']}")
    print(f"  cycles so far        {s['runs']}  (last finished {s['last_run']} UTC)\n")
    print(f"  {'source':34s} {'stored':>6s} {'relevant':>8s} {'w/ text':>7s}")
    for r in s["by_source"]:
        print(f"  {r['source']:34s} {r['n']:6d} {r['relevant'] or 0:8d} {r['with_body'] or 0:7d}")
    return 0


# ------------------------------------------------------------------------ show
def cmd_show(cfg, args) -> int:
    """The latest labelled articles, with Llama's summary and quotes when present."""
    con = sqlite3.connect(_store(cfg).path)
    con.row_factory = sqlite3.Row
    where = "WHERE p.severity = 'High'" if args.high else ""
    rows = con.execute(
        f"SELECT a.headline, a.source_domain, a.body IS NOT NULL AS has_text, p.narrative, "
        f"p.narrative_conf, p.narrative_source, p.severity, p.p_high, p.escalated, "
        f"p.needs_review, e.status AS ex_status, e.summary, e.evidence "
        f"FROM predictions p JOIN articles a ON a.id = p.article_id "
        f"LEFT JOIN explanations e ON e.article_id = a.id {where} "
        f"ORDER BY a.collected_at DESC LIMIT ?", (args.n,)).fetchall()
    for r in rows:
        conf = f"{r['narrative_conf']:.2f}" if r["narrative_conf"] is not None else r["narrative_source"]
        flags = " ".join(f for f, on in (("ESCALATED", r["escalated"]),
                                         ("REVIEW", r["needs_review"]),
                                         ("headline-only", not r["has_text"])) if on)
        print(f"\n  {r['headline'][:110]}")
        print(f"    {r['source_domain']} | {r['narrative']} ({conf}) | {r['severity']} "
              f"(P(High) {r['p_high']:.2f}) | {flags}")
        if r["ex_status"]:
            print(f"    LLM [{r['ex_status']}]: {(r['summary'] or '')[:300]}")
            for q in json.loads(r["evidence"] or "[]"):
                print(f"      “{q[:200]}”")
    if not rows:
        print("  nothing classified yet - run:  python scripts/11_live.py run --once")
    con.close()
    return 0


# ---------------------------------------------------------------------- sample
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
    p = sub.add_parser("run", help="the pipeline: every poll_minutes, or --once")
    p.add_argument("--once", action="store_true", help="one cycle, then stop")
    p.add_argument("--every", type=float, help="minutes between cycles (default: live.yaml)")
    p = sub.add_parser("ingest", help="collect only, once")
    p.add_argument("--source", help="poll only this source (name from live.yaml)")
    sub.add_parser("stats", help="summarise the database")
    p = sub.add_parser("show", help="latest labelled articles")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--high", action="store_true", help="only High severity")
    p = sub.add_parser("sample", help="random admitted / rejected headlines")
    p.add_argument("-n", type=int, default=15)
    args = ap.parse_args()

    cfg = load_config(args.config)
    commands = {"run": cmd_run, "ingest": cmd_ingest, "stats": cmd_stats, "show": cmd_show,
                "sample": cmd_sample}
    try:
        return commands[args.cmd](cfg, args)
    except ImportError as exc:
        print(f"  [!!] {exc}\n       install the extras:  pip install -e \".[ingest,serve,llm]\"")
        return 1


if __name__ == "__main__":
    sys.exit(main())
