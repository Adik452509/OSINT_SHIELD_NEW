#!/usr/bin/env python
"""Run the API: classification (M9) + the live dashboard API (M10c).

    python scripts/10_serve.py                   # API + dashboard API, scheduler stopped
    python scripts/10_serve.py --start           # ... and start collecting right away
    python scripts/10_serve.py --classify-only   # M9 only: /classify, no live database

Then open http://127.0.0.1:8000/docs for interactive docs. The dashboard's
Operations tab starts and stops collection; so do

    Invoke-RestMethod -Method Post http://127.0.0.1:8000/live/scheduler/start
    Invoke-RestMethod -Method Post http://127.0.0.1:8000/live/scheduler/stop

Only one pipeline cycle runs at a time across processes: if `11_live.py run` is
running in another window, the server's cycles wait their turn (D23 lease).
Ctrl+C stops the server. Settings: configs/live.yaml (extends serve.yaml).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.runtime import configure_environment  # noqa: E402

configure_environment()

from osint_shield.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--classify-only", action="store_true",
                    help="serve /classify only (configs/serve.yaml), no live database")
    ap.add_argument("--start", action="store_true",
                    help="start the collection schedule when the server starts")
    args = ap.parse_args()

    try:
        import uvicorn

        from osint_shield.api.main import create_app
    except ImportError as exc:
        print(f"  [!!] {exc}\n       install the API extras:  pip install -e \".[serve]\"")
        return 1

    cfg = load_config("serve.yaml" if args.classify_only else "live.yaml")
    if args.start and "live" in cfg:
        cfg["live"].setdefault("scheduler", {})["autostart"] = True
    api = cfg.get("api", {})
    app = create_app(serve_cfg=cfg, max_articles=int(api.get("max_articles_per_request", 256)))
    host, port = api.get("host", "127.0.0.1"), int(api.get("port", 8000))
    print(f"  loading {cfg['model_dir']} ... the first start takes a few seconds")
    if "live" in cfg:
        state = "collecting every %g min" % cfg["live"].get("poll_minutes", 15) if args.start \
            else "collection stopped - start it from the dashboard or with --start"
        print(f"  live database: {cfg['live']['db_path']}  ({state})")
    print(f"  docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
