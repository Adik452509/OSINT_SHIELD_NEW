#!/usr/bin/env python
"""M9 - run the classification API.

    python scripts/10_serve.py

Then open http://127.0.0.1:8000/docs for interactive docs, or from a second
PowerShell window:

    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify/one `
      -ContentType "application/json" `
      -Body '{"headline": "DRDO conducts salvo launch of two Pralay missiles"}'

Ctrl+C stops it. Settings: configs/serve.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.runtime import configure_environment  # noqa: E402

configure_environment()

from osint_shield.config import load_config  # noqa: E402


def main() -> int:
    try:
        import uvicorn

        from osint_shield.api.main import create_app
    except ImportError as exc:
        print(f"  [!!] {exc}\n       install the API extras:  pip install -e \".[serve]\"")
        return 1

    cfg = load_config("serve.yaml")
    api = cfg.get("api", {})
    app = create_app(serve_cfg=cfg, max_articles=int(api.get("max_articles_per_request", 256)))
    print(f"  loading {cfg['model_dir']} ... the first start takes a few seconds")
    uvicorn.run(app, host=api.get("host", "127.0.0.1"), port=int(api.get("port", 8000)),
                log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
