"""The classification API, plus the live dashboard API when a live config is given.

    POST /classify       {"articles": [{"headline": ..., "body": ...}, ...]}
    POST /classify/one   {"headline": ..., "body": ...}
    GET  /health
    /live/...            dashboard endpoints (M10c) - see api/live.py

Both classify endpoints run through ``ArticleClassifier.classify``, so a single
article and the same article inside a batch cannot diverge by construction.
Interactive docs: http://127.0.0.1:8000/docs once the server is running.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .schemas import ArticleIn, ArticleOut, ClassifyRequest, ClassifyResponse, Health

#: The React dev server (Vite) - the dashboard calls the API from there (M10d).
DEFAULT_CORS = ("http://localhost:5173", "http://127.0.0.1:5173")


class SharedClassifier:
    """One model, two users: API requests and the background pipeline cycle.

    A lock makes them take turns, so the GPU never holds two batches' activations
    at once, and the model is loaded only once on a 6 GB card.
    """

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def classify(self, articles):
        with self._lock:
            return self._inner.classify(articles)

    def __getattr__(self, name):          # model_version, routing, predictor, ...
        return getattr(self._inner, name)


def create_app(classifier=None, *, serve_cfg: dict | None = None,
               max_articles: int = 256, store=None, runner=None) -> FastAPI:
    """Build the app.

    Args:
        classifier: an ``ArticleClassifier``. Tests inject a fake; when ``None``
            the deployed model is loaded from ``serve_cfg`` at startup - once,
            not per request.
        serve_cfg: the loaded config - ``serve.yaml``, or ``live.yaml`` (which
            extends it) to add the ``/live`` dashboard routes.
        max_articles: refuse larger batches rather than exhausting GPU memory.
        store, runner: the live database and cycle runner. Tests inject them;
            otherwise both are built from ``serve_cfg["live"]`` when present.
    """
    cfg = serve_cfg or {}
    live_cfg = cfg.get("live")
    state: dict = {"classifier": SharedClassifier(classifier) if classifier else None,
                   "runner": runner}

    if live_cfg is not None and store is None:
        from ..paths import resolve
        from ..storage import LiveStore

        store = LiveStore(resolve(live_cfg["db_path"]))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["classifier"] is None:
            from ..inference import ArticleClassifier

            state["classifier"] = SharedClassifier(ArticleClassifier.from_config(cfg))
        if live_cfg is not None and state["runner"] is None:
            from ..live import CycleRunner, LivePipeline

            state["runner"] = CycleRunner(
                lambda: LivePipeline.from_config(cfg, classifier=state["classifier"]),
                every_minutes=float(live_cfg.get("poll_minutes", 15)))
            if (live_cfg.get("scheduler") or {}).get("autostart", False):
                state["runner"].start()
        yield
        if state["runner"] is not None:
            state["runner"].close()

    app = FastAPI(title="OSINT Shield", version="0.2.0", lifespan=lifespan,
                  description="Narrative and severity classification for defence & "
                              "security news. Propaganda is not reliably assessed (D20).")
    app.add_middleware(CORSMiddleware,
                       allow_origins=list((cfg.get("api") or {}).get("cors_origins",
                                                                      DEFAULT_CORS)),
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    def clf():
        return state["classifier"]

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """The bare address has nothing of its own - send people to the docs."""
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=Health)
    def health() -> Health:
        c = clf()
        device = getattr(getattr(c, "predictor", None), "device", "unknown")
        return Health(status="ok", model_version=c.model_version, device=str(device),
                      escalate_below=float(c.routing.escalate_below))

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest) -> ClassifyResponse:
        if len(req.articles) > max_articles:
            raise HTTPException(413, f"at most {max_articles} articles per request")
        t0 = time.perf_counter()
        results = clf().classify([a.model_dump() for a in req.articles])
        return ClassifyResponse(model_version=clf().model_version,
                                elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
                                results=results)

    @app.post("/classify/one", response_model=ArticleOut)
    def classify_one(article: ArticleIn) -> ArticleOut:
        return clf().classify([article.model_dump()])[0]

    if store is not None:
        from .live import live_router

        app.include_router(live_router(
            store, live_cfg or {}, get_runner=lambda: state["runner"],
            model_version=lambda: getattr(clf(), "model_version", None)))

    return app
