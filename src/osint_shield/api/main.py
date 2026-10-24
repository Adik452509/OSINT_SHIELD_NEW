"""The classification API.

    POST /classify       {"articles": [{"headline": ..., "body": ...}, ...]}
    POST /classify/one   {"headline": ..., "body": ...}
    GET  /health

Both classify endpoints run through ``ArticleClassifier.classify``, so a single
article and the same article inside a batch cannot diverge by construction.
Interactive docs: http://127.0.0.1:8000/docs once the server is running.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import ArticleIn, ArticleOut, ClassifyRequest, ClassifyResponse, Health


def create_app(classifier=None, *, serve_cfg: dict | None = None,
               max_articles: int = 256) -> FastAPI:
    """Build the app.

    Args:
        classifier: an ``ArticleClassifier``. Tests inject a fake; when ``None``
            the deployed model is loaded from ``serve_cfg`` at startup - once,
            not per request.
        serve_cfg: the loaded ``configs/serve.yaml``.
        max_articles: refuse larger batches rather than exhausting GPU memory.
    """
    state: dict = {"classifier": classifier}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["classifier"] is None:
            from ..inference import ArticleClassifier

            state["classifier"] = ArticleClassifier.from_config(serve_cfg or {})
        yield

    app = FastAPI(title="OSINT Shield", version="0.1.0", lifespan=lifespan,
                  description="Narrative and severity classification for defence & "
                              "security news. Propaganda is not reliably assessed (D20).")

    def clf():
        return state["classifier"]

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

    return app
