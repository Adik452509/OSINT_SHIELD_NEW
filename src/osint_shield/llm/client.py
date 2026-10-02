"""A minimal client for a local Ollama server.

Only what the project needs: list models, and generate a JSON object that
conforms to a schema (Ollama structured outputs). Standard library only -
no extra dependency for two HTTP calls. M10's explain stage reuses it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Ollama is unreachable, or returned something unusable."""


class OllamaClient:
    """Talks to ``OLLAMA_HOST`` (default ``http://127.0.0.1:11434``).

    Args:
        host: server URL; falls back to the ``OLLAMA_HOST`` environment variable.
        model: model tag; falls back to ``OLLAMA_MODEL``, then ``llama3.2:3b``.
        timeout: seconds per request.
    """

    def __init__(self, host: str | None = None, model: str | None = None, *,
                 timeout: float = 120.0):
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if data is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc

    def models(self) -> list[str]:
        """Tags of the locally pulled models."""
        return [m["name"] for m in self._request("/api/tags").get("models", [])]

    def generate_json(self, prompt: str, schema: dict, *, system: str | None = None,
                      temperature: float = 0.0, num_ctx: int = 4096, seed: int = 0,
                      retries: int = 2) -> dict:
        """Generate one JSON object constrained to ``schema``.

        The first attempt is deterministic (temperature 0, fixed seed). If the
        output is not valid JSON, retries change the seed and add a little
        temperature - repeating an identical deterministic call would only
        reproduce the same failure.

        Raises:
            OllamaError: unreachable, or still invalid JSON after the retries.
        """
        text = ""
        for attempt in range(retries + 1):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": schema,
                "options": {"temperature": temperature if attempt == 0 else 0.3,
                            "num_ctx": num_ctx, "seed": seed + attempt},
            }
            if system:
                payload["system"] = system
            text = self._request("/api/generate", payload).get("response", "")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise OllamaError(f"no valid JSON object after {retries + 1} attempts: {text[:200]!r}")
