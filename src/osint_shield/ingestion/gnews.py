"""Resolve Google News article links to the publisher's URL (M10b).

Google News RSS links (``news.google.com/rss/articles/CBMi...``) do not redirect
to the article: they open a Google page, and the publisher URL is only released
by a second call to Google's internal ``batchexecute`` endpoint, using a
signature and timestamp embedded in that page.

    1. GET  news.google.com/articles/<token>  -> data-n-a-sg, data-n-a-ts
    2. POST news.google.com/_/DotsSplashUi/data/batchexecute  -> the real URL

**This is an undocumented interface.** It can change without notice; when it
does, decoding fails, articles fall back to headline-only, and nothing else
breaks. Google also rate-limits it hard: on 2026-11-01 about 40 requests in two
minutes earned an HTTP 429 ("unusual traffic") block. Hence the budget, the
spacing and the cool-down the pipeline enforces (``live.google_news.decode``),
and :class:`GoogleRateLimited`, which must stop decoding immediately.

Measured before the block: 12/12 links resolved to the expected publisher,
~0.7 s each.
"""

from __future__ import annotations

import json
import re
import time
from urllib.parse import quote, urlsplit

_SIGNATURE = re.compile(r'data-n-a-sg="([^"]+)"')
_TIMESTAMP = re.compile(r'data-n-a-ts="([^"]+)"')
_DECODED = re.compile(r'\\"garturlres\\",\\"(.+?)\\"')

ARTICLE_PAGE = "https://news.google.com/articles/{token}"
BATCH_EXECUTE = "https://news.google.com/_/DotsSplashUi/data/batchexecute"

#: The page is ~1 MB; reading stops as soon as both attributes have appeared.
MAX_PAGE_BYTES = 3_000_000


class DecodeError(RuntimeError):
    """This link could not be resolved (layout changed, link expired, ...)."""


class GoogleRateLimited(RuntimeError):
    """Google answered 429 / its "unusual traffic" page. Stop decoding for a while."""


def article_token(url: str) -> str:
    """The ``CBMi...`` token at the end of a Google News article link."""
    token = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
    if not token:
        raise DecodeError(f"no article token in {url!r}")
    return token


def parse_signature(html: str) -> tuple[str, int] | None:
    """``(signature, timestamp)`` from the article page, or None if absent."""
    sg, ts = _SIGNATURE.search(html), _TIMESTAMP.search(html)
    if not (sg and ts):
        return None
    return sg.group(1), int(ts.group(1))


def batch_payload(token: str, timestamp: int, signature: str) -> str:
    """Form body for the ``batchexecute`` call."""
    inner = ["garturlreq",
             [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1, None, None, None,
               None, None, 0, 1], "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
             token, timestamp, signature]
    return "f.req=" + quote(json.dumps([[["Fbv4je", json.dumps(inner), None, "generic"]]]))


def parse_decoded(text: str) -> str | None:
    """The publisher URL from a ``batchexecute`` response, or None."""
    m = _DECODED.search(text)
    if not m:
        return None
    url = m.group(1).encode("utf-8").decode("unicode_escape")
    return url if url.startswith(("http://", "https://")) else None


def _is_rate_limited(resp) -> bool:
    return resp.status_code == 429 or "/sorry/" in str(resp.url)


class GoogleNewsDecoder:
    """Resolves links one at a time, spaced by ``delay_s``.

    Args:
        client: an ``httpx.Client``-like object (tests pass a fake).
        delay_s: pause between decodes - each costs two Google requests.
    """

    def __init__(self, *, timeout: float = 20.0, user_agent: str = "Mozilla/5.0",
                 delay_s: float = 6.0, client=None):
        if client is None:
            import httpx

            client = httpx.Client(timeout=timeout, follow_redirects=True,
                                  headers={"User-Agent": user_agent})
        self._client = client
        self.delay_s = delay_s
        self._last = 0.0

    def _wait(self) -> None:
        pause = self.delay_s - (time.monotonic() - self._last)
        if pause > 0:
            time.sleep(pause)
        self._last = time.monotonic()

    def _signature(self, token: str) -> tuple[str, int]:
        with self._client.stream("GET", ARTICLE_PAGE.format(token=token)) as resp:
            if _is_rate_limited(resp):
                raise GoogleRateLimited(f"http {resp.status_code} on the article page")
            if resp.status_code != 200:
                raise DecodeError(f"article page http {resp.status_code}")
            buf, size = "", 0
            for chunk in resp.iter_text():
                buf += chunk
                size += len(chunk)
                found = parse_signature(buf)
                if found:
                    return found
                if size > MAX_PAGE_BYTES:
                    break
        raise DecodeError("no signature on the article page (layout changed?)")

    def decode(self, url: str) -> str:
        """The publisher URL behind a Google News link.

        Raises:
            GoogleRateLimited: stop all decoding; Google is refusing us.
            DecodeError: this one link failed.
        """
        self._wait()
        token = article_token(url)
        signature, timestamp = self._signature(token)
        resp = self._client.post(
            BATCH_EXECUTE, content=batch_payload(token, timestamp, signature),
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"})
        if _is_rate_limited(resp):
            raise GoogleRateLimited(f"http {resp.status_code} on batchexecute")
        if resp.status_code != 200:
            raise DecodeError(f"batchexecute http {resp.status_code}")
        real = parse_decoded(resp.text)
        if real is None:
            raise DecodeError("no URL in the batchexecute response")
        return real

    def close(self) -> None:
        self._client.close()
