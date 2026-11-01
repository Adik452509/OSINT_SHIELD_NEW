"""Fetch an article's page and extract its text (M10b).

Most feeds carry a headline and a teaser at best (D22), and headline-only
classification is barely better than guessing (D17) - so the body is fetched
from the publisher's page. Text extraction uses trafilatura, which separates
article text from navigation, ads and comments.

Two checks keep a wrong page from becoming an article body:

* **length** - under ``min_chars`` is a cookie wall, a paywall stub or an
  error page rather than an article;
* **headline match** - at least 40% of the headline's content words must appear
  in the page title or the opening text. A consent page or a homepage redirect
  is long enough to pass the first check but fails this one.

Measured 2026-11-01 (3 random relevant articles per source): defence.in,
Indian Express, The Hindu, HT, Livemint, BusinessLine 3/3; TOI 2/3; The Diplomat
and SCMP answer 403 to any automated client (``unfetchable_domains``).
"""

from __future__ import annotations

import re
import time

from ..data.dedupe import strip_source_suffix

#: Common words that say nothing about which article a page is.
_STOP = frozenset("""
about after against among amid also been before being between both could during from have
here into more most over said says such than that their them then there these they this
those through under until upon very were what when where which while with would your will
india indian news today latest live update updates""".split())

_WORD = re.compile(r"[a-z0-9]+")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


class FetchError(RuntimeError):
    """The page could not be turned into an article body. The message says why."""


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOP}


def matches_headline(headline: str, page_title: str, text: str, *,
                     min_share: float = 0.4) -> bool:
    """Does the page look like the article this headline announced?

    Headlines with fewer than 3 content words are accepted: too little to judge.
    """
    wanted = content_words(strip_source_suffix(headline))
    if len(wanted) < 3:
        return True
    present = content_words(f"{page_title}\n{text[:4000]}")
    return len(wanted & present) / len(wanted) >= min_share


def extract_article(html: str, url: str | None = None) -> str:
    """Main article text of an HTML page; empty when none is found."""
    import trafilatura

    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False,
                               favor_precision=True, deduplicate=True)
    return (text or "").strip()


class PageFetcher:
    """Downloads article pages, politely, and returns their text.

    Args:
        timeout: seconds per request.
        user_agent: the same honest agent as the feeds; it was accepted wherever
            a browser agent was (measured), so there is no reason to pretend.
        min_chars: shortest text accepted as a body.
        delay_s: pause between requests, so a backlog is not a burst.
    """

    def __init__(self, *, timeout: float = 20.0, user_agent: str = "Mozilla/5.0",
                 min_chars: int = 400, delay_s: float = 0.5, client=None):
        if client is None:
            import httpx

            client = httpx.Client(timeout=timeout, follow_redirects=True,
                                  headers={"User-Agent": user_agent})
        self._client = client
        self.min_chars = min_chars
        self.delay_s = delay_s
        self._last = 0.0

    def _wait(self) -> None:
        pause = self.delay_s - (time.monotonic() - self._last)
        if pause > 0:
            time.sleep(pause)
        self._last = time.monotonic()

    def fetch(self, url: str, headline: str) -> str:
        """Return the article text at ``url``.

        Raises:
            FetchError: blocked, not HTML, too short, or not the expected article.
        """
        self._wait()
        try:
            resp = self._client.get(url)
        except Exception as exc:  # noqa: BLE001 - timeouts, DNS, TLS alike
            raise FetchError(f"{type(exc).__name__}: {exc}"[:200]) from exc
        if resp.status_code != 200:
            raise FetchError(f"http {resp.status_code}")
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and ctype:
            raise FetchError(f"not a web page ({ctype.split(';')[0]})")

        html = resp.text
        text = extract_article(html, url)
        if len(text) < self.min_chars:
            raise FetchError(f"too short ({len(text)} chars)")
        title = _TITLE.search(html)
        if not matches_headline(headline, title.group(1) if title else "", text):
            raise FetchError("page does not match the headline")
        return text

    def close(self) -> None:
        self._client.close()
