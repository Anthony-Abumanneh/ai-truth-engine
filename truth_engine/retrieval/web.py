"""Live web retrieval. Activated by TAVILY_API_KEY, BRAVE_API_KEY or SERPER_API_KEY.

Results are converted to Source/Passage objects, scored for quality like any
other source, and cached in-process so the supporting and contradiction
searches for the same claim do not pay twice."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

import httpx

from ..config import settings
from ..schemas import Passage, Source
from ..textutils import sentences
from .source_quality import score_source

HIGH_AUTHORITY = {
    ".gov": 0.9, ".edu": 0.8, "who.int": 0.9, "nih.gov": 0.92, "nature.com": 0.88, "science.org": 0.88,
    "thelancet.com": 0.88, "nejm.org": 0.9, "bmj.com": 0.86, "cochranelibrary.com": 0.9, "arxiv.org": 0.65,
    "wikipedia.org": 0.72, "britannica.com": 0.8, "reuters.com": 0.78, "apnews.com": 0.78, "bbc.co.uk": 0.74,
    "nytimes.com": 0.72, "ft.com": 0.74, "wsj.com": 0.72, "sec.gov": 0.95, "europa.eu": 0.86,
}
LOW_AUTHORITY = ("reddit.com", "quora.com", "medium.com", "blogspot.", "wordpress.", "pinterest.", "tiktok.", "facebook.")


def _authority(url: str) -> tuple[float, str]:
    host = urlparse(url).netloc.lower()
    for k, v in HIGH_AUTHORITY.items():
        if host.endswith(k) or k in host:
            stype = "paper" if k in ("nature.com", "science.org", "thelancet.com", "nejm.org", "bmj.com", "arxiv.org") else \
                "government" if k in (".gov", "sec.gov", "europa.eu", "who.int", "nih.gov") else \
                "encyclopedia" if k in ("wikipedia.org", "britannica.com") else "news"
            return v, stype
    if any(x in host for x in LOW_AUTHORITY):
        return 0.2, "forum"
    return 0.45, "web"


class WebSearch:
    def __init__(self):
        self.client = httpx.Client(timeout=25)
        self.cache: dict[str, list[tuple[Source, list[Passage]]]] = {}

    @property
    def available(self) -> bool:
        return settings.has_web and not settings.force_offline

    def search(self, query: str, k: int = 5) -> list[tuple[Source, list[Passage]]]:
        if not self.available:
            return []
        if query in self.cache:
            return self.cache[query]
        try:
            if settings.tavily_key:
                raw = self._tavily(query, k)
            elif settings.brave_key:
                raw = self._brave(query, k)
            else:
                raw = self._serper(query, k)
        except Exception:
            raw = []
        out = []
        for title, url, text, published in raw:
            sid = "web-" + hashlib.sha1(url.encode()).hexdigest()[:10]
            auth, stype = _authority(url)
            src = Source(id=sid, title=title or url, url=url, source_type=stype,
                         publisher=urlparse(url).netloc, published=published, authority=auth,
                         is_primary=stype in ("government", "paper"))
            src.quality = score_source(src)
            passages = [Passage(id=f"{sid}#s{i}", source_id=sid, text=s, retrieval_method="web")
                        for i, s in enumerate(sentences(text)[:12]) if len(s) > 25]
            out.append((src, passages))
        self.cache[query] = out
        return out

    def _tavily(self, q, k):
        r = self.client.post("https://api.tavily.com/search", json={
            "api_key": settings.tavily_key, "query": q, "max_results": k, "search_depth": "advanced",
            "include_answer": False})
        r.raise_for_status()
        return [(x.get("title"), x["url"], x.get("content", ""), x.get("published_date")) for x in r.json().get("results", [])]

    def _brave(self, q, k):
        r = self.client.get("https://api.search.brave.com/res/v1/web/search", params={"q": q, "count": k},
                            headers={"X-Subscription-Token": settings.brave_key, "Accept": "application/json"})
        r.raise_for_status()
        res = r.json().get("web", {}).get("results", [])
        return [(x.get("title"), x["url"], re.sub("<[^>]+>", "", x.get("description", "") + " " +
                                                   " ".join(x.get("extra_snippets", []))), x.get("page_age")) for x in res]

    def _serper(self, q, k):
        r = self.client.post("https://google.serper.dev/search", json={"q": q, "num": k},
                             headers={"X-API-KEY": settings.serper_key})
        r.raise_for_status()
        return [(x.get("title"), x["link"], x.get("snippet", ""), x.get("date")) for x in r.json().get("organic", [])]


_web: WebSearch | None = None


def get_web() -> WebSearch:
    global _web
    if _web is None:
        _web = WebSearch()
    return _web
