"""Retraction Watch data via Crossref (CONCEPTS.md §5: retractions).

Since 2023 the Retraction Watch database is open and served through the
Crossref REST API as `updated-by` notices on the original article's DOI. This
catches retractions PubMed and OpenAlex haven't flagged yet, plus
expressions of concern (an editor's warning short of retraction).
"""
from __future__ import annotations

import json

from ..config import settings
from .cache import CachedFetcher, JsonCache

BASE = "https://api.crossref.org/works"
SERIOUS = {"retraction", "expression_of_concern", "withdrawal", "removal"}


class CrossrefClient:
    def __init__(self, *, cache: JsonCache | None = None, client=None, offline: bool | None = None,
                 email: str | None = None):
        self.email = settings.openalex_email if email is None else email
        self.http = CachedFetcher("crossref", per_second=5, cache=cache, client=client, offline=offline)

    @property
    def errors(self) -> list[str]:
        return self.http.errors

    def notices(self, dois: list[str]) -> dict[str, list[dict]]:
        """{doi (lowercase): [{"type": "retraction", "doi": notice doi, "date": "YYYY-MM-DD", "source": ...}]}
        for every DOI with a serious editorial notice. Batched, 20 DOIs per request."""
        out: dict[str, list[dict]] = {}
        dois = sorted({d.lower() for d in dois if d})
        for i in range(0, len(dois), 20):
            chunk = dois[i:i + 20]
            params = {"filter": ",".join(f"doi:{d}" for d in chunk), "select": "DOI,updated-by",
                      "rows": len(chunk)}
            if self.email:
                params["mailto"] = self.email
            body = self.http.get(BASE, params)
            if not body:
                continue
            try:
                items = json.loads(body)["message"]["items"]
            except (ValueError, KeyError):
                self.http.errors.append("crossref: unparseable response")
                continue
            for it in items:
                found = []
                for u in it.get("updated-by") or []:
                    kind = (u.get("type") or "").lower().replace(" ", "_")
                    if kind in SERIOUS:
                        parts = ((u.get("updated") or {}).get("date-parts") or [[None]])[0]
                        date = "-".join(f"{p:02d}" if j else str(p) for j, p in enumerate(parts) if p) or None
                        found.append({"type": kind, "doi": u.get("DOI"), "date": date, "source": u.get("source")})
                if found:
                    out[it["DOI"].lower()] = found
        return out
