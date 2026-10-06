"""Journal (venue) signals from OpenAlex (CONCEPTS.md §5: conflicts, retractions;
§6: scholarly APIs).

Two signals, both deliberately small:
  * standing  - 2-year mean citedness, log-scaled to 0..1. Prestige is a weak
                proxy for truth (top journals publish non-replicable work too),
                so it moves a source's overall score by at most +-0.05.
  * vetting   - is the journal on any curated list (MEDLINE, DOAJ, CWTS core,
                national quality lists)? A journal on none of them is a
                possible predatory venue: flagged, and penalized 0.1.
We do not keep a hand-made list of "predatory publishers"; being absent from
every curated index is the signal.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field

VETTING_LISTS = {"medline", "doaj", "cwts-core", "doyens", "scielo"}


class Venue(BaseModel):
    openalex_id: str | None = None
    name: str
    issns: list[str] = Field(default_factory=list)
    mean_citedness: float | None = None
    h_index: int | None = None
    is_core: bool = False
    listed_in: list[str] = Field(default_factory=list)

    @property
    def standing(self) -> float | None:
        """0..1; ~0.5 at 2 citations/paper, 1.0 at >=10."""
        if self.mean_citedness is None:
            return None
        return round(min(1.0, math.log1p(self.mean_citedness) / math.log1p(10)), 3)

    @property
    def vetted(self) -> bool:
        return self.is_core or bool(VETTING_LISTS & set(self.listed_in)) or any(
            x.startswith(("jufo-", "norway-", "ki-jl-")) for x in self.listed_in)

    def summary(self) -> dict:
        return {"name": self.name, "standing": self.standing, "vetted": self.vetted,
                "mean_citedness": self.mean_citedness, "listed_in": self.listed_in}


def parse_source(s: dict) -> Venue:
    stats = s.get("summary_stats") or {}
    return Venue(openalex_id=(s.get("id") or "").rsplit("/", 1)[-1] or None, name=s.get("display_name") or "",
                 issns=list(dict.fromkeys([s.get("issn_l")] + (s.get("issn") or []))) if s.get("issn_l") else
                 (s.get("issn") or []),
                 mean_citedness=stats.get("2yr_mean_citedness"), h_index=stats.get("h_index"),
                 is_core=bool(s.get("is_core")), listed_in=s.get("listed_in") or [])
