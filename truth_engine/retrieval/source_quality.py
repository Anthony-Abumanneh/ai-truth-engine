"""Source quality scoring (spec §10).

Sources are not equal. Each is scored on primary-vs-secondary status,
authority, recency, methodology and directness, and the components are kept
separate so the UI can show them instead of a single opaque number.
"""
from __future__ import annotations

from datetime import date

from ..schemas import Source, SourceQuality

TYPE_METHOD = {
    "primary_record": 0.95, "filing": 0.9, "database": 0.95, "internal": 0.8, "government": 0.85,
    "paper": 0.75, "review": 0.9, "encyclopedia": 0.75, "press_release": 0.55, "analyst_report": 0.6,
    "reference": 0.6, "news": 0.45, "web": 0.4, "blog": 0.2, "forum": 0.1, "marketing": 0.05,
    "tool": 0.95, "document": 0.5,
}
DIRECTNESS = {
    "primary_record": 1.0, "filing": 1.0, "database": 1.0, "internal": 0.9, "paper": 0.9, "tool": 1.0,
    "government": 0.9, "press_release": 0.8, "review": 0.75, "encyclopedia": 0.6, "analyst_report": 0.6,
    "reference": 0.55, "news": 0.45, "web": 0.4, "blog": 0.25, "forum": 0.2, "marketing": 0.3, "document": 0.5,
}


def _recency(published: str | None, today: date | None = None) -> float:
    if not published:
        return 0.5
    try:
        y, m, d = (published.split("-") + ["1", "1"])[:3]
        pub = date(int(y), int(m), int(d))
    except Exception:
        return 0.5
    today = today or date.today()
    years_old = max(0.0, (today - pub).days / 365.25)
    # half-life of ~8 years; never below 0.15 because old != wrong
    return max(0.15, 0.5 ** (years_old / 8))


def score_source(src: Source) -> SourceQuality:
    notes: list[str] = []
    meth = TYPE_METHOD.get(src.source_type, 0.5)
    md = src.metadata or {}
    # Study-design adjustments for papers
    if src.source_type in ("paper", "review"):
        n = md.get("sample_size")
        try:
            n = int(n) if n is not None else None
        except ValueError:
            n = None
        if n is not None:
            if n < 100:
                meth -= 0.15
                notes.append(f"small sample (n={n})")
            elif n >= 500:
                meth += 0.1
                notes.append(f"large sample (n={n})")
        if str(md.get("preregistered", "")).lower() == "true":
            meth += 0.05
            notes.append("preregistered")
        elif str(md.get("preregistered", "")).lower() == "false":
            meth -= 0.05
            notes.append("not preregistered")
        if str(md.get("funding", "")).lower() == "manufacturer":
            meth -= 0.1
            notes.append("industry-funded")
    meth = min(1.0, max(0.0, meth))
    direct = DIRECTNESS.get(src.source_type, 0.5)
    rec = _recency(src.published)
    primary_bonus = 0.08 if src.is_primary else 0.0
    if src.derived_from:
        notes.append(f"derived from {', '.join(src.derived_from)}")
        direct *= 0.7
    overall = 0.40 * src.authority + 0.25 * meth + 0.20 * direct + 0.07 * rec + primary_bonus
    overall = round(min(1.0, overall), 3)
    tier = "A" if overall >= 0.78 else "B" if overall >= 0.6 else "C" if overall >= 0.38 else "D"
    if src.source_type in ("forum", "marketing", "blog"):
        notes.append(f"low-reliability source type ({src.source_type})")
    return SourceQuality(primary=src.is_primary, authority=round(src.authority, 3), recency=round(rec, 3),
                         methodology=round(meth, 3), directness=round(direct, 3), overall=overall,
                         tier=tier, notes=notes)
