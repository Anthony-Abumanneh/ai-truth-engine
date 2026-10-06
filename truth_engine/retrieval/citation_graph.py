"""Citation and derivation graph (spec Improvement 8).

Answers one question that a flat retriever cannot: *how many independent
lines of evidence actually support this claim?* Five articles that all repeat
one press release are one line of evidence, not five.

Edges:
  derived_from  - the source restates another (news article -> study). Derived
                  sources collapse onto their ancestor.
  cites         - the source references another but did its own work (a new
                  trial citing an older one). Citation alone does NOT collapse
                  independence.
Sources from the same publisher are additionally treated as one line of
evidence (a company's press releases are not independent of each other).
"""
from __future__ import annotations

from collections import defaultdict

from ..schemas import Source


class CitationGraph:
    def __init__(self, sources: dict[str, Source]):
        self.sources = sources

    def roots(self, source_id: str, _seen: set[str] | None = None) -> set[str]:
        """Original ancestors reached by following derived_from edges."""
        seen = _seen or set()
        if source_id in seen:
            return set()
        seen.add(source_id)
        src = self.sources.get(source_id)
        if not src or not src.derived_from:
            return {source_id}
        out: set[str] = set()
        for parent in src.derived_from:
            if parent in self.sources:
                out |= self.roots(parent, seen)
            else:
                out.add(parent)
        return out or {source_id}

    def lineages(self, source_ids: list[str]) -> list[set[str]]:
        """Group sources into independent lines of evidence."""
        # union-find over sources sharing a root or a publisher
        parent: dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        ids = list(dict.fromkeys(source_ids))
        root_owner: dict[str, str] = {}
        pub_owner: dict[str, str] = {}
        for sid in ids:
            find(sid)
            for r in self.roots(sid):
                if r in root_owner:
                    union(root_owner[r], sid)
                else:
                    root_owner[r] = sid
            src = self.sources.get(sid)
            pub = (src.publisher or "").strip().lower() if src else ""
            if pub:
                if pub in pub_owner:
                    union(pub_owner[pub], sid)
                else:
                    pub_owner[pub] = sid
        groups: dict[str, set[str]] = defaultdict(set)
        for sid in ids:
            groups[find(sid)].add(sid)
        return list(groups.values())

    def independence(self, source_ids: list[str]) -> tuple[int, float]:
        """(number of independent lineages, independence ratio in [0,1])."""
        ids = list(dict.fromkeys(source_ids))
        if not ids:
            return 0, 0.0
        n = len(self.lineages(ids))
        return n, n / len(ids)

    def subgraph(self, source_ids: list[str]) -> dict:
        """Nodes/edges for the UI, including ancestors not directly retrieved."""
        ids = set(source_ids)
        frontier = list(ids)
        while frontier:
            s = self.sources.get(frontier.pop())
            if not s:
                continue
            for p in s.derived_from + s.cites:
                if p not in ids and p in self.sources:
                    ids.add(p)
                    frontier.append(p)
        nodes = []
        for sid in ids:
            s = self.sources.get(sid)
            if s:
                nodes.append({"id": sid, "title": s.title, "type": s.source_type, "publisher": s.publisher,
                              "tier": s.quality.tier if s.quality else None, "retrieved": sid in source_ids})
        edges = []
        for sid in ids:
            s = self.sources.get(sid)
            if not s:
                continue
            for p in s.derived_from:
                if p in ids:
                    edges.append({"from": sid, "to": p, "kind": "derived_from"})
            for p in s.cites:
                if p in ids and p not in s.derived_from:
                    edges.append({"from": sid, "to": p, "kind": "cites"})
        return {"nodes": nodes, "edges": edges,
                "lineages": [sorted(g) for g in self.lineages([i for i in source_ids if i in self.sources])]}
