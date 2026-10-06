"""Knowledge graph / GraphRAG layer (spec Improvement 7).

Entities and relations are explicit, every edge keeps the source that
justifies it, and multi-hop questions are answered by path search rather than
hoping one flat chunk happens to contain the whole chain. The graph
complements text retrieval: paths it returns become evidence items like any
passage, so they still go through verification.

Edges come from:
  * data/graph.json (curated seed)
  * pattern-based relation extraction over the corpus at load time
  * optional LLM triple extraction for newly uploaded documents
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..config import DATA_DIR
from ..textutils import content_tokens, normalize, sentences, stem

# question words that signal each relation (stemmed)
RELATION_TRIGGERS = {
    "born_in": {"born", "birthplace", "birth"},
    "died_in": {"die", "died", "death"},
    "located_in": {"country", "located", "where", "state", "city", "region"},
    "developed": {"develop", "developed", "theory", "formulat", "propos"},
    "influenced": {"influenc", "impact", "shap"},
    "contributed_to": {"contribut", "technology", "enabl", "led", "applic", "use"},
    "discovered_by": {"discover", "discoverer"},
    "created_by": {"creat", "creator", "invent", "wrote", "author"},
    "built_by": {"built", "design", "construct", "engineer"},
    "capital_of": {"capital"},
    "awarded": {"award", "nobel", "prize", "won", "win"},
    "awarded_for": {"nobel", "prize", "award", "for", "why"},
    "launched": {"launch", "releas", "introduc"},
    "discontinued": {"discontinu", "kill", "end", "stop"},
    "succeeded_by": {"replac", "successor", "succeed"},
    "supplies": {"suppl", "supplier", "vendor"},
    "ships_to": {"ship", "deliver"},
}
TARGET_TYPE_WORDS = {
    "country": {"country", "nation"}, "city": {"city", "town"}, "person": {"who", "person", "scientist", "whom"},
    "state": {"state", "region"}, "technology": {"technology"}, "concept": {"theory", "concept", "field"},
    "organization": {"company", "organization", "firm", "supplier"}, "product": {"product"},
}

# (regex, relation, subject-group, object-group)
_PATTERNS = [
    (r"(?P<s>[A-Z][\w\- ]+?) was born (?:on [^,]+? )?in (?P<o>[A-Z][\w\-]+)", "born_in"),
    (r"(?P<s>[A-Z][\w\- ]+?) developed the (?:special |general )?(?:theory of )?(?P<o>[a-z][\w ]+?)(?: in \d{4}| and|\.|,)", "developed"),
    (r"(?P<s>[A-Z][\w\- ]+?) was discovered in \d{4} by (?:the [\w ]+? scientist )?(?P<o>[A-Z][\w]+ [A-Z][\w]+)", "discovered_by"),
    (r"(?P<s>[A-Z][\w\- ]+?) is a [\w\- ,]+? created by (?P<o>[A-Z][\w]+(?: van| von| de)? [A-Z][\w]+)", "created_by"),
    (r"[Tt]he capital of (?P<o>[A-Z][\w]+) is (?P<s>[A-Z][\w]+)", "capital_of"),
    (r"(?P<s>[A-Z][\w\-]+) is a city in [\w\- ]*?(?P<o>[A-Z][\w\-]+)(?=[,.])", "located_in"),
    (r"(?P<s>[A-Z][\w ]+?) (?:today )?launched the (?P<o>[A-Z][\w]+ [A-Z0-9][\w]*)", "launched"),
]


@dataclass
class Edge:
    s: str
    rel: str
    o: str
    source_id: str
    extracted: bool = False


@dataclass
class GraphPath:
    edges: list[Edge]
    score: float
    answer: str
    answer_type: str
    text: str = ""
    sources: list[str] = field(default_factory=list)


class KnowledgeGraph:
    def __init__(self):
        self.types: dict[str, str] = {}
        self.alias: dict[str, str] = {}
        self.out: dict[str, list[Edge]] = defaultdict(list)
        self.inc: dict[str, list[Edge]] = defaultdict(list)
        self.edges: list[Edge] = []
        self.source_quality: dict[str, float] = {}

    # ------------------------------------------------------------ building
    def canonical(self, name: str) -> str:
        key = self._key(name)
        return self.alias.get(key, name.strip())

    @staticmethod
    def _key(name: str) -> str:
        n = normalize(name).lower().strip()
        n = re.sub(r"^(the|a|an)\s+", "", n)
        return re.sub(r"\s+", " ", n)

    def add_entity(self, name: str, etype: str = "entity", aliases: list[str] | None = None):
        self.types.setdefault(name, etype)
        self.alias[self._key(name)] = name
        for a in aliases or []:
            self.alias[self._key(a)] = name

    def add_edge(self, s: str, rel: str, o: str, source_id: str, extracted: bool = False):
        s, o = self.canonical(s), self.canonical(o)
        if any(e.s == s and e.rel == rel and e.o == o for e in self.out[s]):
            return
        e = Edge(s, rel, o, source_id, extracted)
        self.types.setdefault(s, "entity")
        self.types.setdefault(o, "entity")
        self.alias.setdefault(self._key(s), s)
        self.alias.setdefault(self._key(o), o)
        self.out[s].append(e)
        self.inc[o].append(e)
        self.edges.append(e)

    def load_seed(self, path=None):
        path = path or DATA_DIR / "graph.json"
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for name, info in data.get("entities", {}).items():
            self.add_entity(name, info.get("type", "entity"), info.get("aliases", []))
        for s, rel, o, src in data.get("triples", []):
            self.add_edge(s, rel, o, src)

    def extract_from_text(self, text: str, source_id: str) -> int:
        n = 0
        for sent in sentences(normalize(text)):
            for pat, rel in _PATTERNS:
                for m in re.finditer(pat, sent):
                    s, o = m.group("s").strip(), m.group("o").strip()
                    if len(s) < 2 or len(o) < 2 or s.lower() in ("it", "the", "this"):
                        continue
                    self.add_edge(s, rel, o, source_id, extracted=True)
                    n += 1
        return n

    # ------------------------------------------------------------ querying
    def find_entities(self, text: str) -> list[str]:
        """Graph entities mentioned in text, longest alias first."""
        t = " " + self._key(text) + " "
        t = re.sub(r"[^\w\- ]", " ", t)
        found: list[tuple[int, str]] = []
        for key, name in self.alias.items():
            if len(key) < 2:
                continue
            if re.search(r"(?<![\w-])" + re.escape(key) + r"(?![\w-])", t):
                found.append((len(key), name))
        found.sort(reverse=True)
        out: list[str] = []
        for _, name in found:
            if name not in out:
                out.append(name)
        return out

    def neighbors(self, node: str):
        for e in self.out.get(node, []):
            yield e, e.o
        for e in self.inc.get(node, []):
            yield e, e.s

    def multi_hop(self, question: str, max_hops: int = 3, k: int = 3) -> list[GraphPath]:
        anchors = self.find_entities(question)
        if not anchors:
            return []
        qstems = {stem(w) for w in re.findall(r"[a-zA-Z]+", question.lower())} | set(content_tokens(question))
        wanted_rels = {r for r, trig in RELATION_TRIGGERS.items() if qstems & {stem(t) for t in trig}}
        target_types = {t for t, ws in TARGET_TYPE_WORDS.items() if qstems & {stem(w) for w in ws}}

        paths: list[GraphPath] = []
        for anchor in anchors[:3]:
            q = deque([(anchor, [])])
            seen = {anchor}
            while q:
                node, path = q.popleft()
                if path:
                    paths.append(self._score(path, node, wanted_rels, target_types, anchors))
                if len(path) >= max_hops:
                    continue
                for e, nxt in self.neighbors(node):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    q.append((nxt, path + [e]))
        paths = [p for p in paths if p.score > 0]
        paths.sort(key=lambda p: (-p.score, len(p.edges)))
        uniq, seen_ans = [], set()
        for p in paths:
            if p.answer in seen_ans:
                continue
            seen_ans.add(p.answer)
            uniq.append(p)
        return uniq[:k]

    def _score(self, path: list[Edge], end: str, wanted: set[str], target_types: set[str], anchors: list[str]) -> GraphPath:
        rels = [e.rel for e in path]
        covered = len(wanted & set(rels))
        rel_score = covered / max(len(wanted), 1) if wanted else 0.3
        # penalize relations the question did not ask about
        off_topic = sum(1 for r in rels if wanted and r not in wanted)
        etype = self.types.get(end, "entity")
        type_bonus = 0.35 if target_types and etype in target_types else (-0.25 if target_types else 0)
        end_is_anchor = end in anchors
        # a path is only as trustworthy as its weakest edge's source
        weakest = min((self.source_quality.get(e.source_id, 0.7) for e in path), default=0.7)
        score = (rel_score + type_bonus - 0.15 * off_topic - 0.05 * len(path) - (0.6 if end_is_anchor else 0)) \
            * (0.4 + 0.6 * weakest) - (0.3 if weakest < 0.4 else 0)
        text = self._verbalize(path)
        return GraphPath(path, round(score, 3), end, etype, text, sorted({e.source_id for e in path}))

    @staticmethod
    def _verbalize(path: list[Edge]) -> str:
        return "; ".join(f"{e.s} {e.rel.replace('_', ' ')} {e.o}" for e in path)

    def subgraph(self, entities: list[str], depth: int = 1) -> dict:
        nodes, edges = set(), []
        frontier = set(entities)
        for _ in range(depth + 1):
            nxt = set()
            for n in frontier:
                nodes.add(n)
                for e, m in self.neighbors(n):
                    edges.append(e)
                    nxt.add(m)
            frontier = nxt - nodes
        nodes |= {e.s for e in edges} | {e.o for e in edges}
        uniq = {(e.s, e.rel, e.o): e for e in edges}
        return {"nodes": [{"id": n, "type": self.types.get(n, "entity")} for n in sorted(nodes)],
                "edges": [{"from": e.s, "rel": e.rel, "to": e.o, "source": e.source_id, "extracted": e.extracted}
                          for e in uniq.values()]}

    def stats(self) -> dict:
        return {"entities": len(self.types), "edges": len(self.edges),
                "extracted_edges": sum(1 for e in self.edges if e.extracted)}


_graph: KnowledgeGraph | None = None


def get_graph() -> KnowledgeGraph:
    global _graph
    if _graph is None:
        from .store import get_store
        g = KnowledgeGraph()
        g.load_seed()
        store = get_store()
        g.source_quality = {sid: (s.quality.overall if s.quality else 0.5) for sid, s in store.sources.items()}
        for sid in store.sources:
            g.extract_from_text(store.document_text(sid), sid)
        _graph = g
    return _graph
