"""Where does a live audit spend its time? Times every external call and model step.

    python scripts/profile_live.py "claim one. claim two."
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.science import cache as cache_mod  # noqa: E402
from truth_engine.retrieval import web as web_mod  # noqa: E402
from truth_engine.pipeline import nli as nli_mod  # noqa: E402

T = defaultdict(float)
N = defaultdict(int)


def timed(label, fn):
    def wrap(*a, **k):
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            T[label] += time.time() - t
            N[label] += 1
    return wrap


orig_get = cache_mod.CachedFetcher.get
cache_mod.CachedFetcher.get = lambda self, url, params=None, attempts=3: timed(
    f"http:{self.namespace}", orig_get)(self, url, params, attempts)
web_mod.WebSearch._cached_raw = timed("http:web", web_mod.WebSearch._cached_raw)
nli_mod.NLIModel.predict = timed("nli", nli_mod.NLIModel.predict)

from truth_engine.pipeline.audit import audit  # noqa: E402
from truth_engine.retrieval.store import DocumentStore  # noqa: E402
DocumentStore.search = timed("local search", DocumentStore.search)

text = " ".join(sys.argv[1:]) or "Melatonin improves sleep onset in adults. Zinc shortens the common cold."
t0 = time.time()
rep = audit(text, live=True)
total = time.time() - t0
print(f"\ntotal {total:.1f}s for {len(rep.claims)} claim(s)")
for k in sorted(T, key=T.get, reverse=True):
    print(f"  {k:22} {T[k]:6.1f}s  ({N[k]} calls)")
for c, v in zip(rep.claims, rep.verdicts):
    print(f"  {v.status.value:12} {[p.get('tier') for p in rep.routing[c.id]['passes']]} {c.text}")
