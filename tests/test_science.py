"""Step 3: scientific evidence sources (PubMed, OpenAlex), cache, study grading.

API tests replay real responses recorded in tests/fixtures/ through a mock
HTTP transport, so they run offline and deterministically. The live smoke
test at the bottom runs only when PubMed is reachable."""
import json
import socket
from pathlib import Path

import httpx
import pytest

from truth_engine.retrieval import CitationGraph, EvidenceEngine, get_store
from truth_engine.schemas import Claim, ClaimKind
from truth_engine.science.cache import JsonCache
from truth_engine.science.openalex import OpenAlexClient, abstract_from_index, parse_work
from truth_engine.science.papers import to_source
from truth_engine.science.pubmed import PubMedClient, parse_efetch
from truth_engine.science.retractions import CrossrefClient
from truth_engine.science.search import ScienceSearch, query_terms
from truth_engine.science.study_design import classify, sample_size
from truth_engine.textutils import entities

FIX = Path(__file__).parent / "fixtures"


def _fixture_transport(calls: list):
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        path, q = req.url.path, req.url.params
        if path.endswith("esearch.fcgi"):
            return httpx.Response(200, text=(FIX / "pubmed_esearch.json").read_text())
        if path.endswith("efetch.fcgi"):
            name = "pubmed_efetch_retracted.xml" if "9500320" in q["id"] else "pubmed_efetch.xml"
            return httpx.Response(200, text=(FIX / name).read_text())
        if path == "/works/pmid:9500320":
            return httpx.Response(200, text=(FIX / "openalex_work_retracted.json").read_text())
        if req.url.host == "api.crossref.org":
            return httpx.Response(200, text=(FIX / "crossref_updates.json").read_text())
        if path == "/sources":
            return httpx.Response(200, text=(FIX / "openalex_sources.json").read_text())
        if path == "/works" and q.get("filter", "").startswith("cites:"):
            return httpx.Response(200, text=(FIX / "openalex_cited_by.json").read_text())
        if path == "/works":
            return httpx.Response(200, text=(FIX / "openalex_search.json").read_text())
        return httpx.Response(404, text="not found")
    return httpx.MockTransport(handler)


@pytest.fixture
def science(tmp_path):
    calls: list[str] = []
    client = httpx.Client(transport=_fixture_transport(calls))
    cache = JsonCache(tmp_path / "cache")
    s = ScienceSearch(PubMedClient(cache=cache, client=client, offline=False, api_key=""),
                      OpenAlexClient(cache=cache, client=client, offline=False, email=""),
                      CrossrefClient(cache=cache, client=client, offline=False, email=""))
    s.calls = calls
    return s


# ------------------------------------------------------------------ study design
def test_design_from_publication_types_beats_text():
    p = classify("A pilot", "We did a cohort thing.", pub_types=["Journal Article", "Randomized Controlled Trial"])
    assert p.design == "rct" and p.weight == 0.85


def test_design_from_text_ignores_negated_mentions():
    p = classify("Tea and dementia: a prospective cohort study",
                 "No randomized trial has been conducted. 24,000 adults were followed for 12 years.")
    assert p.design == "cohort" and p.sample_size == 24000


def test_animal_study_detected_and_downgraded():
    p = classify("Drug X in mice", "A randomized study of 40 mice showed longer lifespan.")
    assert p.species == "animal" and p.design == "animal" and p.weight < 0.3
    assert classify("t", "x", mesh=["Animals", "Mice"]).species == "animal"
    assert classify("t", "x", mesh=["Humans", "Mice"]).species == "human"


def test_sample_size_prereg_funding():
    assert sample_size("A trial (n = 612) in 2024") == 612
    assert sample_size("We enrolled 1,250 participants in 2019.") == 1250
    p = classify("t", "Registered at ClinicalTrials.gov (NCT01234567). Funded by the manufacturer.")
    assert p.preregistered is True and p.funding == "industry"


def test_review_does_not_inherit_flags_of_studies_it_reviews():
    p = classify("Supplements for insomnia: a systematic review",
                 "One small manufacturer-funded pilot was positive; a larger preregistered trial was null.")
    assert p.design == "systematic_review" and p.funding is None and p.preregistered is None


# ------------------------------------------------------------------ parsers
def test_parse_pubmed_efetch():
    recs = parse_efetch((FIX / "pubmed_efetch.xml").read_text())
    assert [r.pmid for r in recs] == ["35984306", "39070254", "37368234"]
    meta = recs[0]
    assert "Meta-Analysis" in meta.pub_types and "Humans" in meta.mesh
    assert meta.published == "2023-03-10" and meta.journal == "Nutrition reviews"
    assert len(meta.references) > 20 and all(r.startswith("pmid:") for r in meta.references)
    src, passages = to_source(meta)
    assert src.id == "pmid:35984306" and src.source_type == "review" and src.quality.tier == "A"
    assert passages[0].text == meta.title and len(passages) > 5


def test_pubmed_retracted_paper_is_tier_d():
    rec = parse_efetch((FIX / "pubmed_efetch_retracted.xml").read_text())[0]
    assert rec.retracted
    src, _ = to_source(rec)
    assert src.quality.tier == "D" and src.quality.notes[0].startswith("RETRACTED")
    assert src.metadata["study_label"].startswith("RETRACTED")


def test_parse_openalex_work():
    rec = parse_work(json.loads((FIX / "openalex_work_retracted.json").read_text()))
    assert rec.retracted and rec.pmid == "9500320" and rec.openalex_id == "W2117847125"
    assert not rec.title.startswith("RETRACTED")
    assert rec.references and rec.references[0].startswith("openalex:W")
    assert abstract_from_index({"evidence": [1], "Weak": [0]}) == "Weak evidence"


# ------------------------------------------------------------------ clients + cache
def test_pubmed_client_uses_cache_and_offline_mode(tmp_path):
    calls: list[str] = []
    client = httpx.Client(transport=_fixture_transport(calls))
    cache = JsonCache(tmp_path)
    pm = PubMedClient(cache=cache, client=client, offline=False, api_key="")
    recs = pm.search_records("creatine supplementation memory", retmax=3)
    assert len(recs) == 3 and len(calls) == 2
    pm.search_records("creatine supplementation memory", retmax=3)
    assert len(calls) == 2                      # second time served from disk
    offline = PubMedClient(cache=cache, client=client, offline=True, api_key="")
    assert len(offline.search_records("creatine supplementation memory", retmax=3)) == 3
    assert offline.search_records("something never fetched") == [] and len(calls) == 2


def test_cache_key_ignores_secrets(tmp_path):
    c = JsonCache(tmp_path)
    assert c.key("pubmed", "u", {"term": "x", "api_key": "SECRET"}) == c.key("pubmed", "u", {"term": "x"})
    c.put("pubmed", "u", {"term": "x", "api_key": "SECRET"}, "body")
    assert "SECRET" not in next(tmp_path.rglob("*.json")).read_text()


def test_network_failure_degrades_gracefully(tmp_path):
    def boom(req):
        raise httpx.ConnectError("no network")
    pm = PubMedClient(cache=JsonCache(tmp_path), client=httpx.Client(transport=httpx.MockTransport(boom)),
                      offline=False, api_key="")
    assert pm.search("anything") == [] and pm.errors


# ------------------------------------------------------------------ science search
def test_query_terms_drop_finding_words_and_numbers():
    assert query_terms("Creatine supplementation improves memory by 15% in healthy adults") == \
        ["creatine", "supplementation", "memory", "healthy", "adults"]


def test_pubmed_and_openalex_records_merge(science):
    results = science.search("Creatine supplementation improves memory")
    ids = [s.id for s, _ in results]
    assert ids.count("pmid:35984306") == 1                      # same paper from both APIs, merged once
    merged = science.sources["pmid:35984306"]
    assert merged.metadata["openalex_id"] == "W4294349271"
    assert any(c.startswith("openalex:") for c in merged.cites)  # OpenAlex references added


def test_review_derives_from_pooled_trials():
    from truth_engine.science.papers import PaperRecord
    s = ScienceSearch(PubMedClient(offline=True), OpenAlexClient(offline=True), CrossrefClient(offline=True))
    trial_a = PaperRecord(title="Drug Y trial A", abstract="A randomized controlled trial of 300 adults.", pmid="1")
    trial_b = PaperRecord(title="Drug Y trial B", abstract="A randomized controlled trial of 500 adults.", pmid="2")
    review = PaperRecord(title="Drug Y: a meta-analysis", pmid="3", pub_types=["Meta-Analysis"],
                         references=["pmid:1", "pmid:2", "pmid:999"])
    s._ingest([trial_a, trial_b, review])
    assert set(s.sources["pmid:3"].derived_from) == {"pmid:1", "pmid:2"}
    cg = CitationGraph(s.sources)
    assert cg.independence(["pmid:1", "pmid:2", "pmid:3"])[0] == 2    # review adds no new lineage
    assert cg.independence(["pmid:3"])[0] == 1


def test_evidence_engine_gathers_literature(science):
    claim = Claim(text="Creatine supplementation improves memory.", kind=ClaimKind.FACTUAL,
                  entities=entities("Creatine supplementation improves memory."))
    eng = EvidenceEngine(science=science)
    res = eng.gather(claim, eng.plan(claim, ["pubmed", "openalex"]))
    lit = [r for r in res if r.source.id.startswith(("pmid:", "openalex:"))]
    assert lit and any(r.source.id == "pmid:35984306" for r in lit)
    assert set(eng.last_external_sources) >= {r.source.id for r in lit}


# ------------------------------------------------------------------ expanded demo corpus
def test_demo_corpus_science_cases_are_graded():
    src = get_store().sources
    assert src["tanaka-2019"].quality.tier == "D"                               # retracted
    assert src["moreau-2022"].metadata["study"]["design"] == "animal"
    assert src["ito-2022"].metadata["study"]["design"] == "in_vitro"
    assert src["lindqvist-2020"].metadata["study"]["design"] == "cohort"        # observational only
    assert src["castillo-2023"].metadata["study"]["funding"] == "industry"
    assert src["okafor-2024"].quality.overall > src["halvorsen-2021"].quality.overall


def test_demo_echo_chambers_collapse():
    cg = CitationGraph(get_store().sources)
    assert cg.independence(["tanaka-2019", "news-lumorex", "vendor-lumorex"])[0] == 1
    assert cg.independence(["ito-2022", "vendor-nervaplex", "forum-nervaplex"])[0] == 1


def test_falsification_surfaces_retraction_notice():
    text = "Lumorex reverses memory loss in older adults by 40%."
    c = Claim(text=text, kind=ClaimKind.FACTUAL, entities=entities(text))
    eng = EvidenceEngine()
    res = eng.gather(c, eng.plan(c, ["corpus"]))
    assert any(r.source.id == "retraction-tanaka-2019" for r in res)


# ------------------------------------------------------------------ live smoke test
def _online(host="eutils.ncbi.nlm.nih.gov") -> bool:
    try:
        socket.create_connection((host, 443), timeout=3).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _online(), reason="PubMed not reachable")
def test_live_pubmed_and_openalex(tmp_path):
    cache = JsonCache(tmp_path)
    s = ScienceSearch(PubMedClient(cache=cache, offline=False), OpenAlexClient(cache=cache, offline=False))
    res = s.search("creatine supplementation memory", k=3)
    assert res, s.errors
    assert all(src.quality is not None for src, _ in res)
    wake = s.lookup("pmid:9500320")
    assert wake and wake[0].quality.tier == "D"


# ------------------------------------------------------------------ step 3.5: journals, retractions, relevance
def test_venue_signals_attached_and_scored(science):
    science.search("Creatine supplementation improves memory")
    src = science.sources["pmid:35984306"]
    venue = src.metadata["venue"]
    assert venue["name"] == "Nutrition Reviews" and venue["vetted"] and 0.7 < venue["standing"] < 0.8


def test_unvetted_journal_is_flagged():
    from truth_engine.science.papers import PaperRecord
    from truth_engine.science.venues import Venue
    base = dict(title="Drug Z trial", abstract="A randomized controlled trial of 300 adults.", pmid="7")
    vetted = to_source(PaperRecord(**base, venue=Venue(name="Good J", mean_citedness=3, listed_in=["medline"])))[0]
    unvetted = to_source(PaperRecord(**base, venue=Venue(name="Fast Pub J", mean_citedness=0.1)))[0]
    assert unvetted.quality.overall < vetted.quality.overall - 0.1
    assert any("possible predatory venue" in n for n in unvetted.quality.notes)


def test_crossref_retraction_watch_notices(tmp_path):
    calls: list[str] = []
    cr = CrossrefClient(cache=JsonCache(tmp_path), client=httpx.Client(transport=_fixture_transport(calls)),
                        offline=False, email="")
    n = cr.notices(["10.1016/S0140-6736(97)11096-0", "10.1093/nutrit/nuac064"])
    assert list(n) == ["10.1016/s0140-6736(97)11096-0"]
    assert n["10.1016/s0140-6736(97)11096-0"][0]["type"] == "retraction"
    assert n["10.1016/s0140-6736(97)11096-0"][0]["source"] == "retraction-watch"


def test_crossref_notice_alone_marks_paper_retracted():
    from truth_engine.science.papers import PaperRecord
    rec = PaperRecord(title="A trial", abstract="A randomized trial of 200 adults.", doi="10.1/x",
                      notices=[{"type": "retraction", "doi": "10.1/y", "date": "2020-01-01"}])
    assert to_source(rec)[0].quality.tier == "D"


def test_subject_gate_drops_off_topic_passages():
    from truth_engine.retrieval.evidence import claim_subject, mentions
    claim = Claim(text="Vitamin D supplements prevent depression.", kind=ClaimKind.FACTUAL,
                  entities=entities("Vitamin D supplements prevent depression."))
    assert claim_subject(claim) == "Vitamin D"
    assert not mentions("Vitamin D", "Marine omega-3 trial. Omega-3 supplements did not prevent depression.")
    assert mentions("Vitamin D", "Effect of vitamin D3 vs placebo. No difference in depression scores.")
    assert mentions("creatine", "Creatine's effects on cognition")
