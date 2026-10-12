"""Step 2: retrieval, falsification search, citation graph, knowledge graph, source quality."""
from truth_engine.retrieval import CitationGraph, EvidenceEngine, get_graph, get_store
from truth_engine.schemas import Claim, ClaimKind
from truth_engine.textutils import entities


def _gather(text, causal=False):
    c = Claim(text=text, kind=ClaimKind.FACTUAL, entities=entities(text), is_causal=causal)
    eng = EvidenceEngine()
    return eng.gather(c, eng.plan(c, ["corpus", "graph"]))


def test_store_loads_corpus():
    s = get_store().stats()
    assert s["documents"] >= 37 and s["passages"] > 90


def test_hybrid_search_finds_answer():
    top = get_store().search("Who discovered penicillin?", k=1)[0]
    assert top.source_id == "enc-penicillin" and "Fleming" in top.text


def test_falsification_search_surfaces_failed_replication():
    res = _gather("Veridamine improves sleep quality by 18%.")
    contra = [r for r in res if r.found_by == "contradiction_search"]
    assert any(r.source.id == "okafor-2024" for r in contra)
    assert max(r.contradiction_potential for r in res if r.source.id == "okafor-2024") > 0.5


def test_falsification_search_surfaces_nobel_correction():
    res = _gather("Einstein won the Nobel Prize in Physics for the theory of relativity.")
    assert any("not awarded" in r.passage.text and r.contradiction_potential > 0.3 for r in res)


def test_causal_claim_finds_alternative_explanation():
    res = _gather("The Halcyon X1 was discontinued in 2024 because demand declined.", causal=True)
    texts = " ".join(r.passage.text for r in res)
    assert "not by declining demand" in texts and "remained stable" in texts


def test_echo_detector_collapses_derived_sources():
    cg = CitationGraph(get_store().sources)
    n, ratio = cg.independence(["halvorsen-2021", "news-veridamine", "blog-veridamine", "vendor-veridamine"])
    assert (n, ratio) == (1, 0.25)
    # a new trial that cites the pilot is still independent of it
    n2, _ = cg.independence(["halvorsen-2021", "okafor-2024"])
    assert n2 == 2


def test_source_quality_grades_study_design():
    src = get_store().sources
    assert src["okafor-2024"].quality.tier == "A"
    assert src["vendor-veridamine"].quality.tier == "D"
    assert src["halvorsen-2021"].quality.overall < src["okafor-2024"].quality.overall
    assert "industry-funded" in src["halvorsen-2021"].quality.notes


def test_multi_hop_graph_answer():
    best = get_graph().multi_hop("In which country was the person who developed general relativity born?")[0]
    assert best.answer == "Germany"
    assert [e.rel for e in best.edges] == ["developed", "born_in", "located_in"]


def test_graph_prefers_high_quality_edges():
    best = get_graph().multi_hop("What is the capital of Australia?")[0]
    assert best.answer == "Canberra"


def test_web_source_grading():
    from truth_engine.retrieval.web import _authority
    assert _authority("https://www.sciencedirect.com/science/article/x")[1] == "paper"
    assert _authority("https://pmc.ncbi.nlm.nih.gov/articles/PMC1/")[1] == "paper"
    assert _authority("https://www.bioabsorbnutraceuticals.com/blog")[1] == "marketing"
    assert _authority("https://www.reddit.com/r/sleep")[1] == "forum"
    assert _authority("https://www.cdc.gov/sleep")[1] == "government"
