"""Step 1: schemas, provider layer, text utilities."""
from truth_engine.providers import extract_json, get_registry
from truth_engine.schemas import Trace
from truth_engine.textutils import antonym_conflict, entities, has_negation, is_causal, quantities, stem


def test_trace_exports_section14_fields():
    rec = Trace(question="q").section14_record()
    assert set(rec) == {"query_id", "question", "question_type", "difficulty", "risk", "models_selected",
                        "model_outputs", "claims", "retrieved_sources", "evidence_scores", "contradictions",
                        "critic_results", "final_answer", "confidence", "cost", "latency", "human_feedback",
                        "benchmark_score"}


def test_offline_registry_has_diverse_reasoners():
    reg = get_registry()
    offline = [m for m in reg.models.values() if not m.is_llm]
    assert len(offline) >= 9
    assert len({m.family for m in offline}) == len(offline)  # every reasoner uses a different method


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure! ```json\n{"a": [1, 2], "b": "x}"}\n```') == {"a": [1, 2], "b": "x}"}
    assert extract_json('prefix {"nested": {"k": 1}} suffix') == {"nested": {"k": 1}}


def test_quantities_units_scales_and_dates():
    def q(s):
        return [(x.value, x.unit) for x in quantities(s)]
    assert q("sold 4 million units in 2022") == [(4_000_000.0, "unit"), (2022.0, "year")]
    assert q("Halcyon X1 reached 4.1 million units by December 31, 2023") == [(20231231.0, "date"), (4_100_000.0, "unit")]
    assert q("Marketing spend was $412,000, within 2%") == [(412_000.0, "usd"), (2.0, "%")]
    assert q("born on 14 March 1879") == [(18790314.0, "date")]


def test_polarity_and_causal_signals():
    assert antonym_conflict("revenue increased", "revenue decreased")
    assert antonym_conflict("demand declined", "demand remained stable")
    assert has_negation("The prize was not awarded for relativity")
    assert is_causal("discontinued because demand declined")
    assert stem("launched") == stem("launches")


def test_entities():
    ents = entities("Albert Einstein was born in Ulm in the Kingdom of Wurttemberg.")
    assert "Albert Einstein" in ents and "Ulm" in ents
