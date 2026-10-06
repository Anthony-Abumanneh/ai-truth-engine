# CLAUDE.md — AI Truth Engine

Read this first. It carries everything decided so far, written when the
project moved from a claude.ai cloud session to local Claude Code. The
original spec is `docs/spec/AI_Truth_Engine_Specification_v1.docx`; the
README explains the product; `docs/CONCEPTS.md` is the learning guide.

## Working agreement with the user

- **Stop after each step** in the plan below and give a short, plain-language
  update (what was built, what the tests showed, any limitations, what's
  next), then wait for a go-ahead before starting the next step.
- The user is learning while building. Explain concepts plainly and avoid
  unexplained jargon. When a step introduces a concept, point to the
  matching section of `docs/CONCEPTS.md`.
- Keep the README's **Build status** table and project layout current as
  steps complete.
- Be honest about limitations: say what is simulated, fictional or not
  yet working.

## What we're building

The spec describes a general "AI Truth Engine": route a question to the
right models and tools, generate independent answers, split them into
claims, retrieve supporting **and** contradicting evidence, verify each
claim, run an adversarial critic, calibrate confidence, and answer or
abstain.

**Decision (pivot):** we narrowed it to an **evidence auditor for
scientific and health claims**, with two modes:

1. **Audit mode (primary):** paste any text (an AI answer, an article, an
   ad) → every check-worthy claim is labeled supported / partially
   supported / contradicted / unverified / inferred / speculative, with
   evidence, study quality, and independence.
2. **Ask mode:** a question → an answer assembled only from verified
   claims, with calibrated confidence or explicit abstention.

Signature features: echo detector (N sources → K independent studies via
the citation/derivation graph), falsification search, study-quality
grading, science red flags (animal→human, correlation→causation,
relative vs absolute risk, unreplicated/small/industry-funded, retracted),
calibration report card (reliability diagram + ECE), watched claims.

Not medical advice; the UI and README must say so.

## Core design principles

1. **Offline-first, deterministic core.** Every component that can be an
   algorithm *is* one (retrieval, contradiction signals, source quality,
   citation independence, numeric checks, calibration). The engine must
   run end to end with no API keys.
2. **LLMs are an upgrade, not a dependency.** With keys present, stages
   use LLMs for free-form work (claim extraction from messy prose, an LLM
   verifier as a second opinion, critic, synthesis), but algorithmic checks
   still run and their disagreement with the LLM is itself a signal
   ("verifier agreement").
3. **Diversity is measured, not assumed.** Offline reasoners use genuinely
   different methods so their errors are less correlated; model `family`
   in the registry feeds correlated-error risk.
4. **Confidence is computed, not asked for.** Never ask a model "how
   confident are you?". Combine signals (spec §4, Improvement 9), then
   calibrate against benchmark outcomes.
5. **Cost-aware.** Offline reasoners carry *simulated* per-token prices by
   tier so accuracy-per-dollar is meaningful without API spend. Always
   label simulated costs as simulated.
6. **Fictional demo data is labeled fictional.** Veridamine, Northwind /
   Halcyon X1, and the "Acme Retail" company docs are invented test
   fixtures. Encyclopedia docs are real stable facts plus deliberately
   wrong low-quality sources.

## Plan and status

| # | Step | Status |
|---|---|---|
| 1 | Core schemas, config, provider layer | ✅ Done |
| 2 | Retrieval + evidence engine | ✅ Done |
| 3 | Scientific evidence sources + study grading | ⏳ **Next** |
| 4 | Audit mode: claim extraction, verifier, science red flags | Planned |
| 5 | Analyzer, router, compute budgets, independent solvers | Planned |
| 6 | Critic, judge, calibrated confidence, abstention | Planned |
| 7 | Orchestrator, evidence tree, feedback loop, watched claims | Planned |
| 8 | Benchmark (SciFact), baselines A–D, ablations, calibration report card | Planned |
| 9 | FastAPI server, CLI, audit-first web UI | Planned |
| 10 | Tests, end-to-end runs, final docs | Planned |

### Step 3 — scientific evidence sources + study grading (next)
New package `truth_engine/science/`:
- `pubmed.py` — NCBI E-utilities: `esearch` (term → PMIDs) and `efetch`
  (XML → title, abstract sentences, journal, year, publication types, MeSH).
  Optional `NCBI_API_KEY`; respect 3 req/s (10 with key). Convert to
  `Source` + sentence `Passage`s.
- `openalex.py` — search works, fetch `referenced_works` and `cited_by`,
  `is_retracted`, `type`, publication year, concepts. Optional
  `OPENALEX_EMAIL` for the polite pool. Feeds real edges into
  `CitationGraph` (`cites`), so the echo detector works on real literature.
- `study_design.py` — classify a paper/passage as meta-analysis,
  systematic review, RCT, cohort, case-control, cross-sectional, case
  report, animal, in-vitro, opinion/editorial, news, marketing. Use
  PubMed publication types/MeSH when present, regex cues otherwise. Extract
  sample size (n=…, "612 adults"), preregistration, funding/COI, species.
  Map to an evidence-hierarchy weight and feed `score_source`.
- `cache.py` — on-disk JSON cache (`data/cache/`) for every external call,
  so live lookups are reproducible offline and tests can use recorded
  responses (`tests/fixtures/`).
- Wire PubMed/OpenAlex in as retrieval sources ("pubmed", "openalex") in
  `EvidenceEngine.gather`, alongside "corpus", "web", "graph".
- Expand the science demo corpus: more echo-chamber, retraction,
  animal-only, observational-only and relative-risk cases. Fictional
  names only.
- Tests with recorded fixtures; a live smoke test that runs only when the
  network is reachable.

### Steps 4–10 (intended design)
- **4 Audit mode:** `pipeline/claims.py` (sentence → atomic claims;
  split conjunctions; tag kind: efficacy / safety / association /
  mechanism / statistic / causal; check-worthiness filter; keep char
  spans for highlighting). `pipeline/verifier.py`: per-evidence stance
  (entails / contradicts / neutral / partial) from coverage, numeric
  agreement (`Quantity.close_to`), negation and antonym conflicts, hedging,
  plus optional LLM verifier; aggregate per claim into the six statuses
  weighted by source quality and **independent lineage count**, not raw
  count. `science/red_flags.py` for the distortions listed above.
- **5 Router:** analyzer outputs the spec's JSON (domain, type,
  difficulty, risk, requires_retrieval, tools, compute_budget). Budget
  ladder minimal→maximum; escalate when confidence < threshold. Solvers for
  ask mode: the offline reasoners in `providers/registry.py`
  (`quick`, `semantic`, `careful`, plus symbolic/numeric/code/analyst
  strategies kept for general mode) and any LLMs present. Diversity report
  = clusters of normalized answers, majority share, family overlap.
- **6 Critic/judge/calibration:** critic assumes each verdict is wrong,
  issues extra falsification queries, flags red flags; judge selects or
  abstains; confidence = weighted signals (model agreement, evidence
  support, source quality, verifier agreement, historical reliability,
  independence, minus contradiction, unresolved uncertainty, difficulty),
  then Platt/isotonic calibration fit on stored outcomes. Labels VERY
  LOW…VERY HIGH. Abstain below threshold or when strong evidence conflicts.
- **7 Orchestrator:** stage events streamed (for the UI log); trace saved
  to SQLite with the §14 fields (`Trace.section14_record()`); learned
  router (logistic regression / contextual bandit) over stored traces;
  watched claims table + re-check job that diffs verdicts.
- **8 Benchmark:** `scripts/download_scifact.sh` (S3 tarball; run locally);
  loader; built-in science benchmark; baselines A single LLM, B LLM+RAG,
  C multi-model majority vote, D multi-model+RAG+judge, vs full system;
  metrics: verdict accuracy, claim P/R/F1, citation correctness,
  contradiction detection, abstention quality, ECE/Brier + reliability
  diagram, latency, cost, accuracy per dollar; ablations remove one
  component at a time.
- **9 UI:** FastAPI + SSE; single-page UI served by the API, audit-first:
  paste box, highlighted claims, verdict cards, evidence/study-quality
  panels, echo-detector lineage view, investigation tree, ask mode,
  watched claims, report card.
- **10:** pytest suite across components, end-to-end runs, docs.

## Code map (what exists now)

```
truth_engine/
  schemas.py         Pydantic types for every stage + Trace (spec §14 export)
  config.py          env-driven Settings; model names per role
  textutils.py       sentences, stemming, quantities/dates/units, entities,
                     negation, antonyms, causal markers, hedges
  providers/
    base.py          ModelSpec, LLMProvider, JSON extraction, retry/backoff
    http_providers.py Anthropic Messages, OpenAI Chat, OpenAI-compatible
    registry.py      9 offline reasoners + LLMs detected from env
  retrieval/
    store.py         Markdown+front-matter corpus → sentence passages;
                     BM25 + vector hybrid via reciprocal rank fusion
    embeddings.py    hashing embedder (offline); OpenAI / sentence-transformers
    bm25.py          Okapi BM25
    source_quality.py authority, methodology (sample size, prereg, funding),
                     directness, recency, primary → tier A–D
    citation_graph.py derived_from collapses independence; cites does not;
                     same publisher = one lineage (union-find)
    knowledge_graph.py curated seed (data/graph.json) + pattern extraction;
                     multi-hop BFS scored by relation triggers, target type,
                     and weakest-edge source quality
    web.py           Tavily / Brave / Serper adapters
    evidence.py      claim → support queries + falsification queries
                     (value-stripped, polarity-flipped, dispute-framed,
                     effect-without-cause) → ranked Retrieved items with
                     contradiction_potential
data/corpus/         37 demo docs (encyclopedia, research, products, company)
data/graph.json      knowledge-graph seed with provenance per triple
```

## Environment notes

- Python 3.11+ (developed on 3.13). Install: `pip install -r requirements.txt`.
- Env vars for real models/search: see README "Works with or without API keys".
- Default Anthropic models: `claude-opus-5`, `claude-sonnet-5-5`,
  `claude-haiku-4-5-20251001`. Default OpenAI names (`gpt-4.1`,
  `gpt-4.1-mini`) may be stale; override with `TE_OPENAI_STRONG/FAST`.
- The cloud session could not reach PubMed, OpenAlex or the SciFact S3
  bucket. Local machines normally can; verify at the start of step 3.

## Known gaps / notes

- Hashing embedder is a weak semantic signal; prefer `TE_EMBEDDINGS=st`
  or `openai` when available.
- Knowledge-graph pattern extraction is narrow by design; LLM triple
  extraction for uploaded docs is planned.
- The business demo (spec §6) still needs its SQLite sales DB. Target
  numbers: total revenue −11.9%; Product C −41%; Florida accounts for 71%
  of Product C's decline; Florida Product C −43% (Florida ≈ 68% of C's
  August revenue); Product B −2%; Product A +3%. Must stay consistent
  with `data/corpus/company/*`. This is general mode, lower priority after
  the pivot.
- `contradiction_potential` flags "launched" vs "discontinued" as an
  antonym; harmless (the verifier decides stance) but noisy.

## Checks

```
python -m pytest -q          # unit tests
```
