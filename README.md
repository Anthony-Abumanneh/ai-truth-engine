# AI Truth Engine

**An evidence auditor for scientific and health claims.**

Paste in any text — a ChatGPT answer, a news article, a supplement ad, a
report — and the Truth Engine checks it claim by claim against the research
literature. For each claim it tells you whether the evidence supports it,
contradicts it, or doesn't exist, how strong that evidence is, and how many
*independent* studies actually stand behind it.

You can also just ask a question ("Does creatine improve memory?") and get an
answer built the same way: from verified claims, with calibrated confidence,
and an explicit "the evidence isn't good enough to say" when that's the truth.

> **Status:** in active development. See [Build status](#build-status).
> Not medical advice: this tool summarizes published evidence, it does not
> replace a clinician.

---

## Why this exists

Large language models are fluent, not careful. They state claims with the
same confidence whether the evidence is a 50,000-person meta-analysis or a
single blog post, they cite sources that don't say what they claim, and when
several models agree it's often because they learned the same mistake.

Science and health information has a specific version of this problem: the
**echo chamber**. One small, unreplicated study gets covered by a news site,
repeated by twenty blogs, quoted on a product page, and absorbed into AI
training data. Search for the claim and you find dozens of sources that all
agree — and all trace back to the same 48-person pilot trial that a larger
trial later failed to replicate.

The Truth Engine is built to see through that.

## What makes it different

| Feature | What it does |
|---|---|
| **Audit mode** | Checks existing text claim by claim instead of only generating new answers. It doesn't have to be the best writer, only the most careful checker. |
| **Echo detector** | Follows citation and derivation links to count *independent lines of evidence*. "12 sources agree" becomes "12 sources, 1 underlying study". |
| **Falsification search** | For every claim it runs a separate search for evidence that would prove it *wrong*, not just evidence that agrees. |
| **Study-quality grading** | Weighs evidence by design (meta-analysis > RCT > cohort > animal > in-vitro > opinion), sample size, preregistration, funding source, and retraction status. |
| **Science red flags** | Catches the classic distortions: mouse results stated as human results, correlation stated as causation, relative risk without absolute risk, single unreplicated studies, retracted papers. |
| **Calibrated confidence** | Confidence is computed from evidence signals, not asked of a model, and measured: when it says 80%, it should be right about 80% of the time. A public report card proves it. |
| **Abstention** | When the evidence is thin or conflicting it says so, instead of manufacturing certainty. |
| **Watched claims** | Save a claim and the engine re-checks it later, alerting you when a new study or a retraction changes its status. |

## Two modes

**Audit** — input: any text. Output: every check-worthy claim highlighted and labeled.
Illustrative output, using the fictional demo corpus:

```
"Veridamine improves sleep quality by 18% and is clinically proven."

  ✗ CONTRADICTED   "improves sleep quality by 18%"
      The only positive result is a 48-person, manufacturer-funded pilot
      (2021). A preregistered 612-person trial (2024) found no significant
      effect (2%, p = 0.41).
      Echo check: 4 supporting sources → 1 independent study.

  ✗ CONTRADICTED   "clinically proven"
      A 2025 systematic review rates the evidence as low certainty.

  Confidence: HIGH that the claim is not supported.
```

**Ask** — input: a question. Output: an answer assembled from verified
claims, with sources tied to each statement, a confidence level, and the
uncertainty that remains.

## How it works

```
 INPUT (text to audit, or a question)
   │
   ▼
 ANALYZE ─────────── what kind of claim/question? how hard? how risky?
   │
   ▼
 ROUTE ───────────── which models, tools and sources? how much compute?
   │                 (easy → cheap and fast; hard → ensemble + deep checks)
   ▼
 EXTRACT CLAIMS ──── break text into atomic, checkable statements
   │
   ├──────────────────────────────┐
   ▼                              ▼
 SUPPORTING SEARCH         FALSIFICATION SEARCH
   │   PubMed · OpenAlex · local corpus · web · knowledge graph
   └──────────────┬───────────────┘
                  ▼
 GRADE EVIDENCE ──── study design, sample size, funding, retractions,
   │                 and independence via the citation graph (echo detector)
   ▼
 VERIFY ──────────── each claim: supported / partially supported /
   │                 contradicted / unverified / inferred / speculative
   ▼
 CRITIQUE ────────── an adversarial pass assumes each verdict is wrong and
   │                 hunts for counterevidence and red flags
   ▼
 JUDGE + CALIBRATE ─ confidence from many signals, then calibrated
   │
   ▼
 REPORT or ABSTAIN ─ verdicts, evidence trail, sources, uncertainty
```

Every investigation is saved as a structured trace. Those traces train the
router over time, so it learns which models and sources work best for which
kinds of claims while keeping cost and latency down.

## Works with or without API keys

The engine runs **fully offline** on deterministic, built-in reasoners, so
everything can be developed, tested and demoed without spending anything.
Add keys and it upgrades automatically — no code changes:

| Variable | Enables |
|---|---|
| `ANTHROPIC_API_KEY` | Claude models |
| `OPENAI_API_KEY` | OpenAI models (and optional OpenAI embeddings) |
| `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_MODELS` | Any OpenAI-compatible server: Qwen via DashScope, Together, vLLM, Ollama |
| `TAVILY_API_KEY` / `BRAVE_API_KEY` / `SERPER_API_KEY` | Live web search |
| `NCBI_API_KEY` *(optional)* | Higher PubMed rate limits (PubMed works without a key) |
| `OPENALEX_EMAIL` *(optional)* | OpenAlex "polite pool" (OpenAlex works without it) |
| `TE_EMBEDDINGS=openai` or `st` | Neural embeddings instead of the offline hashing embedder |
| `TE_OFFLINE=1` | Force offline mode even when keys are present |

Model names per role can be overridden with `TE_ANTHROPIC_STRONG`,
`TE_ANTHROPIC_MID`, `TE_ANTHROPIC_FAST`, `TE_OPENAI_STRONG`, `TE_OPENAI_FAST`.

## Measuring whether it actually works

The central research question, from the original specification:

> Can an adaptive system that combines routing, specialized ensembles, claim
> decomposition, adaptive retrieval, support + contradiction search, claim
> verification, adversarial critique, calibrated uncertainty, abstention and
> compute-aware escalation produce higher factual reliability and better
> cost-adjusted accuracy than single-model, standard-RAG, or naive
> multi-model systems?

The benchmark harness answers it with numbers:

- **Datasets:** [SciFact](https://github.com/allenai/scifact) (expert-labeled
  scientific claims), plus a built-in science benchmark with echo-chamber,
  retraction and overgeneralization traps.
- **Baselines:** (A) single LLM, (B) LLM + standard RAG, (C) multiple models +
  majority vote, (D) multiple models + RAG + judge — vs. the full system.
- **Metrics:** verdict accuracy, claim-level precision/recall, citation
  correctness, contradiction detection, abstention quality, calibration
  (expected calibration error, reliability diagram), latency, cost, and
  accuracy per dollar.
- **Ablations:** remove one component at a time to find out which ones
  actually matter.

## Project layout

```
truth_engine/
  schemas.py          data model; the full investigation trace (spec §14)
  config.py           environment-driven settings
  textutils.py        claims, numbers, entities, negation, causal language
  providers/          model adapters (Anthropic, OpenAI, OpenAI-compatible, offline)
  retrieval/          document store, BM25 + vector hybrid search, web search,
                      knowledge graph, citation graph, source quality,
                      support + falsification evidence engine
  science/            PubMed, OpenAlex, study-design grading, red flags      (in progress)
  pipeline/           analyzer, router, claims, verifier, critic, judge,
                      calibration, synthesizer, orchestrator                   (in progress)
  feedback/           trace store, learned router, watched claims             (planned)
  bench/              datasets, baselines, metrics, ablations, report card    (planned)
  api/                FastAPI server with streaming investigations            (planned)
web/                  audit-first web interface                               (planned)
data/
  corpus/             demo evidence corpus (Markdown + front matter)
  graph.json          knowledge-graph seed
tests/
```

## Build status

| # | Step | Status |
|---|---|---|
| 1 | Core schemas, config, model provider layer | ✅ Done |
| 2 | Retrieval and evidence engine (hybrid search, falsification search, citation graph, knowledge graph, source quality) | ✅ Done |
| 3 | Scientific evidence sources and study grading (PubMed, OpenAlex, retractions, study design) | ⏳ Next |
| 4 | Audit mode: claim extraction, verifier, science red-flag checks | Planned |
| 5 | Analyzer, router, compute budgets, independent solvers | Planned |
| 6 | Critic, judge, calibrated confidence, abstention | Planned |
| 7 | Orchestrator, evidence tree, feedback loop, watched claims | Planned |
| 8 | Benchmark, baselines, ablations, calibration report card | Planned |
| 9 | API, CLI and audit-first web interface | Planned |
| 10 | Tests, end-to-end verification, docs | Planned |

## About the demo corpus

`data/corpus/` ships with a small evidence collection used for offline
development and tests:

- **Encyclopedia** — real, stable facts, deliberately mixed with low-quality
  sources that contain well-known errors (forum posts, listicles, quiz
  sites), so error detection can be tested.
- **Research** — a **fictional** supplement, *veridamine*, with a complete
  echo chamber: one small pilot study, three sources that repeat it, a large
  trial that failed to replicate it, and a systematic review.
- **Products** and **Company** — **fictional** organizations used to test
  claim decomposition and causal reasoning from the original specification.

Fictional entities exist only to exercise the engine; none of them describe
real products, companies or research.

## Requirements

Python 3.11+, plus `fastapi`, `uvicorn`, `pydantic`, `httpx`, `numpy`, `sympy`.
Install and run instructions will be added when the server and interface land
(step 9).

## Background

This project implements and extends the *AI Truth Engine — Technical Product
& Research Specification* (v1.0, October 2026), narrowed to scientific and
health claims and extended with audit mode, the echo detector, watched claims
and a public calibration report card. [`docs/CONCEPTS.md`](docs/CONCEPTS.md)
lists the ideas you need to understand how it works.
