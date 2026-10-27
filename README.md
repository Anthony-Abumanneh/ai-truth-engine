# AI Truth Engine: Evidence Auditor for Science & Health Claims

Paste in any text (a ChatGPT answer, a news article, a supplement ad) and the engine checks it
claim by claim against the research literature. It labels each claim **supported**,
**contradicted** or **unverified**, shows the evidence, counts how many *independent* studies
stand behind it, and says "the evidence isn't good enough" instead of guessing.

> Not medical advice: this tool summarizes published evidence; it does not replace a clinician.

## What We Built

- **RAG retrieval:** hybrid BM25 + embedding search (bge-base) over a local corpus, plus live
  PubMed, OpenAlex, Crossref and web search (Tavily) through their REST APIs.
- **NLP verification:** text is split into atomic claims; a pretrained DeBERTa-v3-large NLI model
  (PyTorch), fused with rule-based signals, judges whether each sentence supports or contradicts a claim.
- **Falsification search:** a separate search for evidence that would prove each claim *wrong*.
- **Echo detector:** a citation graph collapses sources that restate the same study, so
  "12 sources agree" becomes "12 sources, 1 underlying study".
- **Study grading and red flags:** design (meta-analysis > RCT > cohort > animal > cell), sample
  size, funding and retractions, plus flags for mouse→human, correlation→causation and relative-only risk.
- **Calibrated confidence and abstention:** a logistic-regression calibrator trained on 9 evidence
  signals; a judge withholds low-confidence verdicts.
- **Adaptive routing:** easy claims get a cheap check; weak or one-sided verdicts escalate to
  deeper searches. A LinUCB contextual bandit can learn this policy from feedback.
- **Ask mode, evidence trees, saved traces (SQLite), user feedback and watched claims.**

## Results (SciFact, held-out dev set)

[SciFact](https://github.com/allenai/scifact): expert-labeled scientific claims checked against
5,183 abstracts. All tuning (model choice, thresholds, calibrator) used the **train** split only;
results are on the held-out **dev** split unless marked.

| Component | Before | After |
|---|---|---|
| Retrieval: right abstract in top 5 (keyword BM25 → bge-base embedding search; 300 *train* claims, used to pick the model) | 85% | **94%** |
| Evidence judgment, macro-F1 (rules → rules + DeBERTa-v3-large NLI) | 0.44 | **0.77** |
| Calibration error, ECE (raw → trained calibrator) | 0.25 | **0.043** |
| Evidence judgment after fine-tuning DeBERTa (large / base) on SciFact + HealthVer + COVID-Fact, A100 | 0.74 / 0.68 | **0.81 / 0.82** |
| Expert's evidence among the sources the engine cites (word-overlap gate → MiniLM reranker gate) | 58% | **70%** |
| Accuracy when the engine states a verdict | 73% | **77%** |
| Study design identified from abstract text (rules → rules + TF-IDF classifier trained on ~7,300 PubMed-tagged abstracts; held-out abstracts) | 45% | **80%** |

Fine-tuning ([`training/finetune_verifier.ipynb`](training/finetune_verifier.ipynb)) improved judging
of expert-chosen evidence, but not the end-to-end pipeline (0.58 vs 0.61, not significant). A
diagnosis (`scripts/diagnose_scifact.py`) then showed why: for 27% of claims the expert's evidence
sentence *was* retrieved but a word-overlap filter stopped the model from reading it. A small
reranker now decides relevance by meaning, so the engine uses the right evidence far more often;
final accuracy stays ~0.60 because the remaining errors are in judging and combining evidence.

**End to end vs. LLM baselines** (same 100 dev claims; full report in
[`docs/benchmark_report.md`](docs/benchmark_report.md)):

| System | Accuracy | Calibration error (ECE) | Cost |
|---|---|---|---|
| **Truth Engine (local DeBERTa + bge-base models)** | 0.60 | **0.043** | $0 |
| Truth Engine evidence + gpt-oss-120b as judge | 0.74 | **0.054** | free tier |
| gpt-oss-120b + standard RAG | **0.86** | 0.088 | free tier |
| gpt-oss-120b + RAG + few-shot examples | 0.80 | 0.083 | free tier |
| 3 LLMs + RAG + LLM judge | 0.80 | 0.095 | free tier |
| gpt-oss-120b alone (no retrieval) | 0.52 | 0.325 | free tier |
| 3-LLM majority vote (no retrieval) | 0.44 | 0.442 | free tier |

**What this shows:** a large LLM given good retrieval still reasons better on raw accuracy.
The Truth Engine is the **best-calibrated system** tested, runs fully locally for $0, and beats
LLMs without retrieval. Ablations (300 dev claims, paired bootstrap) show the NLI model is the only
component with a significant accuracy effect (−15 points without it). Abstention trades 1.7 points of
overall accuracy for precision: verdicts the engine states are right 77% of the time with it, 68%
without. Turning off source-quality weighting scored +2.7 points on SciFact (not significant; the
default is unchanged, since tuning on dev would bias the result). SciFact has no echo chambers or low-quality sources, so the echo detector and
source grading are tested separately on a built-in trap set (in-sample; red flags 93%
precision / 100% recall). Its accuracy fell from 93% to 68% after the calibrator was refit: the
verdicts are still right, but general-knowledge contradictions now fall below the confidence bar,
because the calibrator learned from SciFact, where contradictions are right only ~63% of the time.
Calibrating on general-fact data (e.g. FEVER) is open work.

## Hardware & Performance

- **Hardware:** MacBook (Apple Silicon, 8 GB RAM), CPU only. Free-tier LLM APIs (Gemini, Groq)
  for baselines only.
- **Embedding the SciFact corpus (44,705 sentences, bge-base):** ~40 min once, then cached.
- **Full SciFact dev benchmark (300 claims):** ~8 min (~1.6 s per claim, 2 CPU threads).
- **Tests:** 133 tests (110 functions), 79% line coverage; `ruff` lint clean; also pass in a fresh clone
  (model-dependent tests skip until `setup` runs).
- **Live audit (PubMed + OpenAlex + web):** ~25–50 s for a typical 2–4 claim text, claims checked in parallel.

## Setup Instructions

```bash
git clone https://github.com/Anthony-Abumanneh/ai-truth-engine.git && cd ai-truth-engine
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m truth_engine setup                         # models: NLI verifier, reranker, design classifier (~1 GB)
mkdir -p data/scifact && curl -L https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz | tar -xz -C data/scifact
cp .env.example .env                                          # optional keys: Tavily, NCBI, Gemini, Groq
```

All API keys are optional: without them it runs offline on the local corpus.

## Run It

```bash
.venv/bin/python -m truth_engine serve            # web app at http://localhost:8000
.venv/bin/python -m truth_engine audit "Veridamine improves sleep quality by 18% and is clinically proven."
.venv/bin/python -m truth_engine ask --live "Does melatonin improve sleep onset in adults?"
```

The web app is audit-first: paste text and every check-worthy claim is highlighted in place and
colour-coded by verdict, with confidence, red flags, an echo check, and the strongest evidence for
and against (study type, quality grade, link). Progress streams live while it searches. Ask mode,
history, feedback buttons and watched claims are on the other tabs. JSON API: `POST /api/audit`,
`/api/audit/stream` (server-sent events), `/api/ask`, `/api/feedback`, `/api/watch`; see
`truth_engine/api/server.py`.

### Use it from Claude (MCP)

The engine is also an [MCP](https://modelcontextprotocol.io) server, so Claude and other AI
assistants can fact-check text, including their own draft answers. Tools: `audit_text`,
`ask_question`, `watch_claim`, `recheck_watched_claims`.

```bash
claude mcp add truth-engine -- /path/to/ai-truth-engine/.venv/bin/python -m truth_engine.mcp_server
```

Claude Desktop (`claude_desktop_config.json`):
```json
{"mcpServers": {"truth-engine": {"command": "/path/to/ai-truth-engine/.venv/bin/python",
                                 "args": ["-m", "truth_engine.mcp_server"]}}}
```

### Browser extension (Chrome)

Select text on any page → right-click → **Check with Truth Engine**, and a panel shows each claim's
verdict and evidence. Install: `chrome://extensions` → Developer mode → **Load unpacked** → choose the
`extension/` folder; keep the server running (`python -m truth_engine serve`). The extension only
reaches `localhost:8000` (or a server you set in its options) and only touches a page after you use
its menu item.

## How to Reproduce Results

```bash
.venv/bin/python -m pytest -q                                  # test suite
.venv/bin/python -m truth_engine.pipeline.audit "Veridamine improves sleep quality by 18% and is clinically proven."
.venv/bin/python -m truth_engine.pipeline.audit --live "Melatonin improves sleep onset in adults."

.venv/bin/python scripts/eval_retrieval_scifact.py 300        # embedding model comparison (train)
.venv/bin/python scripts/eval_stance_scifact.py dev           # rules vs NLI evidence judgment
.venv/bin/python scripts/calibrate.py                         # fit calibrator on train, report on dev
.venv/bin/python scripts/benchmark.py ours ablations builtin  # engine + ablations + trap set
.venv/bin/python scripts/benchmark.py llm --llm-n=100 --extras  # LLM baselines (needs GEMINI/GROQ keys)
.venv/bin/python scripts/benchmark.py report                  # rebuild docs/benchmark_report.md
```

## Project Structure

```
truth_engine/
  retrieval/   BM25, embeddings, hybrid store, evidence engine, knowledge graph, web search
  science/     PubMed / OpenAlex / Crossref clients, study grading, citation graph, red flags
  pipeline/    claims, NLI, verifier, analyzer, router, critic, confidence, judge, orchestrator
  feedback/    SQLite traces, user feedback, watched claims, router learning
  bench/       SciFact harness, LLM baselines, metrics, built-in trap set
  api/         FastAPI server + JSON views          mcp_server.py   MCP tools for AI assistants
scripts/       evaluation, calibration and benchmark runners
web/           single-page web app                  extension/      Chrome extension (MV3)
docs/          CONCEPTS.md (methods explained), benchmark_report.md, original spec
```

## Roadmap

- [x] Retrieval, scientific sources, audit mode, local NLI verifier
- [x] Router, critic, calibrated confidence, judge, orchestrator, feedback loop
- [x] Benchmark vs LLM baselines, ablations, calibration report card
- [x] Fine-tune the verifier on an A100 (SciFact, HealthVer, COVID-Fact)
- [x] Evidence reranker (relevance gate), study-design classifier, diagnosis of where claims are lost
- [ ] Better evidence aggregation; calibration beyond SciFact (general facts)
- [ ] Open-access full text (PubMed Central / Europe PMC)
- [x] FastAPI server, CLI and web UI (audit-first, streaming progress, feedback, history, watched claims)
- [x] MCP server so LLMs can fact-check their own answers; Chrome extension
- [ ] Deployment, monitoring, GRPO-trained claim writer
