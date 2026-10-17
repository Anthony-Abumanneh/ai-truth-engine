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
5,183 abstracts. All tuning (model choice, thresholds, calibrator) used the **train** split only.

| Component | Before | After |
|---|---|---|
| Retrieval: right abstract in top 5 (BM25 → hybrid with bge-base embeddings) | 85% | **94%** |
| Evidence judgment, macro-F1 (rules → rules + DeBERTa-v3-large NLI) | 0.44 | **0.77** |
| Calibration error, ECE (raw → trained calibrator) | 0.25 | **0.056** |
| Evidence judgment after fine-tuning DeBERTa (large / base) on SciFact + HealthVer + COVID-Fact, A100 | 0.74 / 0.68 | **0.81 / 0.82** |

Fine-tuning ([`training/finetune_verifier.ipynb`](training/finetune_verifier.ipynb)) improved judging
of expert-chosen evidence, but not the end-to-end pipeline (0.58 vs 0.61, not significant): the
remaining errors come from *finding and combining* evidence, which is the next step.

**End to end vs. LLM baselines** (same 100 dev claims; full report in
[`docs/benchmark_report.md`](docs/benchmark_report.md)):

| System | Accuracy | Calibration error (ECE) | Cost |
|---|---|---|---|
| **Truth Engine (local DeBERTa + bge-base models)** | 0.61 | **0.056** | $0 |
| Truth Engine evidence + gpt-oss-120b as judge | 0.74 | **0.054** | free tier |
| gpt-oss-120b + standard RAG | **0.86** | 0.088 | free tier |
| gpt-oss-120b + RAG + few-shot examples | 0.80 | 0.083 | free tier |
| 3 LLMs + RAG + LLM judge | 0.80 | 0.095 | free tier |
| gpt-oss-120b alone (no retrieval) | 0.52 | 0.325 | free tier |
| 3-LLM majority vote (no retrieval) | 0.44 | 0.442 | free tier |

**What this shows:** a large LLM given good retrieval still reasons better on raw accuracy.
The Truth Engine is the **best-calibrated system** tested, runs fully locally for $0, and beats
LLMs without retrieval. Ablations show the NLI model is the main accuracy driver (−12 points
without it). SciFact has no echo chambers or low-quality sources, so the echo detector and
source grading are tested separately on a built-in trap set (93% accuracy, red flags 96%
precision / 100% recall; in-sample). Closing the accuracy gap is the goal of the next step
(fine-tuning the verifier).

## Hardware & Performance

- **Hardware:** MacBook (Apple Silicon, 8 GB RAM), CPU only. Free-tier LLM APIs (Gemini, Groq)
  for baselines only.
- **Embedding the SciFact corpus (44,705 sentences, bge-base):** ~40 min once, then cached.
- **Full SciFact dev benchmark (300 claims):** ~7 min (~1.4 s per claim).
- **Live audit (PubMed + OpenAlex + web):** ~25–50 s for a typical 2–4 claim text, claims checked in parallel.

## Setup Instructions

```bash
git clone https://github.com/Anthony-Abumanneh/ai-truth-engine.git && cd ai-truth-engine
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m truth_engine.pipeline.nli download        # DeBERTa-v3-large NLI model (870 MB)
mkdir -p data/scifact && curl -L https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz | tar -xz -C data/scifact
cp .env.example .env                                          # optional keys: Tavily, NCBI, Gemini, Groq
```

All API keys are optional: without them it runs offline on the local corpus.

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
scripts/       evaluation, calibration and benchmark runners
docs/          CONCEPTS.md (methods explained), benchmark_report.md, original spec
```

## Roadmap

- [x] Retrieval, scientific sources, audit mode, local NLI verifier
- [x] Router, critic, calibrated confidence, judge, orchestrator, feedback loop
- [x] Benchmark vs LLM baselines, ablations, calibration report card
- [x] Fine-tune the verifier on an A100 (SciFact, HealthVer, COVID-Fact)
- [ ] Evidence reranker and better evidence aggregation (the measured bottleneck)
- [ ] Open-access full text (PubMed Central / Europe PMC)
- [ ] FastAPI server and web UI; MCP server so LLMs can fact-check their own answers; browser extension
- [ ] Deployment, monitoring, GRPO-trained claim writer
