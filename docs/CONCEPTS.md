# Concepts behind the AI Truth Engine

The ideas you need to understand how the project works, grouped by area.
**★ = core**: learn these first. Unstarred items deepen your understanding
but aren't needed to follow the code.

Suggested order: 1 → 2 → 4 → 5 → 7 → 8 → 9, then the rest as you meet them in the code.

---

## 1. How LLMs behave
- ★ **Hallucination** — models produce fluent, confident text that is false. The whole project exists to catch this.
- ★ **Tokens, cost and latency** — you pay per token in and out, and bigger models are slower and pricier. This is why the engine scales effort to difficulty.
- ★ **Structured output (JSON)** — getting a model to return machine-readable data so the next stage can use it.
- **Temperature and sampling** — why the same prompt can give different answers, and why independent samples matter.
- **LLM APIs** — how to call Anthropic, OpenAI and OpenAI-compatible servers (messages, system prompts, usage fields).

## 2. Retrieval (finding evidence)
- ★ **RAG (retrieval-augmented generation)** — fetch relevant documents first, then answer from them. It's the baseline this project improves on.
- ★ **BM25 / TF-IDF** — classic keyword search that scores rare matching words highly.
- ★ **Embeddings and cosine similarity** — turning text into vectors so "similar meaning" becomes "nearby vectors".
- ★ **Hybrid search and reciprocal rank fusion (RRF)** — combining keyword and semantic rankings into one.
- **Chunking** — splitting documents into passages. This project uses single sentences so evidence is precise.
- **Query reformulation** — rewriting a claim into several searches, including searches designed to find counterevidence.
- ★ **Bi-encoders vs cross-encoders (two-stage retrieval)** — an embedding model (bi-encoder) encodes claim and sentence separately, so it can search millions of sentences fast; a cross-encoder reads both together, which is slower but judges relevance better. The engine searches with bge-base, then a small cross-encoder (MiniLM) decides which sentences the NLI model reads.

## 3. Knowledge graphs
- ★ **Entities, relations and triples** — facts stored as (subject, relation, object), e.g. (Einstein, born_in, Ulm).
- **Multi-hop reasoning** — answering questions that chain several facts, found by graph search (breadth-first search).
- **GraphRAG** — using graph paths as retrieval results alongside text passages.
- **Entity resolution and aliases** — knowing that "Einstein" and "Albert Einstein" are the same node.

## 4. Claim verification (the core NLP task)
- ★ **Claim decomposition** — breaking text into atomic, individually checkable statements.
- ★ **Natural language inference (NLI)** — classifying whether evidence *entails*, *contradicts* or is *neutral* toward a claim. The verifier is an NLI system.
- ★ **Fact verification** — the research field around datasets like FEVER and **SciFact** (scientific claims labeled SUPPORTS / REFUTES / not enough info).
- **Check-worthiness** — deciding which sentences are factual claims worth checking, as opposed to opinions or filler.
- **Negation, antonyms, number and date matching** — the signals the offline verifier uses to detect contradictions.
- ★ **Transfer learning and fine-tuning** — start from a model already trained on a related task (DeBERTa trained on general NLI) and continue training on in-domain data (SciFact, HealthVer, COVID-Fact). Needs a held-out split for model selection and fp32 master weights (training directly in fp16 overflowed to NaN here).
- **Hard negatives** — training examples that look relevant but aren't (other sentences from the same abstract), which teach finer distinctions than random negatives.

## 5. Evidence-based science and medicine (the niche)
- ★ **Evidence hierarchy** — systematic reviews/meta-analyses > randomized controlled trials > cohort > case-control > case reports > animal > in-vitro > expert opinion.
- ★ **Study designs** — what an RCT, cohort study, case-control study, meta-analysis and systematic review each can and can't show.
- ★ **Correlation vs. causation and confounding** — why observational studies can't prove cause, and how claims get overstated.
- ★ **Sample size and statistical power** — why a 48-person study is weak evidence on its own.
- ★ **Replication** — a single unreplicated finding is provisional; the replication crisis shows how often they fail.
- ★ **Relative vs. absolute risk** — "50% lower risk" can mean going from 2 in 10,000 to 1 in 10,000.
- **p-values, effect sizes and confidence intervals** — reading what a study actually found.
- **Preregistration and p-hacking** — why preregistered trials are more trustworthy.
- **Conflicts of interest** — industry funding as a quality signal.
- **Retractions** — withdrawn papers that still get cited.
- **Generalization** — mouse or cell results don't automatically apply to humans, and one population doesn't imply all populations.

## 6. Citation analysis (the echo detector)
- ★ **Citation graphs** — papers and articles as nodes, citations as edges.
- ★ **Derivation vs. citation** — a news story *restating* a study isn't independent of it; a new trial that *cites* the study but collected its own data is.
- ★ **Source independence** — counting distinct underlying lines of evidence rather than raw source counts.
- **Connected components and union-find** — the graph algorithm used to group sources into independent lineages.
- **Scholarly APIs** — PubMed (E-utilities) and OpenAlex: searching papers, getting references and retraction flags.

## 7. Ensembles, judges and adversarial checking
- ★ **Ensembles** — combining several models' answers.
- ★ **Correlated errors** — models trained on similar data make the same mistakes, so agreement is weaker evidence than it looks.
- ★ **LLM-as-judge bias** — model judges can be confidently wrong, favor longer answers, or favor their own style.
- ★ **Falsification** — Popper's idea that you test a claim by trying to disprove it. That's the critic's job.
- **Red-teaming / devil's advocate** — a component instructed to assume the answer is wrong.

## 8. Uncertainty, calibration and abstention
- ★ **Calibration** — a model is calibrated if its "80% confident" answers are right 80% of the time.
- ★ **Reliability diagram and expected calibration error (ECE)** — the chart and the number behind the calibration report card.
- ★ **Selective prediction / abstention** — answering only when confident enough, and the risk-coverage trade-off.
- **Brier score and log loss** — scoring probabilistic predictions.
- **Platt scaling and isotonic regression** — methods that learn to turn raw scores into calibrated probabilities.

## 9. Routing and cost optimization
- ★ **Model routing and cascades** — send easy inputs to cheap models and escalate only when needed.
- ★ **Cost–accuracy trade-off** — optimizing verified accuracy per dollar, not accuracy at any price.
- **Contextual bandits (explore vs. exploit)** — how the router learns from past traces which choices pay off.
- **Logistic regression and basic classifiers** — the simple learned models used for routing and calibration.

## 10. Evaluation and research method
- ★ **Precision, recall and F1** — measuring claim-level verification quality.
- ★ **Baselines** — the simpler systems (single LLM, standard RAG, majority vote, judge) the engine must beat.
- ★ **Ablation studies** — remove one component at a time to see what actually helps.
- **Train/test separation and benchmark contamination** — why the router and calibrator must be evaluated on data they weren't fit on.
- **Length bias in evaluation** — longer answers can look better without being more correct.
- ★ **Error analysis by pipeline stage** — tracing each failure to the first stage that lost it (retrieval, filtering, stance, aggregation). Here it showed a word-overlap filter, not the model, was the bottleneck, so fine-tuning the model alone couldn't help.
- **Distant supervision** — using labels that come for free (PubMed's publication-type tags) to train a classifier (study design) without hand labelling.
- **Paired bootstrap significance** — resampling the same claims for two systems to see whether an accuracy gap is real or noise.
- **Domain shift in calibration** — a calibrator fit on one dataset (SciFact) can be miscalibrated on another (general facts); calibration has to be checked per domain.

## 11. Software engineering used in the code
- ★ **Python with type hints and Pydantic** — every pipeline object is a validated Pydantic model.
- ★ **REST APIs with FastAPI** — the server the web interface talks to.
- ★ **HTTP clients, retries, exponential backoff and rate limits** — calling external APIs reliably.
- **Server-Sent Events (SSE) / streaming** — sending investigation progress to the browser live.
- **Concurrency (thread pools)** — running several model calls in parallel.
- **SQLite** — storing investigation traces and watched claims.
- **Environment variables for secrets** — how API keys are supplied without touching code.
- **pytest** — the test suite.
- **HTML / CSS / JavaScript** — the web interface.
- **Model Context Protocol (MCP)** — a standard way for AI assistants to call external tools; the engine exposes audit/ask as MCP tools so Claude can check its own answers.
- **Browser extensions (Chrome Manifest V3)** — background service worker, context menus, `activeTab` injection, shadow DOM isolation, and rendering untrusted text with DOM APIs (never `innerHTML`) to prevent XSS.
- **Linting and coverage (ruff, pytest-cov)** — static checks for bugs like unused variables, and measuring which code the tests exercise.
