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
| 3 | Scientific evidence sources + study grading | ✅ Done |
| 3.5 | Journal signals, Retraction Watch, subject relevance gate | ✅ Done |
| 4 | Audit mode: claim extraction, verifier, science red flags | ✅ Done |
| 4.5 | Local NLI model fused with rules | ✅ Done |
| 5 | Analyzer, router (contextual bandit), compute budgets, independent solvers | ✅ Done |
| 6 | Critic, judge, calibrated confidence, abstention | ✅ Done |
| 7 | Orchestrator, evidence tree, feedback loop, watched claims | ✅ Done |
| 8 | Benchmark (SciFact), baselines A–D, ablations, calibration report card | ✅ Done |
| 8.5 | Fine-tuning: verifier (A100), evidence reranker, study-design classifier | ✅ Done |
| 8.6 | Open-access full text (PubMed Central / Europe PMC) with reranked passage selection | Planned |
| 9 | FastAPI server, CLI, audit-first web UI | ✅ Done |
| 9.5 | MCP server (LLM tool) + browser extension, on the same API | ✅ Done |
| 10 | Tests, end-to-end runs, final docs | ✅ Done |
| 11 | Deploy + MLOps: Docker, HF Spaces, CI eval gate, model versioning, monitoring | ✅ Done (Space not yet deployed: needs user's HF token) |
| 12 | Stretch: GRPO fine-tuning of a small LLM with the verifier as reward | Planned |

### Plan additions (agreed 2026-10-06)
Guiding rule from the user: **accurate, not complex.** Every addition must
earn its place with a measured gain on held-out data; drop it otherwise.
Compute: the user has a Colab A100 with 100 compute units (roughly 7–9
A100-hours; check the live rate). Use it for training only; the engine
itself must keep running on CPU. Large datasets are welcome.

- **Step 5 router as a contextual bandit** (RL-lite, CPU): per claim choose a
  strategy (rules only / +NLI / +LLM / deep literature search) from claim
  features; reward = correctness − λ·cost, learned from stored outcomes
  (benchmark labels first, user feedback later). Rule-based policy until
  enough data; log propensities so policies can be evaluated offline.
- **Step 8.5 fine-tuning** (Colab A100; notebooks in `training/`, models
  pushed to HF Hub, loaded by the same `nli.py` backends):
  1. *Verifier* (biggest expected gain): DeBERTa-v3-**large** (and base for
     CPU latency comparison), starting from MNLI/FEVER/ANLI weights, trained
     on SciFact train + HealthVer + COVID-Fact + PubHealth + Climate-FEVER
     (+ SciFact-Open if licensing allows). Map labels to support/
     contradict/neutral; hard negatives = non-rationale sentences from the
     cited abstract. Evaluate on SciFact dev only (test labels are hidden);
     never tune on dev. Report stance F1 and end-to-end effect (ablation).
     Re-tune fusion thresholds on train afterwards; HARD_SIGNALS may shrink.
  2. *Evidence reranker*: cross-encoder scoring (claim, sentence) relevance,
     trained on SciFact rationale sentences vs. other sentences of the cited
     abstracts (+ MS MARCO / BioASQ if useful). Replaces part of the lexical
     relevance filter.
  3. *Study-design classifier*: distant supervision from PubMed publication
     types (RCT, meta-analysis, systematic review, cohort/observational,
     case report, animal via MeSH, in vitro) over a large abstract sample
     (100k+ via E-utilities / PubMed baseline files). Replaces/augments the
     regex rules in `study_design.py`; keep regex as fallback.
  4. *Check-worthiness*: small classifier on ClaimBuster / CheckThat!; only
     keep if it beats the rule score.
  Rules: train-only tuning, record data/versions/seeds per model, compare
  against the current model on the same split, keep CPU inference viable.
- **Step 8.6 open-access full text** (after the 8.5 reranker): fetch full
  text for open-access papers via PubMed Central (E-utilities `efetch
  db=pmc`, JATS XML) or the Europe PMC REST API (by PMID/DOI); Unpaywall as a
  DOI fallback. Parse sections, tables, funding and COI statements (feed
  study grading: funding/COI, methods, sample size, subgroups). A paper is
  ~200–400 sentences, so the reranker picks the top passages before NLI.
  Cache everything (data/cache). Measure: SciFact-Open / live claims with vs
  without full text. Paywalled papers stay abstract-only.
- **Step 9.5 integrations** (user's choice: best use = a verifier *tool* for
  LLMs): an MCP server exposing `audit(text)` / `ask(question)` / `watch` so
  Claude and other agents can check their own drafts; a browser extension
  (highlight text → claims coloured by verdict, evidence on hover) calling the
  hosted API (step 11). Both are thin layers over the step-9 API.
- **Step 11 deploy + MLOps**: Docker image (CPU, ~1.5 GB with torch),
  Hugging Face Spaces deployment, GitHub Actions running tests + the SciFact
  stance eval as a quality gate (fail on macro-F1 regression), model
  versions recorded in every trace, monitoring page from stored traces
  (latency, cost, abstention rate, rule-vs-model agreement as a drift
  signal), rate limiting / LLM spend caps, prominent medical disclaimer.
- **Step 12 (stretch) GRPO**: small open LLM (0.5–1.5B) trained to produce
  answers the verifier scores as supported. Only after 8.5 (a weak verifier
  invites reward hacking, e.g. hedging everything); include held-out human
  spot checks and a reward-hacking analysis.

### Step 3 — scientific evidence sources + study grading (done; original brief kept below)
What landed / decisions made:
- `science/cache.py` `JsonCache` (data/cache/<ns>/<sha1>.json, secrets stripped
  from keys) + `CachedFetcher` (rate limit, retries, never raises; offline =
  cache-only). `papers.py` `PaperRecord` → `to_source` (canonical id `pmid:`,
  else `openalex:W…`, else `doi:`; title is passage s0; COI statement graded
  but never used as evidence).
- `search.py` `ScienceSearch`: keyword query drops "finding words"
  (improves/prevents/proven…) so it finds studies whatever they concluded;
  PubMed back-off 6→4→3→2 terms; falsification = same terms restricted to
  RCT/MA/SR (PubMed `[pt]` filter, OpenAlex `type:review`). Merges the same
  paper across APIs; rewrites OpenAlex reference ids to canonical ids.
- **Echo rule for real literature:** a systematic review / meta-analysis is
  `derived_from` the retrieved primary studies it cites. `CitationGraph` now
  joins a multi-root source to only one root, so a review pooling trials A
  and B doesn't merge A and B into one lineage.
- Study design: front matter > PubMed pub types > negation-aware text cues
  (title before abstract). Reviews never take prereg/funding from abstract
  text (it describes the studies they review). Design weight is blended into
  `methodology`; paper authority = 0.75 + 0.25·(weight − 0.4); retracted →
  overall ≤ 0.15, tier D; n<50 −0.25, n<100 −0.15.
- Corpus +13 fictional docs (lumorex retraction, calvetrin mice, brivo cohort,
  zantorel RRR/ARR, nervaplex in-vitro echo). New front-matter keys:
  `retracted`, `species`, `design`.
- Fixtures in `tests/fixtures/` are real recorded responses (creatine/memory
  search; Wakefield 1998, PMID 9500320, as the retracted example).
  Live test auto-skips when PubMed is unreachable.

Step 3.5 (added after review of step-3 limitations):
- `science/venues.py`: OpenAlex `/sources` (batched by id or ISSN). Standing =
  log-scaled 2-yr mean citedness → ±0.05 on overall. Vetting = on any curated
  list (`listed_in`: medline, doaj, cwts-core, national lists) or `is_core`;
  unvetted → −0.1 + "possible predatory venue" note. No hand-made blacklist.
- `science/retractions.py`: Crossref `updated-by` notices (Retraction Watch
  data), 20 DOIs per request. retraction/withdrawal/removal → retracted;
  expression of concern → −0.15. Uses OPENALEX_EMAIL as contact.
- `ScienceSearch(enrich=True)` runs both batches inside `_ingest`.
- Subject gate in `EvidenceEngine.gather`: outside sources (papers, web) must
  mention the claim's subject (first entity, else first keyword) in passage
  or title. Corpus is exempt (its docs are curated).

Original brief:
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

### Step 4 — audit mode (done)
- `pipeline/claims.py`: sentence spans → clause split (; / but / whereas) →
  predicate split (shared subject; noun lists like "memory and focus" stay
  together) → causal split ("A because B" → A, and A-because-B) → pronoun
  subject resolution. Tags: `science_kind`, `population` (human unless
  mice/cells named), `hedged`, `check_worthiness` (threshold 0.45). Optional
  LLM extraction via `llm=(provider, spec)`.
- `pipeline/verifier.py` `stance()`: coverage gate (sentence ≥0.3, with title
  ≥0.4, subject mentioned in title/sentence/neighbour sentence unless
  coverage ≥0.6) → study caveats neutral → "for X" slot mismatch → polarity
  (null-result/dispute phrases, or a bare negation *in scope*: next 3 content
  words minus subject; "no X" scopes over 2) → antonyms → value mismatch
  (km→m; years/dates exact) → different name → weak-evidence phrases → entail
  (cov ≥0.65) / partial (≥0.45). Certainty words are ignored for coverage.
- Verdict: weight = quality × applicability (animal 0.3, cells 0.2,
  observational for causal 0.5) × stance strength. Derived sources capped at
  their root's strength; retracted root → 0. A source with an explicit
  (non-numeric, non-"disputed") contradiction loses its support items; a
  value mismatch is dropped if the same source states the claimed value.
  Lineages (citation graph) → max weight each → noisy-OR → S, C. Status:
  conflicting (both ≥0.4 within 0.2, or certainty claim with C ≥0.35) →
  UNVERIFIED; C ≥0.5 & C ≥ S+0.2 (or C ≥0.35 & S <0.2) → CONTRADICTED;
  S ≥0.5 & S ≥ C+0.2 → SUPPORTED / INFERRED / PARTIAL; S ≥0.25 → PARTIAL
  (SPECULATIVE if hedged). `confidence` is provisional until step 6.
- Graph paths support at most PARTIAL (stitched facts ≠ proof).
- "X is clinically proven" (no effect of its own) is judged via the effect
  claim from the same sentence: SUPPORTED only with tier-A RCT/MA support and
  no contradiction; CONTRADICTED if the effect is; else UNVERIFIED.
- LLM verifier (optional `llm=(provider, spec)`): its stance wins for the top
  8 items; rule/LLM agreement stored as `verifier_agreement`.
- `pipeline/audit.py`: `audit(text, sources=...)` → `AuditReport`; CLI
  `python -m truth_engine.pipeline.audit [--live] "text"`.
- Tuned against 18 hand-written true/false demo claims (17/18; "Collins
  walked on the Moon" needs world knowledge → LLM). That set is NOT a
  benchmark; step 8 measures on SciFact.

### Step 4.5 — local NLI verifier (done)
- Model: `Xenova/DeBERTa-v3-base-mnli-fever-anli`, 8-bit ONNX (244 MB) in
  `data/models/` (gitignored), via onnxruntime + tokenizers (no PyTorch:
  the disk had ~2 GB free). `python -m truth_engine.pipeline.nli download`.
  `TE_NLI=0` disables; tests default to TE_NLI=0 (conftest) and
  `tests/test_nli.py` covers the model; full suite also passes with TE_NLI=1.
- **No padding**: this ONNX export gives different outputs for padded
  inputs (0.35 vs 0.76 entail on the same pair). One pair per run, ~15 ms.
- `rule_stance()` now returns which signal decided. `fuse()`: HARD_SIGNALS
  (value, value_match, slot, weak, limitation) keep the rule; otherwise
  NLI decides at ≥0.7; P(neutral) ≥0.9 dismisses; else rules ×0.7.
  Rule-"offtopic" sentences: NLI may rescue a paraphrase (entail ≥0.7, ×0.8)
  but a contradiction needs ≥0.9 and counts ×0.6 (NLI calls different
  events contradictions: "launched X1" vs "discontinued X1").
- Premise = sentence, prefixed by title when it never names the subject.
- `verifier_agreement` = rules vs LLM (if run) else vs NLI.
- **PyTorch backend added and now default** (`TE_NLI_BACKEND=auto` → torch,
  else onnx): `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` original
  weights (369 MB), ~80 ms/pair. The 8-bit ONNX export lost real accuracy
  (got Collins and Einstein-relativity wrong; full model gets both).
- Thresholds re-tuned for torch on train: NLI_DECIDE 0.5, NLI_DISMISS 0.95,
  OFFTOPIC_CONTRADICT 0.7 (×0.6). Added hard signal `causal_denial`
  ("not by / rather than / not because" + the claimed cause). Tried a guard
  against NLI arithmetic errors on number-heavy sentences: cost ~1 point on
  SciFact train, removed (so "Zantorel cuts risk in half" vs "0.2% vs 0.4%"
  is read as a contradiction → UNVERIFIED).
- Known: loading torch AND onnxruntime in one process crashes at exit
  (libc++ recursive_mutex) after tests finish; only tests/test_nli.py
  `any_nli` does this. Results are unaffected.
- **Later in 4.5 (after disk was freed): ONNX dropped entirely** (crashed at
  exit alongside torch; less accurate). **Default model is now
  DeBERTa-v3-large** (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`,
  870 MB; `TE_NLI_MODEL=base` for the 369 MB one). Batched inference ~40–55 ms
  per pair on CPU or Apple GPU (`TE_DEVICE`; auto = cuda if present, else CPU; mps opt-in, it starved the 8 GB Mac).
  Re-tuned on train: NLI_DECIDE 0.4, NLI_DISMISS 0.99, OFFTOPIC_CONTRADICT
  0.5; all six hard signals kept. Dev: fused 0.766 acc / 0.771 macro-F1.
- Settings chosen on SciFact **train**; dev numbers in README. SciFact data
  is in `data/scifact/` (gitignored, 3.1 MB).
- My 18 hand-made demo claims: rules 17/18 (overfit), fused 15/18. SciFact
  says the opposite about which is better; trust SciFact.

### Step 5 — analyzer + router (done)
- `pipeline/analyzer.py` `analyze(text)` → `QuestionAnalysis`: domain by
  lexicon hits, type (efficacy claims → causal), difficulty from named
  `signals`, risk (high = medical high-risk terms or safety), budget from a
  difficulty × risk grid, tools.
- `pipeline/router.py`: `STRATEGIES` per tier (sources, NLI, LLM, k, follow-up
  studies). `Router.strategy()` strips what's unavailable (no `live` → no
  PubMed/OpenAlex/web; no key → no web/LLM). Escalation climbs to the next
  tier that *does something different* while the verdict has no evidence,
  conflicts, is unverified, or confidence < 0.35; cap = high (maximum for
  high-risk). `BanditPolicy` = disjoint LinUCB + ε-greedy with logged
  propensities, `reward = correct − 0.05·est_cost`, `ips_value()` for
  offline evaluation, JSON save/load at data/router_policy.json. It takes
  over from rules only after `min_samples` (200) updates; training data
  comes from step 8 benchmark runs. Tested on a synthetic world only.
- `EvidenceEngine.gather(follow_up_studies=True)` (high+, live): reads
  papers citing the found papers (OpenAlex cited_by) as falsification hits.
- `audit(text, live=, max_tier=, budget=, sources=)`: routed per claim;
  `sources=` forces one fixed pass (routing_source "forced").
  `AuditReport.routing[claim_id]` = analysis, tiers, passes, escalation
  reasons. CLI shows the route; `--max-tier=`.
- `pipeline/solvers.py` (Ask mode, step 6/7 will judge + assemble):
  yes/no question → claim (verify it); else graph multi-hop + swift (BM25) +
  scholar (vector) + sage (quality-weighted hybrid) + any LLMs; short
  answers type-aware (who → person, no demonyms). `diversity()` clusters
  short answers; flags same-corpus agreement as correlated. Deferred (pivot):
  math/code/SQL solvers from the general spec.
- Live: "Vitamin D prevents depression" now escalates medium → high and ends
  UNVERIFIED (conflicting) instead of SUPPORTED.

### Step 6 — critic, confidence, judge (done)
- Retrieval upgrade first (measured on 300 SciFact *train* claims,
  `scripts/eval_retrieval_scifact.py`, recall@5 of a gold-evidence abstract):
  hashing hybrid 0.797, BM25 0.847, MiniLM hybrid 0.907, bge-small vector
  0.927, **bge-base vector 0.943 → default embedder** (`TE_EMBEDDINGS=st`,
  `TE_EMBEDDINGS_MODEL`); BGE query prefix; embeddings cached on disk
  (`data/cache/embeddings/`, keyed by model + texts). BM25 now uses an
  inverted index; `DocumentStore.neighbors` is O(1).
- Bugs found via SciFact traces: `population` defaulted to human, so
  molecular claims got the animal penalty; now human only for people/body
  words, countries, or efficacy/safety/association claims, else
  "unspecified".
- Verifier A/B end to end (200 train claims, `scripts/experiment_verifier.py`):
  premise = sentence + neighbours (`PREMISE_CONTEXT=True`), NLI still gated
  by the rules (`NLI_SCOPE="gated"`).
- `bench/scifact.py`: SciFact corpus as a DocumentStore (44,705 sentences),
  `run(split, components=..)` audits each claim as one claim and records
  status, signals, correctness, evidence hit, critic change, tiers.
  Records cached in `data/bench/*.jsonl` (gitignored).
- `pipeline/critic.py`: targeted re-search against the leaning, re-verify;
  stronger design on the losing side → UNVERIFIED (conflicting); claimed
  number not backed by any "same value" support → PARTIALLY_SUPPORTED;
  red flags → findings; causal alternatives.
- `pipeline/confidence.py`: 9 spec signals → raw (hand-weighted logistic) →
  `Calibrator` (logistic regression on signals + status one-hot, or Platt on
  raw; chosen by 5-fold CV Brier) with per-status historical reliability;
  ECE, Brier, reliability table; labels VERY LOW…VERY HIGH.
- `pipeline/judge.py`: `decide()` abstains below the fitted threshold (shown
  as UNVERIFIED with `leaning`), always abstains on conflict, caveats on
  severe red flags / LOW–MODERATE labels. `judge_candidates()` for Ask mode
  ranks by verification, not majority.
- `audit(components=...)`: critic / calibration / abstention switchable
  for ablations. `scripts/calibrate.py` fits on train, picks the threshold
  maximizing train accuracy, saves `data/calibrator.json`, reports dev.
- **Results (SciFact, open retrieval over the 5,183-abstract corpus; claims
  audited as single claims; `data/bench/calibration_report.json`):**
  train 809: accuracy 0.663, macro-F1 0.654. **dev 300: accuracy 0.613,
  macro-F1 0.587** (step-8 baselines will put this in context).
  Calibration: logistic beat Platt (CV Brier 0.205 vs 0.211). Dev ECE 0.251
  → **0.054**, Brier 0.288 → 0.216.
  Abstention: maximizing accuracy chose threshold 0 (main error is
  under-calling: of 809 train claims, 197 supported/contradicted claims came
  back NEI; decisive verdicts were 78% right). Threshold is therefore chosen
  by precision of *stated* verdicts ≥ 0.8 on train → 0.6; dev stated-verdict
  precision 0.73 at 38% coverage (dev is harder than train).
  Critic ablation: dev accuracy 0.610 with vs 0.613 without, ECE 0.060 vs
  0.054, 3× slower → **critic is opt-in** (`components={"critic", ...}`).
  Red flags feed `unresolved_uncertainty` when the critic is off.
- Tests: `TE_CALIBRATOR=0` (conftest) so a refit can't change test results;
  `data/calibrator.json` is committed as the default fitted artifact.
- Next accuracy levers (measured, not guessed): under-calling (recall) and
  gold evidence used for only ~60% of claims despite 94% retrieval recall@5,
  i.e. stance/aggregation, not search. That's what step 8.5's fine-tuned
  verifier + reranker should target.

### Step 7 — orchestrator, traces, feedback, watched claims (done)
- `pipeline/orchestrator.py` `Orchestrator(live, max_tier, components,
  store, workers, …)`: `audit(text, on_event)`, `ask(question, on_event)`,
  `watch()`, `recheck_watched()`. Every run → `Trace` (mode audit/ask,
  claims, verdicts, sources, routing, events, tree, final) saved to SQLite.
- `audit()` = extraction + `audit_claims()` (also used by Ask);
  `workers>1` runs claims in a thread pool (default `settings.max_parallel`
  when live, 1 offline); `emit(StageEvent)` callback for the UI log.
  Thread safety: sources synced from each search's own results (not
  engine state); `ScienceSearch` ingest under an RLock.
  Live 4 claims: ~25 s with 6 workers vs ~32 s sequential; bound by
  PubMed's 3 req/s without `NCBI_API_KEY` and the CPU NLI lock.
- Ask: yes/no → question_to_claim → verify → "Yes / Yes, partly / No /
  The evidence isn't good enough to say (leans …)". Open questions →
  solvers → `answer_type_ok` (graph entity type vs who/which-country…) →
  `to_statement(question, answer)` → verify each distinct answer →
  `judge_candidates` (verified, not majority). Multi-hop "country of the
  person who developed GR" now abstains instead of answering "Einstein".
- `pipeline/tree.py`: root → claims → supports/contradicts per lineage
  (derived copies nested) → red flags (relation "rejected").
- `feedback/traces.py` `TraceStore` (data/truth_engine.sqlite,
  `TRUTH_ENGINE_DB`): traces (§14 record + full JSON), feedback (label must
  be an EvidenceStatus value), watched (history; `record_check` reports
  status changes and confidence moves ≥ 0.15).
- `feedback/learning.py`: feedback → outcome rows → `update_router`
  (reward = correct − 0.05·cost of tiers used, arm = start tier) and
  `calibration_rows`; `learn_from_feedback` consumes feedback once.
- Router: decisive verdicts built from one side only (or one lineage)
  escalate when a higher tier searches more *sources* (live literature,
  follow-up studies). Live melatonin: CONTRADICTED 0.76 from one
  meta-analysis → escalated → UNVERIFIED (conflicting meta-analyses).
  Offline/SciFact behaviour unchanged.
- Known: calibrator is fitted on SciFact (closed corpus, one evidence
  abstract per claim) and over-trusts single sources live; live stance
  errors remain (e.g. "low-dose melatonin is a useful strategy" read as
  contradicting; a Mediterranean-diet trial counted as fish-oil evidence).

### Step 8 — benchmark (done)
- `bench/metrics.py` (accuracy, macro-F1, per-label P/R/F1, coverage, acc. on
  answered, ECE/Brier/reliability on `conf_pred`, citation P/R, latency,
  tokens, cost; `bootstrap_diff` paired CI; `red_flag_scores`),
  `bench/systems.py` (run_ours; baselines A single / B rag / C vote / D judge;
  B few-shot from train examples whose abstracts aren't dev evidence;
  ours_llm_judge = our evidence + red flags + verdict shown to an LLM),
  `bench/builtin.py` (28 in-sample trap claims), `scripts/benchmark.py`
  (records cached in data/bench/sys_*; refuses to save runs with >10% failed
  calls; `report` → docs/benchmark_report.md).
- Components switchable: nli, falsification, independence, source_quality,
  adaptive, calibration, abstention, critic (`audit.DEFAULT_COMPONENTS`).
- LLMs: free tiers via OpenAI-compatible endpoints (providers gemini / groq /
  openrouter; models in .env `TE_GEMINI_MODELS`, `TE_GROQ_MODELS`). Used:
  gemini-flash-lite-latest (flash hit a 20/day cap), groq openai/gpt-oss-120b,
  qwen/qwen3.8-27b. All responses cached (data/cache/llm). Groq gpt-oss has a
  daily quota that a ~400-call run exhausts; LLM wrapper marks a model
  exhausted on daily-quota 429s.
- **Results (SciFact dev; LLM systems on the first 100 claims):** ours 0.603
  acc / ECE 0.056 (n=300; 0.61 on the 100). gpt-oss + RAG 0.86 / 0.088;
  Gemini + RAG 0.84; Qwen + RAG 0.80; few-shot RAG 0.80 (examples *hurt*);
  D judge 0.80 / 0.095; ours evidence + gpt-oss judge 0.74 / **0.054**;
  A gpt-oss alone 0.52; C vote 0.44. Ours significantly beats no-retrieval
  LLMs; RAG + LLM significantly beats ours on accuracy.
- Ablations (300): only NLI matters on SciFact (−0.12). falsification,
  independence, source quality, adaptive, critic: no measurable change
  (SciFact has no echo chambers/low-quality sources; evidence and counter-
  evidence share one abstract). Calibration: same accuracy, ECE 0.25 → 0.056.
- Built-in trap set: 0.93 acc, flags P 0.96 / R 1.00 (in-sample). Fixes it
  drove: relative-risk flag needs a magnitude; judge abstains when all
  support is observational / animal / in-vitro (generalization flags ≥ 0.7).
- Bug found: `check_claim` without a calibrator ran uncalibrated; it now
  loads the default calibrator itself (first ablation pass was redone).
- Latency work: gather() runs web + literature (support + falsification)
  in parallel; PubMed ∥ OpenAlex; venue ∥ Crossref enrichment; incremental
  escalation (skip already-searched sources, keep earlier results); Tavily
  basic depth, 2 web queries/claim; NLI ≤ 24 pairs/pass, 256 tokens,
  length-sorted batches; query-embedding cache; live literature claims start
  at medium. `scripts/profile_live.py` times each stage.
- Interpretation for the README/resume: LLM + RAG reasons better; ours is
  best-calibrated, local and $0. Verifier accuracy is the gap → step 8.5.
- Repo public: github.com/Anthony-Abumanneh/ai-truth-engine (README
  rewritten results-first, 2026-10-14).

### Step 8.5 — fine-tuning (in progress, 2026-10-15)
- Built: `bench/training_data.py` (SciFact train rationale pairs + hard
  negatives + NEI, ×2; HealthVer train; COVID-Fact; val = 10% SciFact train
  claims + HealthVer dev; SciFact dev claims removed from all data, 4 overlaps
  found; 20,094 train / 2,197 val pairs), `scripts/finetune_verifier.py`
  (HF Trainer, --size large|base, best epoch by val macro-F1, saves to
  data/models/local__truth-engine-verifier-{size} = MODELS ft-large/ft-base,
  writes training_card.json; --smoke = tiny CPU check), `scripts/eval_nli.py`
  (raw NLI on SciFact dev pairs + HealthVer test, ms/pair),
  `training/finetune_verifier.ipynb` (Colab A100: clone public repo → data →
  before eval → train large + base → after eval → zip to Drive),
  `requirements-train.txt` (+accelerate).
- Bugs caught by the CPU smoke test: transformers 5.18 has no `warmup_ratio`
  (use warmup_steps); checkpoints are stored fp16 and load as fp16 → first
  AdamW step NaN → now loaded with dtype=float32. Fine-tuned models are
  saved in fp16 (base 360 MB, large ~870 MB). Smoke test PASSED 2026-10-17
  (`--smoke`: 24/16 pairs, batch 4, 2 threads, last layer + head only; peak
  RAM 1.45 GB; full-model CPU training froze the 8 GB Mac — never do that).
  Run anything heavy with `nice -n 19` and OMP_NUM_THREADS=2 locally.
- Disk: ~8 GB free on 2026-10-17; fp16 models need ~1.2 GB zip + ~1.2 GB
  unzipped (delete the zip after).
- Latency finding: with network cached, NLI on CPU is the bottleneck (37 of
  59 s for 2 claims). A fine-tuned *base* verifier (~3× faster) matching large
  would fix it — compare in eval_nli.py.
- **Colab results (2026-10-17, A100 40GB; large 8.5 min, base 4 min; 1 epoch
  is enough: val F1 flat from epoch 1→2 while val loss rose):** raw NLI
  macro-F1 SciFact dev large 0.741 → **0.807**, base 0.677 → **0.816**;
  HealthVer test 0.455 → 0.766 / 0.441 → 0.778 (in-distribution: trained on
  HealthVer train). Fused verifier (ft-large) 0.771 → 0.823. Reproduced
  exactly on the Mac CPU. Executed notebook committed in training/.
- Fusion thresholds re-checked on the 80 held-out SciFact-train claims (84
  pairs): best re-tune +0.004 (1 pair) → kept unchanged.
- **End to end (SciFact dev 300, raw verdicts): ft-base 0.580 vs large 0.613**
  (29 right→wrong, 19 wrong→right; not significant). More NEI-gold claims
  wrongly called (23 vs 15): trained on same-abstract negatives, but the
  pipeline also reads other on-topic abstracts. Dominant error unchanged:
  ~80 SUPPORT/CONTRADICT claims → NEI; gold abstract among cited evidence for
  only ~37%. **Next lever = evidence selection/aggregation, not stance.**
- Default NLI model stays "large" + data/calibrator.json (= calibrator_large_v1);
  ft-base calibrator (Platt, fit on 80 held-out claims, dev ECE 0.27 → 0.07)
  saved as data/calibrator_ft-base_heldout.json. calibrate.py: `--heldout`,
  `--no-critic`, threshold falls back to best accuracy if the precision
  target is unreachable (80 claims → threshold 0.0).
- Old plan line (kept for reference): After Colab: unzip into the project, run eval_nli.py, re-tune fusion
  (eval_stance_scifact train sweep), refit calibrator (scripts/calibrate.py),
  rerun benchmark.py ours ablations, update README results.

### Step 8.5 part 2 — reranker gate, design classifier (done, 2026-10-26)
- `scripts/diagnose_scifact.py` (dev, 188 SUPPORT/CONTRADICT claims): correct
  46%; **rationale retrieved but never read by NLI 27%** (word-overlap
  "offtopic" gate); aggregation 7%; stance wrong 5%; neutral 5%; abstract
  found but not rationale 5%; abstract not found 5%. Retrieval is NOT the
  bottleneck. (Diagnosis looked at dev; the fix's threshold was tuned on train.)
- `pipeline/reranker.py`: cross-encoder/ms-marco-MiniLM-L6-v2 (91 MB,
  off the shelf, no training needed). **Bug: mmap'd safetensors weights gave
  NaN in every matmul on this Mac's BLAS; params are cloned after loading**
  (tests/test_reranker.py guards it). Verifier `RERANK_GATE = 0.5`: sentences
  scored ≥ 0.5 skip the "offtopic" rule and are read/judged normally; NLI
  pairs ordered by reranker score. Train sweep (300 claims): off 0.683, gates
  0.05–0.8 all 0.700–0.703 → 0.5.
- `science/design_classifier.py`: TF-IDF + logistic regression, distant
  labels from PubMed publication types/MeSH (8 designs, 7,300 abstracts,
  data/finetune/study_design.jsonl; model data/models/study_design_clf.joblib,
  rebuild with `python -m truth_engine.science.design_classifier`). Held-out
  PMIDs: classifier 0.92 acc; rules+classifier 0.80 vs rules-only 0.45
  (unknown 51% → 16%; animal recall 0.13 → 0.76). Used only for papers with no
  PubMed tags and when rules say unknown (`scripts/eval_study_design.py`).
- Fixes found on the way: risk wording ("increased risk", "increases the risk
  of", "predicts", "described by") = association, not causal (was blocking
  correct SciFact supports via correlation_as_causation); red flags no longer
  count as `unresolved_uncertainty` for CONTRADICTED verdicts (they weaken
  the support, i.e. strengthen the contradiction).
- **Results, dev 300:** cited expert evidence 58% → **70%**; pre-judge
  accuracy 0.613 → 0.620; final (calibrated, abstention 0.65) 0.603 → 0.603;
  stated-verdict accuracy 0.73 → 0.77 at 37% coverage; ECE 0.056 → **0.043**.
  Train: ECE 0.037, stated 0.816.
- **Known regression: trap set 0.93 → 0.68.** Verdicts still correct
  (conf_pred right) but general-knowledge contradictions get 0.39–0.53
  confidence < 0.65 bar: calibrator fit on SciFact (contradicted reliability
  0.63) is domain-specific. Proper fix = calibration data beyond SciFact
  (FEVER / general facts), not threshold hacking.
- Check-worthiness classifier: dropped (only political-debate data exists;
  no science set to show it beats the rules).
- Old calibrator kept as data/calibrator_large_v1.json.

### Step 9 — API, CLI, web UI (done, 2026-10-26)
- `api/server.py` `create_app(engine, store, warm)`: FastAPI; models warm up in
  a background thread (`/api/health` ready flag). Endpoints: health, audit,
  audit/stream (SSE via POST; worker thread → queue → `event: progress|result|
  error`), ask, traces (list/get), feedback (400 on bad label), watch
  (GET/POST/DELETE, recheck). Input limits: audit text ≤ 6000 chars.
- `api/views.py` `trace_view`: compact JSON (claims with sentence spans for
  highlighting, ≤5 evidence per side deduped by source, flags, route, answer).
- `web/index.html`: single file, no build, vanilla JS; tabs Audit / Ask /
  History / Watched; highlighted spans colour = worst status in the sentence;
  all server text HTML-escaped, only http(s) links. Light/dark via
  prefers-color-scheme; checked at 375 px (no horizontal scroll), no console
  errors.
- `truth_engine/__main__.py`: `python -m truth_engine serve|audit|ask|watch|recheck`.
- Preview: entry "truth-engine" in ~/DSA_DASHBOARD_PROJECT/.claude/launch.json
  (port 8600, OMP_NUM_THREADS=2, nice 10). Server holds ~2.5 GB RAM — stop it
  when done on the 8 GB Mac.
- tests/test_api.py (6 tests, TestClient, offline engine, temp SQLite).

### Step 9.5 — MCP server + Chrome extension (done, 2026-10-26)
- `truth_engine/mcp_server.py`: **mcp 2.x API** (`from mcp.server.mcpserver
  import MCPServer`; FastMCP was renamed in v2). Tools audit_text, ask_question,
  watch_claim, recheck_watched_claims; compact JSON (≤2 evidence per side,
  why ≤ 400 chars; ~2.2 kB for 2 claims). Lazy engine in module `_orch`
  (tests inject an offline Orchestrator). stdio; logs quieted. Verified over a
  real stdio subprocess with production models: 25 s first call (model load).
  tests/test_mcp.py uses `mcp.client.client.Client(server)` in-process.
  User has NOT registered it in Claude yet (persistent config: ask first).
- `extension/` (Chrome MV3): context menu on selection → background worker
  injects render.js + content.js (activeTab) and POSTs /api/audit
  (host_permissions localhost:8000; other servers via optional permission
  requested in options). Panel in a closed shadow DOM, DOM APIs only (no
  innerHTML); popup (paste or page selection); options (server URL, live).
  `extension/dev/harness.html` + fixture.json stub chrome.* and drive the real
  scripts; verified in the preview (panel, close, XSS title rendered as text,
  no console errors). Launch entry "truth-engine-ext-dev" (port 8601).
  Not tested in real Chrome by me (needs Load unpacked by the user).

### Step 10 — tests, end-to-end runs, docs (done, 2026-10-27)
- MCP server registered for the user in ~/.claude.json top-level
  `mcpServers.truth-engine` (stdio, OMP_NUM_THREADS=2); backup at
  ~/.claude.json.bak-truth-engine. No `claude` CLI on PATH (desktop app only).
- `python -m truth_engine setup`: NLI large + reranker download, design
  classifier fetch+train; each skipped if present. Found via a fresh-clone
  test: without it the reranker gate and classifier silently fall back.
- ruff (pyproject.toml: E,F,W; E501 ignored) clean; requirements-dev.txt.
  133 tests pass in fast config, production config (TE_EMBEDDINGS=st
  TE_NLI=1) and a fresh GitHub clone (10 model tests skip there). Coverage
  79% (lowest: LLM HTTP providers, web search, classifier training path).
- Benchmark rerun (current system, dev 300): ours 0.600 acc, macro-F1 0.567,
  ECE 0.043, coverage 36% at 0.771, citation recall 0.70, 1.6 s/claim.
  Ablations Δ (full − variant): nli +0.150 (only significant); abstention /
  calibration −0.017 (but answered-precision 0.77 vs 0.68, ECE 0.043 vs
  0.246); source_quality −0.027 [−0.057, +0.007] n.s. (design classifier
  down-weights animal/in-vitro abstracts SciFact accepts) — not changed on
  dev; would need a train check; adaptive +0.003; falsification /
  independence 0; critic +0.010 n.s., 3.6× slower.
- README: table fixes (retrieval row = bge-base embedding search on 300
  *train* claims), setup uses `truth_engine setup`. CONCEPTS.md: bi- vs
  cross-encoders, fine-tuning, hard negatives, stage-wise error analysis,
  distant supervision, paired bootstrap, calibration domain shift, MCP,
  MV3 extensions, lint/coverage.

### Step 11 — deploy + MLOps (2026-10-27)
- MCP fix: Claude launched the server from another cwd → ModuleNotFoundError.
  Project is now an installable package (pyproject [project], `pip install
  -e .`; console scripts `truth-engine`, `truth-engine-mcp`). Verified the
  handshake from a foreign cwd.
- `models_registry.py`: REVISIONS pins HF commits (NLI large/base, MiniLM
  reranker, bge-base; downloads use resolve/<sha>, SentenceTransformer
  revision=); `write_manifest()` → data/models/MANIFEST.json (SHA-256 of 20
  files incl. calibrator + classifier); `verify()`; summary in /api/health.
  `truth-engine setup` also warms bge-base and writes the manifest.
- `api/monitoring.py`: Metrics (requests/latency histograms per path, verdict
  and decision counts, confidence, live runs, 5xx, rate-limited), drift vs
  SciFact dev mix (support 0.26 / contradicted 0.103 / unverified 0.637,
  abstain 0.637; alert if any grouped share moves > 0.25 after ≥ 50 claims),
  Prometheus /metrics, /api/stats (+ feedback agreement), JSON access log
  with hashed client id. RateLimiter token bucket (TE_RATE_LIMIT/min on
  audit/stream/ask/recheck → 429). TE_ALLOW_LIVE=0 → 403 for live; UI
  disables the live checkboxes.
- Dockerfile (python:3.12-slim, CPU torch wheel, uid 1000, `truth-engine
  setup` + verify at build, HEALTHCHECK, port 7860, TE_RATE_LIMIT=10).
  .dockerignore excludes .env, models, caches, training, extension.
  **Not built locally (no Docker on the Mac): CI builds and smoke-tests it.**
- Workflows: ci.yml (ruff + pytest), docker.yml (build, wait ready, audit
  smoke test), eval-gate.yml (SciFact dev, caches models + embeddings,
  `scripts/eval_gate.py` vs bench/eval_baseline.json with tolerances acc /
  macro-F1 0.02, answered-precision / citation recall 0.03, ECE 0.02;
  verified it fails on a degraded report). Baseline is from the Mac; if the
  first Linux run differs within noise, re-baseline from its artifact.
- Data licensing: HealthVer / COVID-Fact / derived train files and 7k PubMed
  abstracts untracked; classifier set published as 7,199 PMIDs + labels
  (`dataset()` re-fetches). Old copies remain in git history (rewrite only
  with the user's go-ahead).
- `scripts/deploy_space.py <user>/<space>` (HF_TOKEN): uploads tracked files
  minus training/extension/.github, prepends Space YAML header to README.

### Steps 8.6–10 (intended design)
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
  pipeline/
    analyzer.py      claim/question → domain, type, difficulty, risk, budget
    router.py        tier strategies, escalation, LinUCB bandit, IPS
    solvers.py       Ask-mode candidate answers + diversity report
    nli.py           local DeBERTa NLI (PyTorch), memoized, cuda/mps/cpu
    claims.py        text → atomic claims (+ spans, kind, population, hedging,
                     check-worthiness); optional LLM extraction
    verifier.py      stance per evidence sentence; verdict per claim via
                     lineage-aware noisy-OR; optional LLM second opinion
    audit.py         step-4 pipeline + CLI
  science/
    red_flags.py     11 red-flag checks on claim wording + root studies
    venues.py        journal standing / vetting (OpenAlex)
    retractions.py   Crossref / Retraction Watch notices
    study_design.py  design, evidence-hierarchy weight, n, prereg, funding,
                     species, retraction → StudyProfile (+ .label)
    pubmed.py        E-utilities esearch/efetch + XML parser
    openalex.py      search, work, cited_by, references; abstract rebuild
    papers.py        PaperRecord (merge across APIs) → Source + Passages
    search.py        ScienceSearch: queries, merging, review→trial derivation
    cache.py         JsonCache, RateLimiter, CachedFetcher
data/corpus/         50 demo docs (encyclopedia, research, products, company)
data/graph.json      knowledge-graph seed with provenance per triple
```

## Environment notes

- Python 3.11+ (developed on 3.13). Install: `pip install -r requirements.txt`.
- Env vars for real models/search: see README "Works with or without API keys".
- Default Anthropic models: `claude-opus-5`, `claude-sonnet-5-5`,
  `claude-haiku-4-5-20251001`. Default OpenAI names (`gpt-4.1`,
  `gpt-4.1-mini`) may be stale; override with `TE_OPENAI_STRONG/FAST`.
- PubMed and OpenAlex are reachable locally (verified in step 3).
- Keys go in `~/ai-truth-engine/.env` (gitignored; loaded by config.py,
  real env vars win; `.env.example` lists them). Never commit or echo keys. SciFact S3
  not yet checked (step 8).
- Use the project venv: `.venv/bin/python -m pytest -q`.
- **Hardware: the user's Mac is arm64 with 8 GB RAM** (disk was nearly full;
  ~20 GB freed on 2026-10-06). Large NLI (~1.7 GB fp32) + bge-base fit on
  CPU; running both on Apple's GPU (mps) pinned memory and swapped the
  machine to a crawl, so `TE_DEVICE` auto = cuda else cpu. Long benchmark
  runs (full SciFact, other datasets) and all training belong on the user's
  **Colab A100 (100 compute units)**; keep local runs to subsets.
- Tests run with `TE_NLI=0 TE_EMBEDDINGS=hashing` (conftest) for speed and
  determinism; the production config (bge-base + large NLI) also passes:
  `TE_EMBEDDINGS=st TE_NLI=1 .venv/bin/python -m pytest -q`.

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
- Literature retrieval is topical, not stance-aware; the subject gate drops
  off-subject passages but the step-4 verifier still judges stance.
- Source tier grades the *paper*, not the claim: an animal or cohort study
  can be tier A/B. Generalization (mouse→human, correlation→causation) is a
  step-4 red flag, not a quality penalty.
- Stance is ~70% accurate on SciFact fused (torch). NLI can't do arithmetic
  and misreads claims attributed to a named study ("Okafor's trial found…")
  when the name isn't in the text; causal claims rely on the causal_denial
  rule because NLI reads "not by declining demand" as neutral.
- Aggregation saturates: three supportive meta-analyses outvote one strong
  null RCT (live: "Vitamin D prevents depression" → SUPPORTED). Step 6 must
  weigh design and detect conflict (critic, abstention).
- Audit runs sequentially (~30 s live, cold cache, for 4 claims with
  escalation); parallelize in step 7.
- Multi-hop questions: extractive solvers agree on the intermediate entity
  (Einstein) while the graph gets the answer (Germany); majority vote is
  wrong there. Step 6's judge must weigh solver type / verification.
- Journal standing is citation-based (prestige ≠ truth), hence capped at ±0.05.
  Subject gate matches "vitamin" for "Vitamin D" via stems (single-letter
  tokens are dropped), so "vitamin B12" passages can pass.

## Checks

```
python -m pytest -q          # unit tests
```
