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
| 8 | Benchmark (SciFact), baselines A–D, ablations, calibration report card | ⏳ **Next** |
| 8.5 | Fine-tuning on A100: verifier, evidence reranker, study-design classifier, check-worthiness | Planned |
| 8.6 | Open-access full text (PubMed Central / Europe PMC) with reranked passage selection | Planned |
| 9 | FastAPI server, CLI, audit-first web UI | Planned |
| 10 | Tests, end-to-end runs, final docs | Planned |
| 11 | Deploy + MLOps: Docker, HF Spaces, CI eval gate, model versioning, monitoring | Planned |
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

### Steps 8–10 (intended design)
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
