"""Core data model for the AI Truth Engine.

Every stage of the pipeline reads and writes these types, and the full
investigation is captured in a single `Trace` that maps onto the Section 14
training-data schema of the specification.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------- #
# Question analysis and routing (Improvements 1, 2)
# --------------------------------------------------------------------------- #

class Domain(str, Enum):
    MATHEMATICS = "mathematics"
    CODING = "coding"
    BUSINESS = "business"
    SCIENCE = "science"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    FINANCE = "finance"
    MEDICINE = "medicine"
    LAW = "law"
    TECHNOLOGY = "technology"
    GENERAL = "general"


class QuestionType(str, Enum):
    FACTUAL = "factual"
    PROOF = "proof"
    CALCULATION = "calculation"
    CAUSAL = "causal"
    DEBUGGING = "debugging"
    RESEARCH_SYNTHESIS = "research_synthesis"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    OPINION = "opinion"
    DEFINITION = "definition"


class Risk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BudgetTier(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


BUDGET_ORDER = [BudgetTier.MINIMAL, BudgetTier.LOW, BudgetTier.MEDIUM, BudgetTier.HIGH, BudgetTier.MAXIMUM]


class QuestionAnalysis(BaseModel):
    """Output of the question analyzer (spec §4, Improvement 1)."""
    domain: Domain
    question_type: QuestionType
    difficulty: float = Field(ge=0, le=1)
    risk: Risk
    requires_retrieval: bool
    requires_current_data: bool = False
    tools: list[str] = Field(default_factory=list)
    compute_budget: BudgetTier
    entities: list[str] = Field(default_factory=list)
    signals: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class ComputePlan(BaseModel):
    """What the router decided to spend (Improvement 2)."""
    tier: BudgetTier
    solver_models: list[str]
    verifier_model: str
    critic_model: str | None
    judge_model: str
    retrieval: bool
    retrieval_sources: list[str] = Field(default_factory=list)
    contradiction_search: bool
    adversarial_review: bool
    use_graph: bool
    tools: list[str] = Field(default_factory=list)
    max_cost_usd: float
    max_latency_s: float
    escalation_allowed: bool = True
    escalation_threshold: float = 0.55
    routing_source: Literal["rules", "learned", "forced"] = "rules"
    rationale: str = ""


# --------------------------------------------------------------------------- #
# Candidate answers (Improvement 3)
# --------------------------------------------------------------------------- #

class CandidateAnswer(BaseModel):
    id: str = Field(default_factory=lambda: _id("cand"))
    model: str
    role: str = "solver"
    answer: str
    reasoning: str = ""
    final_value: str | None = None
    tool_calls: list["ToolCall"] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


class DiversityReport(BaseModel):
    """Measured (not assumed) independence of the candidate set (Challenge 1)."""
    n_candidates: int
    agreement_rate: float
    answer_clusters: list[list[str]]
    pairwise_similarity: dict[str, float]
    majority_value: str | None
    majority_share: float
    correlated_error_risk: float
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Claims and evidence (Improvements 5, 6, 8)
# --------------------------------------------------------------------------- #

class ClaimKind(str, Enum):
    FACTUAL = "factual"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    INFERENTIAL = "inferential"
    DEFINITIONAL = "definitional"
    MATHEMATICAL = "mathematical"


class EvidenceStatus(str, Enum):
    """The six evidence statuses of spec §4, Improvement 5 (table)."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    INFERRED = "inferred"
    SPECULATIVE = "speculative"


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: _id("clm"))
    text: str
    kind: ClaimKind
    source_candidates: list[str] = Field(default_factory=list)
    importance: float = 0.5
    numbers: list[float] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    is_causal: bool = False
    # audit mode (step 4)
    sentence: str = ""                       # the original sentence it came from
    span: tuple[int, int] | None = None      # char offsets of that sentence in the input
    science_kind: str = "general"            # efficacy | safety | association | mechanism | statistic | causal | general
    population: str = "human"                # who the claim is about: human | animal | in_vitro
    hedged: bool = False                     # "may", "could", "suggests"
    check_worthiness: float = 1.0


class SourceQuality(BaseModel):
    primary: bool
    authority: float
    recency: float
    methodology: float
    directness: float
    overall: float
    tier: Literal["A", "B", "C", "D"]
    notes: list[str] = Field(default_factory=list)


class Source(BaseModel):
    id: str
    title: str
    url: str | None = None
    source_type: str = "document"  # document | web | database | paper | tool | graph
    publisher: str | None = None
    published: str | None = None
    authority: float = 0.5
    is_primary: bool = False
    cites: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)
    quality: SourceQuality | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Passage(BaseModel):
    id: str
    source_id: str
    text: str
    score: float = 0.0
    retrieval_method: str = "hybrid"


class Stance(str, Enum):
    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    PARTIAL = "partial"


class EvidenceItem(BaseModel):
    passage: Passage
    source: Source
    stance: Stance
    stance_score: float
    relevance: float
    found_by: Literal["supporting_search", "contradiction_search", "graph", "tool", "database"]
    rationale: str = ""
    applicability: float = 1.0               # does this study's design/species fit the claim? (0..1)
    weight: float = 0.0                      # quality x applicability x stance strength
    lineage: int | None = None               # index into ClaimVerdict.lineages
    llm_stance: Stance | None = None         # second opinion, when an LLM verifier ran
    rule_stance: Stance | None = None        # what the word-level rules alone said
    nli: dict[str, float] | None = None      # local NLI model probabilities, when it ran


class RedFlag(BaseModel):
    """A classic distortion of scientific evidence (CONCEPTS.md §5)."""
    kind: Literal["animal_to_human", "in_vitro_to_human", "correlation_as_causation", "relative_risk_only",
                  "single_study", "failed_replication", "small_sample", "industry_funded", "retracted",
                  "overclaimed_certainty", "echo_chamber"]
    severity: float = Field(ge=0, le=1)
    message: str
    source_ids: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim_id: str
    status: EvidenceStatus
    confidence: float
    supporting: list[EvidenceItem] = Field(default_factory=list)
    contradicting: list[EvidenceItem] = Field(default_factory=list)
    independent_support_count: int = 0
    source_independence: float = 0.0
    rationale: str = ""
    support_score: float = 0.0               # noisy-OR over independent supporting lineages
    contradiction_score: float = 0.0
    conflicting: bool = False                # strong evidence on both sides
    supporting_source_count: int = 0         # raw count, for "N sources -> K independent"
    lineages: list[list[str]] = Field(default_factory=list)
    red_flags: list[RedFlag] = Field(default_factory=list)
    verifier_agreement: float | None = None  # rule-based vs LLM stance agreement, when an LLM ran


# --------------------------------------------------------------------------- #
# Critique, judging, calibration (Improvements 4, 9)
# --------------------------------------------------------------------------- #

class CritiqueFinding(BaseModel):
    kind: Literal["counterevidence", "hidden_assumption", "alternative_explanation",
                  "correlation_not_causation", "unsupported_number", "logical_gap",
                  "missing_condition", "counterexample", "scope_overreach"]
    target_claim_id: str | None = None
    description: str
    severity: float = Field(ge=0, le=1)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class CritiqueReport(BaseModel):
    model: str
    findings: list[CritiqueFinding] = Field(default_factory=list)
    survived: bool = True
    confidence_adjustment: float = 0.0
    alternative_hypotheses: list[str] = Field(default_factory=list)


class ConfidenceBreakdown(BaseModel):
    """Every signal named in spec §4, Improvement 9."""
    model_agreement: float
    evidence_support: float
    source_quality: float
    verifier_agreement: float
    historical_reliability: float
    source_independence: float
    contradiction: float
    unresolved_uncertainty: float
    difficulty_penalty: float
    raw_score: float
    calibrated: float
    label: Literal["VERY LOW", "LOW", "MODERATE", "HIGH", "VERY HIGH"]


class Decision(str, Enum):
    ANSWER = "answer"
    ANSWER_WITH_CAVEATS = "answer_with_caveats"
    ABSTAIN = "abstain"


class JudgeResult(BaseModel):
    model: str
    selected_candidate_id: str | None
    decision: Decision
    rationale: str
    rejected: list[dict[str, str]] = Field(default_factory=list)
    reliability_requirements_met: bool


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

class ToolCall(BaseModel):
    tool: str
    input: Any
    output: Any = None
    ok: bool = True
    error: str | None = None
    latency_s: float = 0.0


# --------------------------------------------------------------------------- #
# Investigation tree (spec §12)
# --------------------------------------------------------------------------- #

class TreeNode(BaseModel):
    id: str = Field(default_factory=lambda: _id("node"))
    label: str
    detail: str = ""
    value: str | None = None
    relation: Literal["root", "contributes", "explains", "correlates", "causes",
                      "rejected", "supports", "contradicts", "step"] = "step"
    status: EvidenceStatus | None = None
    weight: float = 0.0
    children: list["TreeNode"] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Events and the full trace (spec §14)
# --------------------------------------------------------------------------- #

class StageEvent(BaseModel):
    stage: str
    status: Literal["started", "completed", "skipped", "escalated", "failed"]
    message: str
    t: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    decision: Decision
    summary: str
    body: str
    confidence: ConfidenceBreakdown
    uncertainty: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    citations: dict[str, list[str]] = Field(default_factory=dict)  # claim_id -> source ids


class AuditReport(BaseModel):
    """Audit mode output: every check-worthy claim in a text, verified."""
    id: str = Field(default_factory=lambda: _id("audit"))
    text: str
    claims: list[Claim] = Field(default_factory=list)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)   # sentences judged not check-worthy
    sources: dict[str, Source] = Field(default_factory=dict)
    retrieval_sources: list[str] = Field(default_factory=list)
    verifiers: list[str] = Field(default_factory=list)            # e.g. ["rules", "nli:<model>"]
    routing: dict[str, dict[str, Any]] = Field(default_factory=dict)  # claim_id -> analysis, tiers, escalations
    errors: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    disclaimer: str = "Not medical advice: this summarizes published evidence and does not replace a clinician."


class Trace(BaseModel):
    """One complete investigation. Field names follow spec §14."""
    query_id: str = Field(default_factory=lambda: _id("q"))
    question: str
    created_at: float = Field(default_factory=time.time)
    analysis: QuestionAnalysis | None = None
    plan: ComputePlan | None = None
    escalations: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[CandidateAnswer] = Field(default_factory=list)
    diversity: DiversityReport | None = None
    claims: list[Claim] = Field(default_factory=list)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    sources: dict[str, Source] = Field(default_factory=dict)
    critique: CritiqueReport | None = None
    judge: JudgeResult | None = None
    final: FinalAnswer | None = None
    tree: TreeNode | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    events: list[StageEvent] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    disabled_components: list[str] = Field(default_factory=list)
    human_feedback: dict[str, Any] | None = None
    benchmark_score: float | None = None

    # Convenience accessors matching §14 names exactly
    @property
    def question_type(self) -> str | None:
        return self.analysis.question_type.value if self.analysis else None

    @property
    def difficulty(self) -> float | None:
        return self.analysis.difficulty if self.analysis else None

    @property
    def models_selected(self) -> list[str]:
        return self.plan.solver_models if self.plan else []

    def section14_record(self) -> dict[str, Any]:
        """Flatten to exactly the §14 field list for training-data export."""
        return {
            "query_id": self.query_id,
            "question": self.question,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "risk": self.analysis.risk.value if self.analysis else None,
            "models_selected": self.models_selected,
            "model_outputs": [c.answer for c in self.candidates],
            "claims": [c.text for c in self.claims],
            "retrieved_sources": list(self.sources.keys()),
            "evidence_scores": {v.claim_id: v.confidence for v in self.verdicts},
            "contradictions": [v.claim_id for v in self.verdicts if v.status == EvidenceStatus.CONTRADICTED],
            "critic_results": self.critique.model_dump() if self.critique else None,
            "final_answer": self.final.summary if self.final else None,
            "confidence": self.final.confidence.calibrated if self.final else None,
            "cost": self.cost_usd,
            "latency": self.latency_s,
            "human_feedback": self.human_feedback,
            "benchmark_score": self.benchmark_score,
        }


CandidateAnswer.model_rebuild()
TreeNode.model_rebuild()
