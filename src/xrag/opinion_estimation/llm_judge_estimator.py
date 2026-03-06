"""LLM-as-judge opinion estimator.

Prompts an LLM to assess document relevance and support for a query,
then maps the structured judgment to an SL opinion.

Architecture:
    LLMBackend (ABC)
        ├── HuggingFaceBackend  (local models — stub, deferred)
        └── APIBackend          (OpenAI/Anthropic — stub, deferred)

    LLMJudgeEstimator(BaseOpinionEstimator)
        Uses any LLMBackend to produce LLMJudgment → Opinion.

Opinion mapping:
    The LLM produces relevance_score (r) and support_score (s), both in [0, 1].

    effective_support    = r × s
    effective_contradict = r × (1 − s)
    effective_unknown    = 1 − r

    base_u = 1 / (W + 1)
    scale  = 1 − base_u

    b = effective_support    × scale
    d = effective_contradict × scale
    u = base_u + effective_unknown × scale

    Algebraic proof that b + d + u = 1:
        b + d = (r×s + r×(1−s)) × scale = r × scale
        u = base_u + (1−r) × scale
        b + d + u = r×scale + base_u + (1−r)×scale
                  = scale × (r + 1 − r) + base_u
                  = scale + base_u
                  = (1 − base_u) + base_u = 1  ✓
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from jsonld_ex.confidence_algebra import Opinion

from xrag.opinion_estimation.base import BaseOpinionEstimator, EstimationResult


# ---------------------------------------------------------------------------
# Enums — categorical levels with normalized scores
# ---------------------------------------------------------------------------


class RelevanceLevel(Enum):
    """How relevant a document is to the query.

    Each member has a .score attribute in [0, 1].
    """

    HIGH = ("high", 1.0)
    MEDIUM = ("medium", 0.67)
    LOW = ("low", 0.33)
    NONE = ("none", 0.0)

    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self._score = score

    @property
    def score(self) -> float:
        return self._score

    @classmethod
    def from_string(cls, s: str) -> RelevanceLevel:
        """Parse a string label (case-insensitive) to a RelevanceLevel."""
        s_lower = s.strip().lower()
        for member in cls:
            if member.label == s_lower:
                return member
        raise ValueError(f"Unknown relevance level: {s!r}")


class SupportLevel(Enum):
    """How much a document supports or contradicts the query/answer.

    Each member has a .score attribute in [0, 1].
    1.0 = full support, 0.0 = full contradiction.
    """

    SUPPORTS = ("supports", 1.0)
    PARTIALLY_SUPPORTS = ("partially_supports", 0.67)
    NEUTRAL = ("neutral", 0.33)
    CONTRADICTS = ("contradicts", 0.0)

    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self._score = score

    @property
    def score(self) -> float:
        return self._score

    @classmethod
    def from_string(cls, s: str) -> SupportLevel:
        """Parse a string label (case-insensitive) to a SupportLevel."""
        s_lower = s.strip().lower()
        for member in cls:
            if member.label == s_lower:
                return member
        raise ValueError(f"Unknown support level: {s!r}")


# ---------------------------------------------------------------------------
# LLMJudgment dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMJudgment:
    """Structured judgment from an LLM about a (query, document) pair.

    Attributes:
        relevance: Categorical relevance level.
        relevance_score: Normalized relevance score in [0, 1].
        support: Categorical support level.
        support_score: Normalized support score in [0, 1].
        raw_response: The raw string response from the LLM.
    """

    relevance: RelevanceLevel
    relevance_score: float
    support: SupportLevel
    support_score: float
    raw_response: str


# ---------------------------------------------------------------------------
# LLM backend abstraction
# ---------------------------------------------------------------------------


class LLMBackend(abc.ABC):
    """Abstract base for LLM generation backends."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The full prompt string.

        Returns:
            The LLM's response string.
        """
        ...


class HuggingFaceBackend(LLMBackend):
    """Local HuggingFace model backend (stub).

    Full implementation deferred — requires model loading,
    tokenization, and generation with transformers.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            f"HuggingFaceBackend({self.model_name}) is a stub. "
            "Full implementation requires model loading and generation setup."
        )


class APIBackend(LLMBackend):
    """API-based LLM backend for OpenAI/Anthropic (stub).

    Full implementation deferred — requires API key configuration
    and client library integration.
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
    ) -> None:
        self.provider = provider
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            f"APIBackend({self.provider}/{self.model_name}) is a stub. "
            "Full implementation requires API key and client setup."
        )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """You are an expert judge evaluating whether a document is relevant to a question and whether it supports or contradicts the answer.

Given the following question and document, assess:
1. **Relevance**: How relevant is the document to answering the question?
   - "high": Directly addresses the question
   - "medium": Partially relevant, contains some useful information
   - "low": Tangentially related
   - "none": Completely irrelevant

2. **Support**: Does the document support or contradict a correct answer?
   - "supports": Provides evidence for a correct answer
   - "partially_supports": Some evidence for, some ambiguous
   - "neutral": Neither supports nor contradicts
   - "contradicts": Provides evidence against a correct answer

Respond with ONLY a JSON object, no other text:
{{"relevance": "<level>", "support": "<level>"}}

Question: {question}

Document: {document}

JSON response:"""


# ---------------------------------------------------------------------------
# LLMJudgeEstimator
# ---------------------------------------------------------------------------

# Default base rate for SL opinions
_DEFAULT_BASE_RATE = 0.5

# Default evidence weight
_DEFAULT_EVIDENCE_WEIGHT = 10.0


class LLMJudgeEstimator(BaseOpinionEstimator):
    """Opinion estimator that uses an LLM as a judge.

    Prompts an LLM to assess relevance and support, then maps
    the structured judgment to an SL opinion.

    Args:
        backend: An LLMBackend instance for generation.
        evidence_weight: W in base_u = 1/(W+1). Higher → lower base uncertainty.
            Must be positive.
    """

    def __init__(
        self,
        backend: LLMBackend,
        evidence_weight: float = _DEFAULT_EVIDENCE_WEIGHT,
    ) -> None:
        if evidence_weight <= 0:
            raise ValueError(
                f"evidence_weight must be positive, got {evidence_weight}"
            )
        self.backend = backend
        self.evidence_weight = evidence_weight

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, document: str) -> str:
        """Build the judge prompt for a (query, document) pair.

        Args:
            query: The question string.
            document: The document text.

        Returns:
            The formatted prompt string.
        """
        return _JUDGE_PROMPT_TEMPLATE.format(question=query, document=document)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, response: str) -> LLMJudgment:
        """Parse the LLM's JSON response into an LLMJudgment.

        Handles JSON wrapped in markdown fences, case variations,
        and gracefully falls back to vacuous judgment on parse errors.

        Args:
            response: The raw LLM response string.

        Returns:
            LLMJudgment with parsed levels and scores.
        """
        # Strip markdown JSON fences if present
        cleaned = response.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            # Fallback: unparseable → vacuous judgment
            return LLMJudgment(
                relevance=RelevanceLevel.NONE,
                relevance_score=RelevanceLevel.NONE.score,
                support=SupportLevel.NEUTRAL,
                support_score=SupportLevel.NEUTRAL.score,
                raw_response=response,
            )

        # Parse relevance
        try:
            relevance = RelevanceLevel.from_string(data.get("relevance", "none"))
        except (ValueError, AttributeError):
            relevance = RelevanceLevel.NONE

        # Parse support
        try:
            support = SupportLevel.from_string(data.get("support", "neutral"))
        except (ValueError, AttributeError):
            support = SupportLevel.NEUTRAL

        return LLMJudgment(
            relevance=relevance,
            relevance_score=relevance.score,
            support=support,
            support_score=support.score,
            raw_response=response,
        )

    # ------------------------------------------------------------------
    # Judgment → Opinion mapping
    # ------------------------------------------------------------------

    def judgment_to_opinion(self, judgment: LLMJudgment) -> Opinion:
        """Map an LLMJudgment to an SL Opinion.

        Mapping formula:
            r = relevance_score, s = support_score
            effective_support    = r × s
            effective_contradict = r × (1 − s)
            effective_unknown    = 1 − r
            base_u = 1 / (W + 1)
            scale  = 1 − base_u
            b = effective_support    × scale
            d = effective_contradict × scale
            u = base_u + effective_unknown × scale

        Args:
            judgment: The structured LLM judgment.

        Returns:
            SL Opinion satisfying b + d + u = 1.
        """
        r = judgment.relevance_score
        s = judgment.support_score

        base_u = 1.0 / (self.evidence_weight + 1.0)
        scale = 1.0 - base_u

        b = r * s * scale
        d = r * (1.0 - s) * scale
        u = base_u + (1.0 - r) * scale

        return Opinion(
            belief=b,
            disbelief=d,
            uncertainty=u,
            base_rate=_DEFAULT_BASE_RATE,
        )

    # ------------------------------------------------------------------
    # BaseOpinionEstimator interface
    # ------------------------------------------------------------------

    def estimate(self, query: str, document: str) -> EstimationResult:
        """Estimate opinion for a single (query, document) pair.

        Args:
            query: The query string.
            document: The document string.

        Returns:
            EstimationResult with SL opinion and metadata.
        """
        prompt = self.build_prompt(query, document)
        response = self.backend.generate(prompt)
        judgment = self.parse_response(response)
        opinion = self.judgment_to_opinion(judgment)

        return EstimationResult(
            opinion=opinion,
            nli_scores=None,  # LLM judge does not produce NLI scores
            metadata={"judgment": judgment},
        )

    def estimate_batch(
        self,
        queries: Sequence[str],
        documents: Sequence[str],
    ) -> list[EstimationResult]:
        """Estimate opinions for a batch of (query, document) pairs.

        Currently calls estimate() sequentially. Future optimization:
        batch API calls or parallel generation.

        Args:
            queries: List of query strings.
            documents: List of document strings. Must be same length.

        Returns:
            List of EstimationResult, one per pair.

        Raises:
            ValueError: If queries and documents have different lengths.
        """
        if len(queries) != len(documents):
            raise ValueError(
                f"queries and documents must have same length, "
                f"got {len(queries)} and {len(documents)}"
            )

        if len(queries) == 0:
            return []

        return [
            self.estimate(q, d) for q, d in zip(queries, documents)
        ]
