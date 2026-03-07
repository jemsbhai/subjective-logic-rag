"""Decision layer — generate, abstain, or flag conflict.

Uses fused opinion uncertainty and conflict detection results to make
a principled decision about whether to generate an answer, abstain
due to insufficient evidence, or flag contradictory sources.

Decision logic (priority order):
    1. If max_conflict_score > τ_conflict → FLAG_CONFLICT
    2. If fused_uncertainty > τ_abstain  → ABSTAIN
    3. Else                              → GENERATE

Conflict takes priority: contradicting sources are a qualitatively
different signal than mere ignorance and should be surfaced first.

All comparisons are strictly-greater-than (not >=), so scores exactly
at the threshold produce GENERATE.

Design:
    - SLDecisionLayer takes τ_abstain and τ_conflict thresholds.
    - Full traceability via self.last_result (SLDecisionResult).
    - Paper 1: thresholds tuned on calibration split, reported across
      full range, threshold-free metrics for figures.
    - Ablation A9: NoOpDecisionLayer (always generate, no abstention).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from jsonld_ex.confidence_algebra import Opinion

from xrag.pipeline.conflict_layer import ConflictResult


# ═══════════════════════════════════════════════════════════════════
# DecisionResult — returned by decide()
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DecisionResult:
    """Result of the decision layer.

    Attributes:
        decision: One of "generate", "abstain", "flag_conflict".
        reason: Human-readable explanation for the decision.
    """

    decision: str  # "generate" | "abstain" | "flag_conflict"
    reason: str


# ═══════════════════════════════════════════════════════════════════
# SLDecisionResult — enriched metadata for the real implementation
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SLDecisionResult:
    """Enriched decision result with full traceability.

    Records the actual scores and thresholds so paper analysis
    can measure how close each decision was to the boundary.

    Attributes:
        decision:              One of "generate", "abstain", "flag_conflict".
        reason:                Human-readable explanation.
        fused_uncertainty:     The fused opinion's uncertainty component.
        max_conflict_score:    The highest pairwise conflict from ConflictResult.
        tau_abstain:           The abstention threshold used.
        tau_conflict:          The conflict threshold used.
        projected_probability: P(ω) = b + a·u of the fused opinion.
    """

    decision: str
    reason: str
    fused_uncertainty: float
    max_conflict_score: float
    tau_abstain: float
    tau_conflict: float
    projected_probability: float


# ═══════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════


class DecisionLayer(abc.ABC):
    """Abstract base for decision layers."""

    @abc.abstractmethod
    def decide(
        self,
        fused_opinion: Opinion,
        conflict_result: ConflictResult,
    ) -> DecisionResult:
        """Decide whether to generate, abstain, or flag conflict.

        Args:
            fused_opinion: The aggregate fused opinion.
            conflict_result: Conflict detection results.

        Returns:
            DecisionResult with the action and reason.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# NoOp (unchanged — backward compatible)
# ═══════════════════════════════════════════════════════════════════


class NoOpDecisionLayer(DecisionLayer):
    """No-op decision layer — always decides to generate."""

    def decide(
        self,
        fused_opinion: Opinion,
        conflict_result: ConflictResult,
    ) -> DecisionResult:
        return DecisionResult(decision="generate", reason="no-op: always generate")


# ═══════════════════════════════════════════════════════════════════
# Real implementation
# ═══════════════════════════════════════════════════════════════════


class SLDecisionLayer(DecisionLayer):
    """Threshold-based decision layer using fused uncertainty and conflict.

    Decision logic (priority order, all strictly-greater-than):
        1. max_conflict_score > τ_conflict → FLAG_CONFLICT
        2. fused_uncertainty > τ_abstain   → ABSTAIN
        3. Otherwise                       → GENERATE

    Args:
        tau_abstain:  Uncertainty threshold above which we abstain.
                      Default 0.7 (high uncertainty required to abstain).
        tau_conflict: Conflict score threshold above which we flag.
                      Default 0.5 (moderate conflict required to flag).
    """

    def __init__(
        self,
        tau_abstain: float = 0.7,
        tau_conflict: float = 0.5,
    ) -> None:
        self.tau_abstain = tau_abstain
        self.tau_conflict = tau_conflict
        self.last_result: SLDecisionResult | None = None

    def decide(
        self,
        fused_opinion: Opinion,
        conflict_result: ConflictResult,
    ) -> DecisionResult:
        """Decide whether to generate, abstain, or flag conflict.

        Args:
            fused_opinion: The aggregate fused opinion.
            conflict_result: Conflict detection results.

        Returns:
            DecisionResult with the action and reason.
        """
        u = fused_opinion.uncertainty
        c = conflict_result.max_conflict_score
        pp = fused_opinion.projected_probability()

        # Priority 1: conflict
        if c > self.tau_conflict:
            decision = "flag_conflict"
            reason = (
                f"Conflict score {c:.4f} exceeds threshold "
                f"{self.tau_conflict:.4f} — sources contradict each other."
            )
        # Priority 2: abstain
        elif u > self.tau_abstain:
            decision = "abstain"
            reason = (
                f"Uncertainty {u:.4f} exceeds threshold "
                f"{self.tau_abstain:.4f} — insufficient evidence to answer."
            )
        # Default: generate
        else:
            decision = "generate"
            reason = (
                f"Uncertainty {u:.4f} <= {self.tau_abstain:.4f} and "
                f"conflict {c:.4f} <= {self.tau_conflict:.4f} — "
                f"sufficient evidence to generate (P={pp:.4f})."
            )

        self.last_result = SLDecisionResult(
            decision=decision,
            reason=reason,
            fused_uncertainty=u,
            max_conflict_score=c,
            tau_abstain=self.tau_abstain,
            tau_conflict=self.tau_conflict,
            projected_probability=pp,
        )

        return DecisionResult(decision=decision, reason=reason)
