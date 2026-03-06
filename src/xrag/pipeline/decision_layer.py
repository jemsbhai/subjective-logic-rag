"""Decision layer — generate, abstain, or flag conflict.

Uses fused opinion uncertainty and conflict detection to decide
whether to generate an answer, abstain, or flag a conflict.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from jsonld_ex.confidence_algebra import Opinion

from xrag.pipeline.conflict_layer import ConflictResult


@dataclass
class DecisionResult:
    """Result of the decision layer.

    Attributes:
        decision: One of "generate", "abstain", "flag_conflict".
        reason: Human-readable explanation for the decision.
    """

    decision: str  # "generate" | "abstain" | "flag_conflict"
    reason: str


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


class NoOpDecisionLayer(DecisionLayer):
    """No-op decision layer — always decides to generate."""

    def decide(
        self,
        fused_opinion: Opinion,
        conflict_result: ConflictResult,
    ) -> DecisionResult:
        return DecisionResult(decision="generate", reason="no-op: always generate")
