"""Conflict detection layer — identifies contradictions between documents.

Uses SL pairwise_conflict() and conflict_metric() to detect disagreement.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from jsonld_ex.confidence_algebra import Opinion


@dataclass
class ConflictResult:
    """Result of conflict detection across document opinions.

    Attributes:
        conflict_detected: Whether significant conflict was found.
        conflict_pairs: List of (i, j) index pairs with high conflict.
        max_conflict_score: Highest pairwise conflict score observed.
    """

    conflict_detected: bool
    conflict_pairs: list[tuple[int, int]]
    max_conflict_score: float


class ConflictLayer(abc.ABC):
    """Abstract base for conflict detection layers."""

    @abc.abstractmethod
    def detect(self, opinions: list[Opinion]) -> ConflictResult:
        """Detect conflicts among document opinions.

        Args:
            opinions: Per-document SL opinions.

        Returns:
            ConflictResult summarizing detected conflicts.
        """
        ...


class NoOpConflictLayer(ConflictLayer):
    """No-op conflict layer — always reports no conflict."""

    def detect(self, opinions: list[Opinion]) -> ConflictResult:
        return ConflictResult(
            conflict_detected=False,
            conflict_pairs=[],
            max_conflict_score=0.0,
        )
