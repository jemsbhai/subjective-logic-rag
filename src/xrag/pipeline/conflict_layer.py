"""Conflict detection layer — identifies contradictions between documents.

Uses SL pairwise_conflict() from jsonld-ex to build an NxN conflict
matrix and flag document pairs whose conflict exceeds a threshold.

Design:
    - SLConflictLayer builds the full pairwise conflict matrix.
    - Pairs exceeding threshold are flagged in conflict_pairs.
    - Enriched metadata: conflict_matrix, mean_conflict, discord_scores.
    - Full traceability via self.last_result.

Paper 1 usage:
    - Main experiments: threshold-based conflict flagging
    - Corruption suites: detect injected contradictions
    - Ablation A8: NoOpConflictLayer (skip conflict detection)
    - Paper figures: conflict matrix visualization (Fig 6)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from jsonld_ex.confidence_algebra import Opinion, pairwise_conflict


# ═══════════════════════════════════════════════════════════════════
# ConflictResult — returned by detect()
# ═══════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════
# SLConflictResult — enriched metadata for the real implementation
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SLConflictResult:
    """Enriched conflict detection result with full metadata.

    Extends ConflictResult with the conflict matrix, mean conflict,
    and per-doc discord scores for paper analysis and visualization.

    Attributes:
        conflict_detected:  Whether any pair exceeds threshold.
        conflict_pairs:     List of (i, j) pairs above threshold.
        max_conflict_score: Highest pairwise conflict.
        conflict_matrix:    NxN symmetric pairwise conflict matrix.
        mean_conflict:      Mean of all upper-triangle entries.
        discord_scores:     Per-doc mean conflict with all other docs.
    """

    conflict_detected: bool
    conflict_pairs: list[tuple[int, int]]
    max_conflict_score: float
    conflict_matrix: list[list[float]]
    mean_conflict: float
    discord_scores: list[float]


# ═══════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════
# NoOp (unchanged — backward compatible)
# ═══════════════════════════════════════════════════════════════════


class NoOpConflictLayer(ConflictLayer):
    """No-op conflict layer — always reports no conflict."""

    def detect(self, opinions: list[Opinion]) -> ConflictResult:
        return ConflictResult(
            conflict_detected=False,
            conflict_pairs=[],
            max_conflict_score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════
# Real implementation
# ═══════════════════════════════════════════════════════════════════


class SLConflictLayer(ConflictLayer):
    """Conflict detection layer using SL pairwise_conflict().

    Builds the full NxN pairwise conflict matrix and flags pairs
    whose conflict score strictly exceeds the threshold.

    pairwise_conflict(A, B) = b_A*d_B + d_A*b_B

    This measures evidential tension: the degree to which one source's
    belief overlaps with the other's disbelief.

    Args:
        threshold: Conflict score above which a pair is flagged.
                   Default 0.3 (moderately conservative).
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold
        self.last_result: SLConflictResult | None = None

    def detect(self, opinions: list[Opinion]) -> ConflictResult:
        """Detect conflicts among document opinions.

        Args:
            opinions: Per-document SL opinions.

        Returns:
            ConflictResult with conflict_detected, conflict_pairs,
            and max_conflict_score.  Full metadata available via
            self.last_result.
        """
        n = len(opinions)

        # Handle trivial cases
        if n <= 1:
            self.last_result = SLConflictResult(
                conflict_detected=False,
                conflict_pairs=[],
                max_conflict_score=0.0,
                conflict_matrix=[[0.0] * n for _ in range(n)],
                mean_conflict=0.0,
                discord_scores=[0.0] * n,
            )
            return ConflictResult(
                conflict_detected=False,
                conflict_pairs=[],
                max_conflict_score=0.0,
            )

        # Build NxN pairwise conflict matrix
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = pairwise_conflict(opinions[i], opinions[j])
                matrix[i][j] = c
                matrix[j][i] = c

        # Find max conflict and flagged pairs
        max_score = 0.0
        flagged_pairs: list[tuple[int, int]] = []
        total_conflict = 0.0
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                c = matrix[i][j]
                total_conflict += c
                pair_count += 1
                if c > max_score:
                    max_score = c
                if c > self.threshold:
                    flagged_pairs.append((i, j))

        mean_conflict = total_conflict / pair_count if pair_count > 0 else 0.0

        # Per-doc discord scores: mean conflict with all other docs
        discord_scores: list[float] = []
        for i in range(n):
            doc_total = 0.0
            for j in range(n):
                if i != j:
                    doc_total += matrix[i][j]
            discord_scores.append(doc_total / (n - 1) if n > 1 else 0.0)

        conflict_detected = len(flagged_pairs) > 0

        self.last_result = SLConflictResult(
            conflict_detected=conflict_detected,
            conflict_pairs=flagged_pairs,
            max_conflict_score=max_score,
            conflict_matrix=matrix,
            mean_conflict=mean_conflict,
            discord_scores=discord_scores,
        )

        return ConflictResult(
            conflict_detected=conflict_detected,
            conflict_pairs=flagged_pairs,
            max_conflict_score=max_score,
        )
