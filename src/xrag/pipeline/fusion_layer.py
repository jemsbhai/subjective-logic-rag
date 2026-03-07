"""Evidence fusion layer — combines multiple document opinions into one.

Uses SL cumulative_fuse(), averaging_fuse(), robust_fuse(), or byzantine_fuse()
from jsonld-ex.  All six fusion strategies from the ablation table are supported.

The FusionLayer ABC defines the interface; SLFusionLayer is the real
implementation that delegates to jsonld-ex operators.  NoOpFusionLayer
remains for skeleton/testing use.

Design: fuse() returns an Opinion (keeping the pipeline interface simple).
Full metadata (removed indices, conflict matrix, cohesion, per-removal
records) is preserved in self.last_result as a FusionResult dataclass,
accessible after each call.  This ensures zero fidelity loss for
downstream analysis, paper figures, and ablation reporting.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    robust_fuse,
)
from jsonld_ex.confidence_byzantine import (
    AgentRemoval,
    ByzantineConfig,
    ByzantineFusionReport,
    byzantine_fuse,
)


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════

FusionStrategy = Literal[
    "cumulative",
    "averaging",
    "robust",
    "byzantine_most_conflicting",
    "byzantine_least_trusted",
    "byzantine_combined",
]

FUSION_STRATEGIES: tuple[FusionStrategy, ...] = (
    "cumulative",
    "averaging",
    "robust",
    "byzantine_most_conflicting",
    "byzantine_least_trusted",
    "byzantine_combined",
)

# Map our strategy names to jsonld-ex ByzantineConfig strategy values
_BYZANTINE_STRATEGY_MAP: dict[FusionStrategy, str] = {
    "byzantine_most_conflicting": "most_conflicting",
    "byzantine_least_trusted": "least_trusted",
    "byzantine_combined": "combined",
}


# ═══════════════════════════════════════════════════════════════════
# FusionResult — full-fidelity metadata from a fusion operation
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FusionResult:
    """Complete record of a fusion operation, preserving all metadata.

    For cumulative/averaging: removed_indices is empty, byzantine-only
    fields are None.

    For robust: removed_indices populated, byzantine-only fields None.

    For byzantine: all fields populated from ByzantineFusionReport.

    Attributes:
        fused:              The fused opinion.
        strategy:           Which fusion strategy was used.
        input_count:        Number of opinions that went in.
        removed_indices:    Indices of removed agents ([] if none).
        surviving_indices:  Indices of agents that contributed to fusion.
        conflict_matrix:    NxN pairwise conflict matrix (byzantine only).
        cohesion:           Group cohesion of surviving agents (byzantine only).
        removal_details:    Rich per-removal records (byzantine only).
    """

    fused: Opinion
    strategy: str
    input_count: int
    removed_indices: list[int]
    surviving_indices: list[int]
    conflict_matrix: list[list[float]] | None
    cohesion: float | None
    removal_details: list[AgentRemoval] | None


# ═══════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════


class FusionLayer(abc.ABC):
    """Abstract base for evidence fusion layers."""

    @abc.abstractmethod
    def fuse(self, opinions: list[Opinion]) -> Opinion:
        """Fuse multiple document opinions into a single aggregate opinion.

        Args:
            opinions: Per-document SL opinions.

        Returns:
            Fused SL opinion representing aggregate evidence.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# NoOp (unchanged — backward compatible)
# ═══════════════════════════════════════════════════════════════════


class NoOpFusionLayer(FusionLayer):
    """No-op fusion layer — returns a vacuous (max uncertainty) opinion."""

    def fuse(self, opinions: list[Opinion]) -> Opinion:
        return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


# ═══════════════════════════════════════════════════════════════════
# Real implementation
# ═══════════════════════════════════════════════════════════════════


class SLFusionLayer(FusionLayer):
    """Evidence fusion layer using Subjective Logic operators from jsonld-ex.

    Supports all 6 fusion strategies from the paper's ablation table:

    - **cumulative**: Independent source assumption. Uncertainty shrinks
      as evidence accumulates. (Jøsang 2016 §12.3)
    - **averaging**: Correlated source assumption. Idempotent — fusing
      duplicates doesn't change the result. (Jøsang 2016 §12.5)
    - **robust**: Iterative conflict filtering → cumulative fusion.
      Removes outlier agents whose mean discord exceeds threshold.
    - **byzantine_most_conflicting**: Byzantine filtering by discord score.
    - **byzantine_least_trusted**: Byzantine filtering by trust weight.
    - **byzantine_combined**: Byzantine filtering by discord × distrust.

    After each call to fuse(), full metadata is available via
    ``self.last_result`` as a :class:`FusionResult`.

    Args:
        strategy:         One of the 6 FusionStrategy values.
        robust_threshold: Discord threshold for robust fusion (default 0.15).
        byzantine_config: Optional ByzantineConfig override for byzantine strategies.
                          If None, a default config is built from the strategy name
                          and trust_weights.
        trust_weights:    Per-agent trust scores in [0, 1].  Required for
                          byzantine_least_trusted and byzantine_combined.
    """

    def __init__(
        self,
        strategy: FusionStrategy = "cumulative",
        robust_threshold: float = 0.15,
        byzantine_config: ByzantineConfig | None = None,
        trust_weights: list[float] | None = None,
    ) -> None:
        if strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion strategy: {strategy!r}. "
                f"Must be one of {FUSION_STRATEGIES}"
            )
        self.strategy: FusionStrategy = strategy
        self.robust_threshold = robust_threshold
        self.byzantine_config = byzantine_config
        self.trust_weights = trust_weights
        self.last_result: FusionResult | None = None

    def fuse(self, opinions: list[Opinion]) -> Opinion:
        """Fuse opinions using the configured strategy.

        Args:
            opinions: Per-document SL opinions. Must be non-empty.

        Returns:
            Fused SL opinion.

        Raises:
            ValueError: If opinions is empty, or if trust_weights are
                        required but missing/mismatched.
        """
        if len(opinions) == 0:
            raise ValueError("Cannot fuse an empty list of opinions")

        if self.strategy == "cumulative":
            result = self._fuse_cumulative(opinions)
        elif self.strategy == "averaging":
            result = self._fuse_averaging(opinions)
        elif self.strategy == "robust":
            result = self._fuse_robust(opinions)
        else:
            # One of the three byzantine strategies
            result = self._fuse_byzantine(opinions)

        self.last_result = result
        return result.fused

    # ── Strategy implementations ──────────────────────────────────

    def _fuse_cumulative(self, opinions: list[Opinion]) -> FusionResult:
        fused = cumulative_fuse(*opinions)
        return FusionResult(
            fused=fused,
            strategy="cumulative",
            input_count=len(opinions),
            removed_indices=[],
            surviving_indices=list(range(len(opinions))),
            conflict_matrix=None,
            cohesion=None,
            removal_details=None,
        )

    def _fuse_averaging(self, opinions: list[Opinion]) -> FusionResult:
        fused = averaging_fuse(*opinions)
        return FusionResult(
            fused=fused,
            strategy="averaging",
            input_count=len(opinions),
            removed_indices=[],
            surviving_indices=list(range(len(opinions))),
            conflict_matrix=None,
            cohesion=None,
            removal_details=None,
        )

    def _fuse_robust(self, opinions: list[Opinion]) -> FusionResult:
        fused_op, removed = robust_fuse(
            opinions, threshold=self.robust_threshold
        )
        all_indices = set(range(len(opinions)))
        removed_set = set(removed)
        surviving = sorted(all_indices - removed_set)
        return FusionResult(
            fused=fused_op,
            strategy="robust",
            input_count=len(opinions),
            removed_indices=list(removed),
            surviving_indices=surviving,
            conflict_matrix=None,
            cohesion=None,
            removal_details=None,
        )

    def _fuse_byzantine(self, opinions: list[Opinion]) -> FusionResult:
        byz_strategy = _BYZANTINE_STRATEGY_MAP[self.strategy]

        # Build or use provided ByzantineConfig
        if self.byzantine_config is not None:
            cfg = self.byzantine_config
        else:
            # Resolve trust weights
            tw = self.trust_weights
            if byz_strategy in ("least_trusted", "combined") and tw is None:
                raise ValueError(
                    f"Strategy {self.strategy!r} requires trust_weights "
                    f"but none were provided."
                )
            if tw is not None and len(tw) != len(opinions):
                raise ValueError(
                    f"trust_weights length ({len(tw)}) does not match "
                    f"opinions length ({len(opinions)})."
                )
            cfg = ByzantineConfig(
                strategy=byz_strategy,
                trust_weights=tw,
            )

        report: ByzantineFusionReport = byzantine_fuse(opinions, config=cfg)

        removed_indices = [r.index for r in report.removed]

        return FusionResult(
            fused=report.fused,
            strategy=self.strategy,
            input_count=len(opinions),
            removed_indices=removed_indices,
            surviving_indices=list(report.surviving_indices),
            conflict_matrix=report.conflict_matrix,
            cohesion=report.cohesion_score,
            removal_details=list(report.removed),
        )
