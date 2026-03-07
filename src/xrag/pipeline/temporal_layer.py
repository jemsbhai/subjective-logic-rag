"""Temporal decay layer — ages opinions based on document timestamps.

Uses SL decay_opinion() from jsonld-ex to migrate mass from belief and
disbelief into uncertainty, modeling the loss of evidential relevance
over time.

Design:
    - SLTemporalLayer takes a half_life and optional decay_fn.
    - Per-doc elapsed times via the elapsed_times kwarg to apply().
    - elapsed_times=None → passthrough (datasets without timestamps).
    - Individual None entries → that doc skips decay (no timestamp).
    - Full traceability via self.last_result (TemporalResult dataclass).

Paper 1 usage:
    - Main experiments: passthrough (NQ/PopQA have no timestamps)
    - Corruption suites: backdated timestamps → real decay
    - Ablation A6: NoOpTemporalLayer (skip decay entirely)
    - Decay function comparison: exponential vs linear vs step
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import (
    DecayFunction,
    decay_opinion,
    exponential_decay,
)


# ═══════════════════════════════════════════════════════════════════
# TemporalResult — full-fidelity metadata from a decay pass
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TemporalResult:
    """Complete record of a temporal decay pass.

    Attributes:
        opinions_before: The original document opinions (input).
        opinions_after:  The decayed opinions (output).
        elapsed_times:   Per-doc elapsed time used (None if skipped).
        decay_factors:   Per-doc decay factor λ applied (None if skipped).
    """

    opinions_before: list[Opinion]
    opinions_after: list[Opinion]
    elapsed_times: list[float | None]
    decay_factors: list[float | None]


# ═══════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════


class TemporalLayer(abc.ABC):
    """Abstract base for temporal decay layers."""

    @abc.abstractmethod
    def apply(self, opinions: list[Opinion], **kwargs) -> list[Opinion]:
        """Apply temporal decay to document opinions.

        Args:
            opinions: Per-document SL opinions.

        Returns:
            Temporally decayed opinions (same length).
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# NoOp (unchanged — backward compatible)
# ═══════════════════════════════════════════════════════════════════


class NoOpTemporalLayer(TemporalLayer):
    """No-op temporal layer — passes opinions through unchanged."""

    def apply(self, opinions: list[Opinion], **kwargs) -> list[Opinion]:
        return list(opinions)


# ═══════════════════════════════════════════════════════════════════
# Real implementation
# ═══════════════════════════════════════════════════════════════════


class SLTemporalLayer(TemporalLayer):
    """Temporal decay layer using Subjective Logic decay_opinion().

    Applies temporal decay to each document opinion based on its
    elapsed time.  The decay process migrates mass from belief and
    disbelief into uncertainty while preserving the b/d ratio.

    Three built-in decay functions:
        - exponential_decay: λ = 2^(−t/τ)  — smooth, never zero
        - linear_decay: λ = max(0, 1 − t/(2τ))  — hard expiry at 2×τ
        - step_decay: λ = 1 if t < τ else 0  — binary freshness

    Args:
        half_life: Time for belief/disbelief to halve.  Must be positive.
                   Units are arbitrary but must match elapsed_times.
        decay_fn:  Decay function to use.  Default: exponential_decay.
    """

    def __init__(
        self,
        half_life: float,
        decay_fn: DecayFunction | None = None,
    ) -> None:
        if half_life <= 0:
            raise ValueError(
                f"half_life must be positive, got: {half_life}"
            )
        self.half_life = half_life
        self.decay_fn: DecayFunction = decay_fn if decay_fn is not None else exponential_decay
        self.last_result: TemporalResult | None = None

    def apply(
        self,
        opinions: list[Opinion],
        elapsed_times: list[float | None] | None = None,
        **kwargs,
    ) -> list[Opinion]:
        """Apply temporal decay to document opinions.

        Args:
            opinions: Per-document SL opinions.
            elapsed_times: Per-doc elapsed times.
                - None → passthrough for all docs (no timestamps).
                - List with None entries → those docs skip decay.
                - List with float entries → decay_opinion() applied.

        Returns:
            Decayed opinions (same length as input).

        Raises:
            ValueError: If elapsed_times length doesn't match opinions.
        """
        if elapsed_times is not None and len(elapsed_times) != len(opinions):
            raise ValueError(
                f"elapsed_times length ({len(elapsed_times)}) does not "
                f"match opinions length ({len(opinions)})."
            )

        # No timestamps → full passthrough
        if elapsed_times is None:
            elapsed_times_resolved: list[float | None] = [None] * len(opinions)
        else:
            elapsed_times_resolved = list(elapsed_times)

        decayed: list[Opinion] = []
        factors: list[float | None] = []

        for op, elapsed in zip(opinions, elapsed_times_resolved):
            if elapsed is None:
                # No timestamp for this doc — pass through
                decayed.append(op)
                factors.append(None)
            else:
                # Compute decay factor for traceability
                factor = self.decay_fn(elapsed, self.half_life)
                factors.append(factor)
                decayed.append(
                    decay_opinion(
                        op,
                        elapsed=elapsed,
                        half_life=self.half_life,
                        decay_fn=self.decay_fn,
                    )
                )

        self.last_result = TemporalResult(
            opinions_before=opinions,
            opinions_after=decayed,
            elapsed_times=elapsed_times_resolved,
            decay_factors=factors,
        )

        return decayed
