"""Trust discount layer — applies source reliability priors.

Uses SL trust_discount() from jsonld-ex to weight document opinions
by source trustworthiness.

Design:
    - SLTrustLayer takes a default_trust Opinion applied uniformly.
    - Per-doc trust overrides via the trust_opinions kwarg to apply().
    - Full traceability via self.last_result (TrustResult dataclass).

Paper 1 usage:
    - Main experiments: default_trust = high (Wikipedia sources)
    - Corruption suites: per-doc trusts (injected docs get lower trust)
    - Ablation A5: NoOpTrustLayer (skip discounting entirely)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from jsonld_ex.confidence_algebra import Opinion, trust_discount


# ═══════════════════════════════════════════════════════════════════
# TrustResult — full-fidelity metadata from a trust discount pass
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrustResult:
    """Complete record of a trust discounting pass.

    Attributes:
        opinions_before:    The original document opinions (input).
        opinions_after:     The trust-discounted opinions (output).
        trust_opinions_used: The trust opinion applied to each document
                             (for full traceability in paper analysis).
    """

    opinions_before: list[Opinion]
    opinions_after: list[Opinion]
    trust_opinions_used: list[Opinion]


# ═══════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════


class TrustLayer(abc.ABC):
    """Abstract base for trust discount layers."""

    @abc.abstractmethod
    def apply(self, opinions: list[Opinion], **kwargs) -> list[Opinion]:
        """Apply trust discounting to document opinions.

        Args:
            opinions: Per-document SL opinions.

        Returns:
            Trust-discounted opinions (same length).
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# NoOp (unchanged — backward compatible)
# ═══════════════════════════════════════════════════════════════════


class NoOpTrustLayer(TrustLayer):
    """No-op trust layer — passes opinions through unchanged."""

    def apply(self, opinions: list[Opinion], **kwargs) -> list[Opinion]:
        return list(opinions)


# ═══════════════════════════════════════════════════════════════════
# Real implementation
# ═══════════════════════════════════════════════════════════════════


class SLTrustLayer(TrustLayer):
    """Trust discount layer using Subjective Logic trust_discount().

    Applies trust discounting to each document opinion individually.
    The trust opinion controls how much of the original evidence is
    retained:

        b_discounted = b_trust * b_original
        d_discounted = b_trust * d_original
        u_discounted = d_trust + u_trust + b_trust * u_original

    Full trust (b=1, d=0, u=0) is the identity — opinion unchanged.
    Vacuous trust (b=0, d=0, u=1) yields a vacuous opinion.
    Partial trust dilutes evidence toward uncertainty.

    Args:
        default_trust: Trust opinion applied uniformly to all documents
                       when no per-doc overrides are given.
    """

    def __init__(self, default_trust: Opinion) -> None:
        self.default_trust = default_trust
        self.last_result: TrustResult | None = None

    def apply(
        self,
        opinions: list[Opinion],
        trust_opinions: list[Opinion] | None = None,
        **kwargs,
    ) -> list[Opinion]:
        """Apply trust discounting to document opinions.

        Args:
            opinions: Per-document SL opinions.
            trust_opinions: Optional per-document trust opinions.
                            If None, default_trust is used for all.
                            If provided, must have same length as opinions.

        Returns:
            Trust-discounted opinions (same length as input).

        Raises:
            ValueError: If trust_opinions length doesn't match opinions.
        """
        if trust_opinions is not None and len(trust_opinions) != len(opinions):
            raise ValueError(
                f"trust_opinions length ({len(trust_opinions)}) does not "
                f"match opinions length ({len(opinions)})."
            )

        # Resolve trust opinion for each document
        trusts_used: list[Opinion]
        if trust_opinions is not None:
            trusts_used = list(trust_opinions)
        else:
            trusts_used = [self.default_trust] * len(opinions)

        # Apply trust_discount to each (trust, doc) pair
        discounted = [
            trust_discount(t, op) for t, op in zip(trusts_used, opinions)
        ]

        self.last_result = TrustResult(
            opinions_before=opinions,
            opinions_after=discounted,
            trust_opinions_used=trusts_used,
        )

        return discounted
