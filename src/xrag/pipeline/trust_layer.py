"""Trust discount layer — applies source reliability priors.

Uses SL trust_discount() to weight opinions by source trustworthiness.
"""

from __future__ import annotations

import abc

from jsonld_ex.confidence_algebra import Opinion


class TrustLayer(abc.ABC):
    """Abstract base for trust discount layers."""

    @abc.abstractmethod
    def apply(self, opinions: list[Opinion]) -> list[Opinion]:
        """Apply trust discounting to document opinions.

        Args:
            opinions: Per-document SL opinions.

        Returns:
            Trust-discounted opinions (same length).
        """
        ...


class NoOpTrustLayer(TrustLayer):
    """No-op trust layer — passes opinions through unchanged."""

    def apply(self, opinions: list[Opinion]) -> list[Opinion]:
        return list(opinions)
