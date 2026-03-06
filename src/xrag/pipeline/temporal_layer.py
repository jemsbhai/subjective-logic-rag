"""Temporal decay layer — ages opinions based on document timestamps.

Uses SL decay_opinion() to reduce belief in stale evidence.
"""

from __future__ import annotations

import abc

from jsonld_ex.confidence_algebra import Opinion


class TemporalLayer(abc.ABC):
    """Abstract base for temporal decay layers."""

    @abc.abstractmethod
    def apply(self, opinions: list[Opinion]) -> list[Opinion]:
        """Apply temporal decay to document opinions.

        Args:
            opinions: Per-document SL opinions.

        Returns:
            Temporally decayed opinions (same length).
        """
        ...


class NoOpTemporalLayer(TemporalLayer):
    """No-op temporal layer — passes opinions through unchanged."""

    def apply(self, opinions: list[Opinion]) -> list[Opinion]:
        return list(opinions)
