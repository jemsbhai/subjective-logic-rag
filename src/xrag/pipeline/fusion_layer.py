"""Evidence fusion layer — combines multiple document opinions into one.

Uses SL cumulative_fuse(), averaging_fuse(), robust_fuse(), or byzantine_fuse().
"""

from __future__ import annotations

import abc

from jsonld_ex.confidence_algebra import Opinion


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


class NoOpFusionLayer(FusionLayer):
    """No-op fusion layer — returns a vacuous (max uncertainty) opinion."""

    def fuse(self, opinions: list[Opinion]) -> Opinion:
        return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
