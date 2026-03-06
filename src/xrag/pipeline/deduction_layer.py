"""Multi-hop deduction layer — chains reasoning across hops.

Uses SL deduce() to propagate uncertainty across multi-hop inference.
"""

from __future__ import annotations

import abc
from typing import Optional

from jsonld_ex.confidence_algebra import Opinion


class DeductionLayer(abc.ABC):
    """Abstract base for multi-hop deduction layers."""

    @abc.abstractmethod
    def deduce(
        self,
        fused_opinion: Opinion,
        sub_questions: Optional[list[str]] = None,
    ) -> Opinion:
        """Apply multi-hop deduction to the fused opinion.

        For single-hop questions, this is typically a passthrough.
        For multi-hop, it chains per-hop opinions via deduce().

        Args:
            fused_opinion: The fused opinion from the fusion layer.
            sub_questions: Optional list of sub-question strings
                for multi-hop decomposition.

        Returns:
            Final opinion after deduction (may carry more uncertainty
            than the input if hops are uncertain).
        """
        ...


class NoOpDeductionLayer(DeductionLayer):
    """No-op deduction layer — passes the fused opinion through unchanged."""

    def deduce(
        self,
        fused_opinion: Opinion,
        sub_questions: Optional[list[str]] = None,
    ) -> Opinion:
        return fused_opinion
