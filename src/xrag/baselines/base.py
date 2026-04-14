"""Base classes for UQ baseline scorers.

All baselines produce a UQScore and implement the UQScorer interface,
enabling uniform comparison with SL-RAG's opinion-based confidence.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class UQScore:
    """Result from a UQ baseline scorer.

    Attributes:
        confidence: Scalar confidence in [0, 1]. Higher = more confident.
        method: Identifier for the scoring method used.
        metadata: Method-specific details (raw values, intermediates, etc.).
    """

    confidence: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


class UQScorer(abc.ABC):
    """Abstract base for uncertainty quantification scorers.

    All baselines implement this interface so they can be swapped
    interchangeably in experiment harnesses.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name for this scorer."""
        ...

    @property
    @abc.abstractmethod
    def cost_category(self) -> str:
        """Computational cost category.

        One of:
            "free"       — post-hoc from single generation (no extra compute)
            "n_samples"  — requires N forward passes (e.g. semantic entropy)
            "extra_call" — requires 1 additional LLM call (e.g. P(True))
        """
        ...

    @abc.abstractmethod
    def score(self, **inputs: Any) -> UQScore:
        """Compute a confidence score from the given inputs.

        Subclasses define which keyword arguments they require.
        """
        ...

    def score_batch(self, inputs_list: list[Any]) -> np.ndarray:
        """Score a batch, returning an array of confidence values.

        Default implementation loops over inputs. Subclasses may override
        for vectorized computation.

        Args:
            inputs_list: List of inputs (type depends on subclass).

        Returns:
            1D numpy array of confidence scores.
        """
        return np.array([self.score(**inp).confidence if isinstance(inp, dict)
                         else self._score_single(inp).confidence
                         for inp in inputs_list])

    def _score_single(self, inp: Any) -> UQScore:
        """Override point for subclasses with non-dict batch inputs."""
        raise NotImplementedError
