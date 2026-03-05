"""Abstract base class for opinion estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from jsonld_ex.confidence_algebra import Opinion


@dataclass(frozen=True)
class EstimationResult:
    """Result of opinion estimation for a single (query, document) pair.

    Attributes:
        opinion: The SL opinion representing evidential support.
        nli_scores: Raw NLI scores if available (entailment, neutral, contradiction).
        metadata: Optional dict for estimator-specific info (e.g., model name, latency).
    """

    opinion: Opinion
    nli_scores: tuple[float, float, float] | None = None  # (entail, neutral, contradict)
    metadata: dict | None = None


class BaseOpinionEstimator(ABC):
    """Abstract base class for opinion estimators.

    The primary interface is `estimate_batch()`, which processes
    multiple (query, document) pairs efficiently (e.g., on GPU).

    `estimate()` is a convenience method for single pairs that
    delegates to `estimate_batch()`.
    """

    @abstractmethod
    def estimate_batch(
        self,
        queries: Sequence[str],
        documents: Sequence[str],
    ) -> list[EstimationResult]:
        """Estimate opinions for a batch of (query, document) pairs.

        Implementations should optimize for batch processing
        (e.g., GPU batching for NLI models).

        Args:
            queries: List of query strings.
            documents: List of document strings. Must be same length as queries.

        Returns:
            List of EstimationResult, one per (query, document) pair.

        Raises:
            ValueError: If queries and documents have different lengths.
        """
        ...

    @abstractmethod
    def estimate(self, query: str, document: str) -> EstimationResult:
        """Estimate opinion for a single (query, document) pair.

        Implementations may optimize this independently from estimate_batch()
        (e.g., avoiding batch overhead for single pairs).

        Args:
            query: The query string.
            document: The document string.

        Returns:
            EstimationResult for the pair.
        """
        ...
