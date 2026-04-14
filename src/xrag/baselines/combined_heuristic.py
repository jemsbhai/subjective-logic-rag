"""Combined heuristic baseline for UQ in RAG.

Scalar combination of retrieval and generation confidence.
Cost: FREE (combines two free post-hoc signals).

This is the "obvious baseline" a reviewer would propose:
"Why not just multiply retrieval score by softmax confidence?"

Three strategies:
    product:   conf = retrieval_conf × generation_conf
    mean:      conf = (retrieval_conf + generation_conf) / 2
    weighted:  conf = w·retrieval_conf + (1-w)·generation_conf
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xrag.baselines.base import UQScore, UQScorer


_STRATEGIES = {"product", "mean", "weighted"}


class CombinedHeuristicScorer(UQScorer):
    """Combines retrieval and generation confidence into a single scalar.

    Args:
        strategy: One of "product", "mean", "weighted".
        weight: Weight for retrieval confidence when strategy="weighted".
            Must be in [0, 1]. Default 0.5 (equivalent to "mean").

    Raises:
        ValueError: If strategy or weight is invalid.
    """

    def __init__(
        self,
        strategy: str = "product",
        weight: float = 0.5,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Strategy must be one of {sorted(_STRATEGIES)}, got '{strategy}'"
            )
        if strategy == "weighted" and not (0.0 <= weight <= 1.0):
            raise ValueError(
                f"Weight must be in [0, 1], got {weight}"
            )
        self._strategy = strategy
        self._weight = weight

    @property
    def name(self) -> str:
        if self._strategy == "weighted":
            return f"combined_{self._strategy}_w{self._weight:.2f}"
        return f"combined_{self._strategy}"

    @property
    def cost_category(self) -> str:
        return "free"

    def score(
        self,
        *,
        retrieval_confidence: float | None = None,
        generation_confidence: float | None = None,
        **kwargs: Any,
    ) -> UQScore:
        """Compute combined confidence from retrieval and generation scores.

        Args:
            retrieval_confidence: Retrieval confidence ∈ [0, 1].
            generation_confidence: Generation confidence ∈ [0, 1].

        Returns:
            UQScore with combined confidence.

        Raises:
            ValueError: If either input is None.
        """
        if retrieval_confidence is None:
            raise ValueError("retrieval_confidence is required")
        if generation_confidence is None:
            raise ValueError("generation_confidence is required")

        rc = float(retrieval_confidence)
        gc = float(generation_confidence)

        if self._strategy == "product":
            confidence = rc * gc
        elif self._strategy == "mean":
            confidence = (rc + gc) / 2.0
        else:  # weighted
            confidence = self._weight * rc + (1.0 - self._weight) * gc

        return UQScore(
            confidence=float(confidence),
            method=self.name,
            metadata={
                "retrieval_confidence": rc,
                "generation_confidence": gc,
                "strategy": self._strategy,
                "weight": self._weight if self._strategy == "weighted" else None,
            },
        )

    def score_batch(self, inputs_list: list[dict[str, float]]) -> np.ndarray:
        """Score a batch of retrieval/generation confidence pairs.

        Args:
            inputs_list: List of dicts with "retrieval_confidence" and
                "generation_confidence" keys.

        Returns:
            1D numpy array of combined confidence values.
        """
        return np.array([
            self.score(**inp).confidence for inp in inputs_list
        ])
