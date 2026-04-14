"""Retrieval confidence baselines for UQ in RAG.

Compute uncertainty from retriever scores of retrieved passages.
Cost: FREE (post-hoc, no extra compute).

Four variants:
    max_retrieval_score:   max(scores)                        — best-match
    mean_retrieval_score:  mean(scores)                       — average quality
    score_gap:             sorted_desc[0] - sorted_desc[1]    — separation
    score_entropy:         1 - H(softmax(scores))/log(K)      — concentration
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from xrag.baselines.base import UQScore, UQScorer


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def max_retrieval_score(scores: list[float]) -> float:
    """Maximum retrieval score across passages.

    Args:
        scores: Per-passage retrieval scores.

    Returns:
        Maximum score (float).

    Raises:
        ValueError: If list is empty.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    return float(max(scores))


def mean_retrieval_score(scores: list[float]) -> float:
    """Mean retrieval score across passages.

    Args:
        scores: Per-passage retrieval scores.

    Returns:
        Mean score (float).

    Raises:
        ValueError: If list is empty.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    return float(sum(scores) / len(scores))


def score_gap(scores: list[float]) -> float:
    """Gap between top-1 and top-2 retrieval scores.

    Measures how much the best result dominates the second-best.
    If only one score is provided, returns the score itself.

    Args:
        scores: Per-passage retrieval scores (need not be sorted).

    Returns:
        Score gap (float, non-negative).

    Raises:
        ValueError: If list is empty.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    if len(scores) == 1:
        return float(scores[0])
    sorted_desc = sorted(scores, reverse=True)
    return float(sorted_desc[0] - sorted_desc[1])


def score_entropy(scores: list[float]) -> float:
    """Confidence from entropy over softmax-normalized retrieval scores.

    Applies softmax to scores, computes Shannon entropy, and returns
    confidence = 1 - H/log(K), where K is the number of scores.

    Single score → entropy = 0 → confidence = 1.0.
    Uniform scores → max entropy → confidence = 0.0.

    Args:
        scores: Per-passage retrieval scores.

    Returns:
        Confidence ∈ [0, 1]. Higher = more concentrated.

    Raises:
        ValueError: If list is empty.
    """
    if not scores:
        raise ValueError("scores must be non-empty")

    k = len(scores)
    if k == 1:
        return 1.0

    # Softmax with numerical stability
    arr = np.array(scores, dtype=np.float64)
    arr = arr - np.max(arr)  # shift for stability
    exp_arr = np.exp(arr)
    probs = exp_arr / np.sum(exp_arr)

    # Shannon entropy
    # Avoid log(0) by filtering zeros
    nonzero = probs[probs > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))

    # Normalize to [0, 1] and invert (high entropy → low confidence)
    max_entropy = math.log(k)
    normalized_entropy = entropy / max_entropy
    confidence = 1.0 - normalized_entropy

    return float(np.clip(confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

_METHODS = {
    "max_retrieval_score": max_retrieval_score,
    "mean_retrieval_score": mean_retrieval_score,
    "score_gap": score_gap,
    "score_entropy": score_entropy,
}


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class RetrievalConfidenceScorer(UQScorer):
    """Confidence scorer using retrieval scores.

    Args:
        method: One of "max_retrieval_score", "mean_retrieval_score",
            "score_gap", "score_entropy".

    Raises:
        ValueError: If method is not recognized.
    """

    def __init__(self, method: str = "max_retrieval_score") -> None:
        if method not in _METHODS:
            raise ValueError(
                f"Method must be one of {list(_METHODS.keys())}, got '{method}'"
            )
        self._method = method
        self._fn = _METHODS[method]

    @property
    def name(self) -> str:
        return f"retrieval_{self._method}"

    @property
    def cost_category(self) -> str:
        return "free"

    def score(self, *, retrieval_scores: list[float] | None = None, **kwargs: Any) -> UQScore:
        """Compute confidence from retrieval scores.

        Args:
            retrieval_scores: Per-passage retrieval scores.

        Returns:
            UQScore with confidence value.

        Raises:
            ValueError: If retrieval_scores is None or empty.
        """
        if retrieval_scores is None:
            raise ValueError("retrieval_scores is required")

        raw_value = self._fn(retrieval_scores)

        return UQScore(
            confidence=float(raw_value),
            method=self.name,
            metadata={
                "raw_value": raw_value,
                "n_passages": len(retrieval_scores),
            },
        )

    def _score_single(self, retrieval_scores: list[float]) -> UQScore:
        """Score a single item for batch processing."""
        return self.score(retrieval_scores=retrieval_scores)

    def score_batch(self, inputs_list: list[list[float]]) -> np.ndarray:
        """Score a batch of retrieval score lists.

        Args:
            inputs_list: List of per-query retrieval score lists.

        Returns:
            1D numpy array of confidence values.
        """
        return np.array([
            self._score_single(scores).confidence for scores in inputs_list
        ])
