"""Softmax confidence baselines for UQ in RAG.

Compute uncertainty from token-level log probabilities of a single
greedy generation.  Cost: FREE (post-hoc, no extra compute).

Four variants:
    mean_logprob:         (1/T) Σ log p(t_i)           — raw average
    normalized_seq_prob:  exp((1/T) Σ log p(t_i))       — geometric mean ∈ [0,1]
    perplexity_to_conf:   1 / exp(-(1/T) Σ log p(t_i)) — inverse perplexity ∈ (0,1]
    min_token_prob:       exp(min_i log p(t_i))          — worst-token ∈ [0,1]

Note: perplexity_to_conf ≡ normalized_seq_prob algebraically, but we
keep both for clarity in the paper (reviewers expect to see "perplexity"
and "sequence probability" as separate baselines even though they yield
identical rankings).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from xrag.baselines.base import UQScore, UQScorer


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def mean_logprob(token_logprobs: list[float]) -> float:
    """Mean log probability across generated tokens.

    Higher (less negative) = more confident.

    Args:
        token_logprobs: Per-token log probabilities.

    Returns:
        Mean log probability (float, ≤ 0).

    Raises:
        ValueError: If list is empty.
    """
    if not token_logprobs:
        raise ValueError("token_logprobs must be non-empty")
    return float(sum(token_logprobs) / len(token_logprobs))


def normalized_seq_prob(token_logprobs: list[float]) -> float:
    """Normalized sequence probability: geometric mean of token probabilities.

    exp((1/T) Σ log p(t_i)) = (Π p(t_i))^(1/T)

    Args:
        token_logprobs: Per-token log probabilities.

    Returns:
        Geometric mean probability ∈ (0, 1].

    Raises:
        ValueError: If list is empty.
    """
    if not token_logprobs:
        raise ValueError("token_logprobs must be non-empty")
    return float(math.exp(sum(token_logprobs) / len(token_logprobs)))


def perplexity_to_conf(token_logprobs: list[float]) -> float:
    """Inverse perplexity as confidence.

    confidence = 1 / PPL = 1 / exp(-(1/T) Σ log p(t_i))
               = exp((1/T) Σ log p(t_i))

    Algebraically identical to normalized_seq_prob, but provided
    as a separate named function for paper clarity.

    Args:
        token_logprobs: Per-token log probabilities.

    Returns:
        Inverse perplexity ∈ (0, 1].

    Raises:
        ValueError: If list is empty.
    """
    if not token_logprobs:
        raise ValueError("token_logprobs must be non-empty")
    return float(math.exp(sum(token_logprobs) / len(token_logprobs)))


def min_token_prob(token_logprobs: list[float]) -> float:
    """Minimum token probability (worst-token confidence).

    exp(min_i log p(t_i))

    Captures the "weakest link" — a single uncertain token
    drives the score down.

    Args:
        token_logprobs: Per-token log probabilities.

    Returns:
        Minimum token probability ∈ (0, 1].

    Raises:
        ValueError: If list is empty.
    """
    if not token_logprobs:
        raise ValueError("token_logprobs must be non-empty")
    return float(math.exp(min(token_logprobs)))


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

_METHODS = {
    "mean_logprob": mean_logprob,
    "normalized_seq_prob": normalized_seq_prob,
    "perplexity": perplexity_to_conf,
    "min_token_prob": min_token_prob,
}

# Methods that return values in [0, 1] directly
_BOUNDED_METHODS = {"normalized_seq_prob", "perplexity", "min_token_prob"}


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class SoftmaxConfidenceScorer(UQScorer):
    """Confidence scorer using token-level softmax log probabilities.

    Wraps the pure functions above into the UQScorer interface.

    Args:
        method: One of "mean_logprob", "normalized_seq_prob",
            "perplexity", "min_token_prob".

    Raises:
        ValueError: If method is not recognized.
    """

    def __init__(self, method: str = "normalized_seq_prob") -> None:
        if method not in _METHODS:
            raise ValueError(
                f"Method must be one of {list(_METHODS.keys())}, got '{method}'"
            )
        self._method = method
        self._fn = _METHODS[method]

    @property
    def name(self) -> str:
        return f"softmax_{self._method}"

    @property
    def cost_category(self) -> str:
        return "free"

    def score(self, *, token_logprobs: list[float] | None = None, **kwargs: Any) -> UQScore:
        """Compute confidence from token log probabilities.

        Args:
            token_logprobs: Per-token log probabilities from GenerationResult.

        Returns:
            UQScore with confidence ∈ [0, 1].

        Raises:
            ValueError: If token_logprobs is None or empty.
        """
        if token_logprobs is None:
            raise ValueError(
                "token_logprobs is required (pass return_logprobs=True to generator)"
            )

        raw_value = self._fn(token_logprobs)

        # Map to [0, 1] confidence
        if self._method in _BOUNDED_METHODS:
            confidence = raw_value
        else:
            # mean_logprob is ≤ 0; map via exp to [0, 1]
            confidence = math.exp(raw_value)

        ml = mean_logprob(token_logprobs)

        return UQScore(
            confidence=float(confidence),
            method=self.name,
            metadata={
                "raw_value": raw_value,
                "mean_logprob": ml,
                "n_tokens": len(token_logprobs),
            },
        )

    def _score_single(self, token_logprobs: list[float]) -> UQScore:
        """Score a single item for batch processing."""
        return self.score(token_logprobs=token_logprobs)

    def score_batch(self, inputs_list: list[list[float]]) -> np.ndarray:
        """Score a batch of token log probability sequences.

        Args:
            inputs_list: List of per-example token_logprobs lists.

        Returns:
            1D numpy array of confidence values.
        """
        return np.array([
            self._score_single(lps).confidence for lps in inputs_list
        ])
