"""Tests for baselines/retrieval_confidence.py.

Retrieval confidence baselines compute uncertainty from retriever scores.
Cost: FREE (post-hoc from retrieval results).

Four variants:
    1. max_retrieval_score:   max(scores)                — best-match
    2. mean_retrieval_score:  mean(scores)               — average quality
    3. score_gap:             score[0] - score[1]        — separation
    4. score_entropy:         -Σ p_i·log(p_i) on softmax — concentration

Plus RetrievalConfidenceScorer wrapper.
"""

import math

import numpy as np
import pytest

from xrag.baselines.base import UQScore, UQScorer
from xrag.baselines.retrieval_confidence import (
    max_retrieval_score,
    mean_retrieval_score,
    score_gap,
    score_entropy,
    RetrievalConfidenceScorer,
)


# =============================================================================
# max_retrieval_score
# =============================================================================


class TestMaxRetrievalScore:
    """max_retrieval_score = max(scores). Higher = more confident."""

    def test_basic(self):
        assert max_retrieval_score([0.9, 0.7, 0.3]) == pytest.approx(0.9)

    def test_single_score(self):
        assert max_retrieval_score([0.5]) == pytest.approx(0.5)

    def test_all_equal(self):
        assert max_retrieval_score([0.6, 0.6, 0.6]) == pytest.approx(0.6)

    def test_max_at_end(self):
        assert max_retrieval_score([0.1, 0.2, 0.95]) == pytest.approx(0.95)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            max_retrieval_score([])

    def test_returns_float(self):
        assert isinstance(max_retrieval_score([0.5]), float)


# =============================================================================
# mean_retrieval_score
# =============================================================================


class TestMeanRetrievalScore:
    """mean_retrieval_score = mean(scores). Higher = more confident."""

    def test_basic(self):
        assert mean_retrieval_score([0.9, 0.6, 0.3]) == pytest.approx(0.6)

    def test_single_score(self):
        assert mean_retrieval_score([0.5]) == pytest.approx(0.5)

    def test_all_equal(self):
        assert mean_retrieval_score([0.4, 0.4, 0.4]) == pytest.approx(0.4)

    def test_leq_max(self):
        scores = [0.9, 0.3, 0.5]
        assert mean_retrieval_score(scores) <= max_retrieval_score(scores) + 1e-12

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean_retrieval_score([])

    def test_returns_float(self):
        assert isinstance(mean_retrieval_score([0.5]), float)


# =============================================================================
# score_gap
# =============================================================================


class TestScoreGap:
    """score_gap = sorted_desc[0] - sorted_desc[1]. Higher = more separation."""

    def test_basic(self):
        """
        scores = [0.9, 0.7, 0.3]  → sorted desc: [0.9, 0.7, 0.3]
        gap = 0.9 - 0.7 = 0.2
        """
        assert score_gap([0.9, 0.7, 0.3]) == pytest.approx(0.2)

    def test_unsorted_input(self):
        """Scores may not be pre-sorted."""
        assert score_gap([0.3, 0.9, 0.7]) == pytest.approx(0.2)

    def test_no_gap(self):
        assert score_gap([0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_large_gap(self):
        assert score_gap([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_two_scores(self):
        assert score_gap([0.8, 0.3]) == pytest.approx(0.5)

    def test_single_score(self):
        """Only one passage → gap is meaningless. Return the score itself
        or 0 — either convention is acceptable."""
        result = score_gap([0.5])
        assert isinstance(result, float)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            score_gap([])

    def test_nonnegative(self):
        """Gap should always be >= 0."""
        assert score_gap([0.3, 0.9, 0.7]) >= 0.0

    def test_returns_float(self):
        assert isinstance(score_gap([0.5, 0.3]), float)


# =============================================================================
# score_entropy
# =============================================================================


class TestScoreEntropy:
    """score_entropy: entropy over softmax-normalized retrieval scores.
    
    Lower entropy = more concentrated (confident). We return confidence
    as 1 - normalized_entropy so higher = more confident.
    """

    def test_uniform_scores_high_entropy(self):
        """All equal scores → maximum entropy → low confidence."""
        scores = [1.0, 1.0, 1.0, 1.0]
        result = score_entropy(scores)
        # Uniform over 4 → max entropy = log(4)
        # Confidence should be low
        assert result < 0.5

    def test_concentrated_scores_low_entropy(self):
        """One dominant score → low entropy → high confidence."""
        scores = [100.0, 0.1, 0.1, 0.1]
        result = score_entropy(scores)
        assert result > 0.9

    def test_two_scores_equal(self):
        """Two equal scores → entropy = log(2) → moderate confidence."""
        scores = [1.0, 1.0]
        result = score_entropy(scores)
        # Normalized entropy = 1.0 → confidence = 0.0
        assert result == pytest.approx(0.0, abs=0.01)

    def test_single_score(self):
        """Single score → entropy = 0 → max confidence."""
        result = score_entropy([0.5])
        assert result == pytest.approx(1.0)

    def test_bounded_0_1(self):
        """Confidence should be in [0, 1]."""
        for scores in [[0.5, 0.3, 0.1], [1.0, 1.0], [10.0, 0.01]]:
            result = score_entropy(scores)
            assert 0.0 <= result <= 1.0 + 1e-12, f"Out of bounds for {scores}"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            score_entropy([])

    def test_returns_float(self):
        assert isinstance(score_entropy([0.5, 0.3]), float)

    def test_higher_concentration_higher_confidence(self):
        """More concentrated → higher confidence."""
        concentrated = [10.0, 0.1, 0.1]
        spread = [1.0, 1.0, 1.0]
        assert score_entropy(concentrated) > score_entropy(spread)


# =============================================================================
# RetrievalConfidenceScorer — wrapper class
# =============================================================================


class TestRetrievalConfidenceScorer:
    """High-level scorer wrapping retrieval confidence functions."""

    def test_is_uq_scorer(self):
        scorer = RetrievalConfidenceScorer(method="max_retrieval_score")
        assert isinstance(scorer, UQScorer)

    def test_score_returns_uq_score(self):
        scorer = RetrievalConfidenceScorer(method="max_retrieval_score")
        result = scorer.score(retrieval_scores=[0.9, 0.7, 0.3])
        assert isinstance(result, UQScore)

    def test_method_max(self):
        scorer = RetrievalConfidenceScorer(method="max_retrieval_score")
        result = scorer.score(retrieval_scores=[0.9, 0.7, 0.3])
        assert result.confidence == pytest.approx(0.9)

    def test_method_mean(self):
        scorer = RetrievalConfidenceScorer(method="mean_retrieval_score")
        result = scorer.score(retrieval_scores=[0.9, 0.6, 0.3])
        assert result.confidence == pytest.approx(0.6)

    def test_method_gap(self):
        scorer = RetrievalConfidenceScorer(method="score_gap")
        result = scorer.score(retrieval_scores=[0.9, 0.7, 0.3])
        assert result.confidence == pytest.approx(0.2)

    def test_method_entropy(self):
        scorer = RetrievalConfidenceScorer(method="score_entropy")
        result = scorer.score(retrieval_scores=[100.0, 0.1, 0.1])
        assert result.confidence > 0.8

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="[Mm]ethod"):
            RetrievalConfidenceScorer(method="invalid")

    def test_name_property(self):
        scorer = RetrievalConfidenceScorer(method="max_retrieval_score")
        assert "retrieval" in scorer.name.lower()

    def test_cost_category(self):
        scorer = RetrievalConfidenceScorer(method="max_retrieval_score")
        assert scorer.cost_category == "free"

    def test_none_scores_raises(self):
        scorer = RetrievalConfidenceScorer(method="max_retrieval_score")
        with pytest.raises((ValueError, TypeError)):
            scorer.score(retrieval_scores=None)

    def test_score_batch(self):
        scorer = RetrievalConfidenceScorer(method="max_retrieval_score")
        batch = [[0.9, 0.3], [0.5, 0.4], [0.2, 0.1]]
        results = scorer.score_batch(batch)
        assert isinstance(results, np.ndarray)
        assert len(results) == 3
        assert results[0] == pytest.approx(0.9)
        assert results[1] == pytest.approx(0.5)
        assert results[2] == pytest.approx(0.2)

    def test_metadata_present(self):
        scorer = RetrievalConfidenceScorer(method="max_retrieval_score")
        result = scorer.score(retrieval_scores=[0.9, 0.3])
        assert "raw_value" in result.metadata
