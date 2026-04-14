"""Tests for baselines/base.py and baselines/softmax_confidence.py.

Softmax confidence baselines compute uncertainty from token-level log
probabilities of a single greedy generation. Cost: FREE (post-hoc).

Four variants tested, each with hand-computed expected values:
    1. mean_logprob:           (1/T) Σ log p(t_i)
    2. normalized_seq_prob:    exp((1/T) Σ log p(t_i))  ∈ [0,1]
    3. perplexity_to_conf:     1 / exp(-(1/T) Σ log p(t_i))  ∈ (0,1]
    4. min_token_prob:          exp(min_i log p(t_i))  ∈ [0,1]
"""

import math

import numpy as np
import pytest

from xrag.baselines.base import UQScore, UQScorer
from xrag.baselines.softmax_confidence import (
    mean_logprob,
    normalized_seq_prob,
    perplexity_to_conf,
    min_token_prob,
    SoftmaxConfidenceScorer,
)


# =============================================================================
# UQScore dataclass
# =============================================================================


class TestUQScore:
    """UQScore: common result type for all baselines."""

    def test_construction(self):
        score = UQScore(confidence=0.8, method="test", metadata={})
        assert score.confidence == 0.8
        assert score.method == "test"

    def test_metadata(self):
        score = UQScore(confidence=0.5, method="m", metadata={"key": "val"})
        assert score.metadata["key"] == "val"


# =============================================================================
# mean_logprob
# =============================================================================


class TestMeanLogprob:
    """mean_logprob = (1/T) Σ log p(t_i). Higher = more confident."""

    def test_hand_computed(self):
        """
        logprobs = [log(0.9), log(0.8), log(0.7)]
                 ≈ [-0.10536, -0.22314, -0.35667]
        mean = (-0.10536 + -0.22314 + -0.35667) / 3 ≈ -0.22839
        """
        lps = [math.log(0.9), math.log(0.8), math.log(0.7)]
        result = mean_logprob(lps)
        expected = sum(lps) / 3
        assert result == pytest.approx(expected)

    def test_perfect_confidence(self):
        """All tokens have probability 1.0 → mean logprob = 0."""
        lps = [0.0, 0.0, 0.0]  # log(1) = 0
        assert mean_logprob(lps) == pytest.approx(0.0)

    def test_single_token(self):
        lp = math.log(0.5)
        assert mean_logprob([lp]) == pytest.approx(lp)

    def test_always_negative_or_zero(self):
        """Log probs are <= 0, so mean is <= 0."""
        lps = [math.log(0.3), math.log(0.6), math.log(0.9)]
        assert mean_logprob(lps) <= 0.0 + 1e-12

    def test_empty_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            mean_logprob([])

    def test_returns_float(self):
        assert isinstance(mean_logprob([math.log(0.5)]), float)

    def test_more_confident_higher_value(self):
        """Higher token probs → higher mean logprob."""
        high = [math.log(0.95), math.log(0.90)]
        low = [math.log(0.30), math.log(0.20)]
        assert mean_logprob(high) > mean_logprob(low)


# =============================================================================
# normalized_seq_prob
# =============================================================================


class TestNormalizedSeqProb:
    """normalized_seq_prob = exp(mean_logprob). Geometric mean of probs. ∈ [0,1]."""

    def test_hand_computed(self):
        """
        logprobs = [log(0.9), log(0.8)]
        mean = (log(0.9) + log(0.8)) / 2 = log(0.72) / 2
        nsp = exp(mean) = (0.9 * 0.8)^(1/2) = sqrt(0.72) ≈ 0.8485
        """
        lps = [math.log(0.9), math.log(0.8)]
        result = normalized_seq_prob(lps)
        expected = math.sqrt(0.9 * 0.8)
        assert result == pytest.approx(expected)

    def test_perfect_confidence(self):
        lps = [0.0, 0.0, 0.0]
        assert normalized_seq_prob(lps) == pytest.approx(1.0)

    def test_bounded_0_1(self):
        """Result should always be in (0, 1]."""
        lps = [math.log(0.1), math.log(0.2), math.log(0.3)]
        result = normalized_seq_prob(lps)
        assert 0.0 < result <= 1.0

    def test_single_token(self):
        """Single token: nsp = exp(log(p)) = p."""
        p = 0.75
        assert normalized_seq_prob([math.log(p)]) == pytest.approx(p)

    def test_empty_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            normalized_seq_prob([])

    def test_returns_float(self):
        assert isinstance(normalized_seq_prob([math.log(0.5)]), float)

    def test_monotonic_with_mean_logprob(self):
        """normalized_seq_prob is monotonic with mean_logprob (exp is monotonic)."""
        high = [math.log(0.95), math.log(0.90)]
        low = [math.log(0.30), math.log(0.20)]
        assert normalized_seq_prob(high) > normalized_seq_prob(low)


# =============================================================================
# perplexity_to_conf
# =============================================================================


class TestPerplexityToConf:
    """perplexity_to_conf = 1 / perplexity = 1 / exp(-mean_logprob). ∈ (0,1]."""

    def test_hand_computed(self):
        """
        logprobs = [log(0.5), log(0.5)]
        mean_lp = log(0.5) ≈ -0.6931
        perplexity = exp(0.6931) = 2.0
        confidence = 1/2 = 0.5
        """
        lps = [math.log(0.5), math.log(0.5)]
        result = perplexity_to_conf(lps)
        assert result == pytest.approx(0.5)

    def test_perfect_confidence(self):
        """Perplexity = 1 → confidence = 1."""
        lps = [0.0, 0.0]
        assert perplexity_to_conf(lps) == pytest.approx(1.0)

    def test_high_perplexity_low_confidence(self):
        """Very uncertain tokens → low confidence."""
        lps = [math.log(0.01), math.log(0.01)]
        result = perplexity_to_conf(lps)
        assert result < 0.02  # perplexity ≈ 100, conf ≈ 0.01

    def test_bounded_0_1(self):
        lps = [math.log(0.3), math.log(0.4)]
        result = perplexity_to_conf(lps)
        assert 0.0 < result <= 1.0

    def test_equals_normalized_seq_prob(self):
        """perplexity_to_conf = 1/PPL = exp(mean_logprob) = normalized_seq_prob."""
        lps = [math.log(0.6), math.log(0.7), math.log(0.8)]
        assert perplexity_to_conf(lps) == pytest.approx(normalized_seq_prob(lps))

    def test_empty_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            perplexity_to_conf([])

    def test_returns_float(self):
        assert isinstance(perplexity_to_conf([math.log(0.5)]), float)


# =============================================================================
# min_token_prob
# =============================================================================


class TestMinTokenProb:
    """min_token_prob = exp(min log_prob). Worst-token confidence. ∈ [0,1]."""

    def test_hand_computed(self):
        """
        logprobs = [log(0.9), log(0.3), log(0.7)]
        min = log(0.3) ≈ -1.2040
        result = exp(log(0.3)) = 0.3
        """
        lps = [math.log(0.9), math.log(0.3), math.log(0.7)]
        result = min_token_prob(lps)
        assert result == pytest.approx(0.3)

    def test_perfect_confidence(self):
        lps = [0.0, 0.0, 0.0]
        assert min_token_prob(lps) == pytest.approx(1.0)

    def test_single_bad_token_dominates(self):
        """One very low-probability token should drive the score down."""
        lps = [math.log(0.99), math.log(0.99), math.log(0.01)]
        result = min_token_prob(lps)
        assert result == pytest.approx(0.01)

    def test_bounded_0_1(self):
        lps = [math.log(0.5), math.log(0.8)]
        result = min_token_prob(lps)
        assert 0.0 < result <= 1.0

    def test_leq_normalized_seq_prob(self):
        """min ≤ geometric mean always."""
        lps = [math.log(0.3), math.log(0.6), math.log(0.9)]
        assert min_token_prob(lps) <= normalized_seq_prob(lps) + 1e-12

    def test_single_token(self):
        p = 0.42
        assert min_token_prob([math.log(p)]) == pytest.approx(p)

    def test_empty_raises(self):
        with pytest.raises((ValueError,)):
            min_token_prob([])

    def test_returns_float(self):
        assert isinstance(min_token_prob([math.log(0.5)]), float)


# =============================================================================
# SoftmaxConfidenceScorer — wrapper class
# =============================================================================


class TestSoftmaxConfidenceScorer:
    """High-level scorer wrapping the pure functions."""

    def test_is_uq_scorer(self):
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        assert isinstance(scorer, UQScorer)

    def test_score_returns_uq_score(self):
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        lps = [math.log(0.9), math.log(0.8)]
        result = scorer.score(token_logprobs=lps)
        assert isinstance(result, UQScore)

    def test_score_method_mean_logprob(self):
        scorer = SoftmaxConfidenceScorer(method="mean_logprob")
        lps = [math.log(0.9), math.log(0.8)]
        result = scorer.score(token_logprobs=lps)
        # mean_logprob is negative; scorer should map to [0,1] via exp
        assert 0.0 < result.confidence <= 1.0

    def test_score_method_normalized_seq_prob(self):
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        lps = [math.log(0.9), math.log(0.8)]
        result = scorer.score(token_logprobs=lps)
        expected = math.sqrt(0.9 * 0.8)
        assert result.confidence == pytest.approx(expected)

    def test_score_method_perplexity(self):
        scorer = SoftmaxConfidenceScorer(method="perplexity")
        lps = [math.log(0.5), math.log(0.5)]
        result = scorer.score(token_logprobs=lps)
        assert result.confidence == pytest.approx(0.5)

    def test_score_method_min_token_prob(self):
        scorer = SoftmaxConfidenceScorer(method="min_token_prob")
        lps = [math.log(0.9), math.log(0.3)]
        result = scorer.score(token_logprobs=lps)
        assert result.confidence == pytest.approx(0.3)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="[Mm]ethod"):
            SoftmaxConfidenceScorer(method="invalid")

    def test_name_property(self):
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        assert "softmax" in scorer.name.lower() or "seq_prob" in scorer.name.lower()

    def test_cost_category(self):
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        assert scorer.cost_category == "free"

    def test_none_logprobs_raises(self):
        """Should raise if token_logprobs is None (generation didn't return them)."""
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        with pytest.raises((ValueError, TypeError)):
            scorer.score(token_logprobs=None)

    def test_score_batch(self):
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        batch = [
            [math.log(0.9), math.log(0.8)],
            [math.log(0.5), math.log(0.5)],
            [math.log(0.1), math.log(0.2)],
        ]
        results = scorer.score_batch(batch)
        assert isinstance(results, np.ndarray)
        assert len(results) == 3
        assert results[0] > results[1] > results[2]  # monotonic ordering

    def test_score_batch_returns_confidences(self):
        """score_batch returns array of confidence floats, not UQScore objects."""
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        batch = [[math.log(0.9)], [math.log(0.5)]]
        results = scorer.score_batch(batch)
        assert all(0.0 < r <= 1.0 for r in results)

    def test_metadata_contains_raw_value(self):
        """Metadata should include the raw computed value for debugging."""
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        result = scorer.score(token_logprobs=[math.log(0.8)])
        assert "raw_value" in result.metadata or "mean_logprob" in result.metadata
