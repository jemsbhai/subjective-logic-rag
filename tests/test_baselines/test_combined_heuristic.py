"""Tests for baselines/combined_heuristic.py.

Combined heuristic: scalar combination of retrieval and generation confidence.
Cost: FREE (combines two free signals).

This is the "obvious baseline" a NeurIPS reviewer would propose:
"Why not just multiply retrieval score by softmax confidence?"

Three combination strategies:
    1. product:   conf = retrieval_conf × generation_conf
    2. mean:      conf = (retrieval_conf + generation_conf) / 2
    3. weighted:  conf = w·retrieval_conf + (1-w)·generation_conf
"""

import numpy as np
import pytest

from xrag.baselines.base import UQScore, UQScorer
from xrag.baselines.combined_heuristic import (
    CombinedHeuristicScorer,
)


# =============================================================================
# Product combination
# =============================================================================


class TestProductCombination:
    """conf = retrieval_conf × generation_conf."""

    def test_both_certain(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        result = scorer.score(retrieval_confidence=1.0, generation_confidence=1.0)
        assert result.confidence == pytest.approx(1.0)

    def test_one_zero(self):
        """If either source is 0, product is 0."""
        scorer = CombinedHeuristicScorer(strategy="product")
        result = scorer.score(retrieval_confidence=0.0, generation_confidence=0.9)
        assert result.confidence == pytest.approx(0.0)

    def test_hand_computed(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        result = scorer.score(retrieval_confidence=0.8, generation_confidence=0.6)
        assert result.confidence == pytest.approx(0.48)

    def test_symmetric(self):
        """product(a, b) == product(b, a)."""
        scorer = CombinedHeuristicScorer(strategy="product")
        r1 = scorer.score(retrieval_confidence=0.7, generation_confidence=0.3)
        r2 = scorer.score(retrieval_confidence=0.3, generation_confidence=0.7)
        assert r1.confidence == pytest.approx(r2.confidence)

    def test_bounded_0_1(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        result = scorer.score(retrieval_confidence=0.5, generation_confidence=0.5)
        assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Mean combination
# =============================================================================


class TestMeanCombination:
    """conf = (retrieval_conf + generation_conf) / 2."""

    def test_both_certain(self):
        scorer = CombinedHeuristicScorer(strategy="mean")
        result = scorer.score(retrieval_confidence=1.0, generation_confidence=1.0)
        assert result.confidence == pytest.approx(1.0)

    def test_one_zero(self):
        scorer = CombinedHeuristicScorer(strategy="mean")
        result = scorer.score(retrieval_confidence=0.0, generation_confidence=0.8)
        assert result.confidence == pytest.approx(0.4)

    def test_hand_computed(self):
        scorer = CombinedHeuristicScorer(strategy="mean")
        result = scorer.score(retrieval_confidence=0.8, generation_confidence=0.6)
        assert result.confidence == pytest.approx(0.7)

    def test_symmetric(self):
        scorer = CombinedHeuristicScorer(strategy="mean")
        r1 = scorer.score(retrieval_confidence=0.7, generation_confidence=0.3)
        r2 = scorer.score(retrieval_confidence=0.3, generation_confidence=0.7)
        assert r1.confidence == pytest.approx(r2.confidence)

    def test_bounded_0_1(self):
        scorer = CombinedHeuristicScorer(strategy="mean")
        result = scorer.score(retrieval_confidence=0.5, generation_confidence=0.5)
        assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Weighted combination
# =============================================================================


class TestWeightedCombination:
    """conf = w·retrieval_conf + (1-w)·generation_conf."""

    def test_default_weight_is_half(self):
        """Default w=0.5 → same as mean."""
        scorer = CombinedHeuristicScorer(strategy="weighted")
        result = scorer.score(retrieval_confidence=0.8, generation_confidence=0.6)
        assert result.confidence == pytest.approx(0.7)

    def test_weight_1_is_retrieval_only(self):
        scorer = CombinedHeuristicScorer(strategy="weighted", weight=1.0)
        result = scorer.score(retrieval_confidence=0.8, generation_confidence=0.2)
        assert result.confidence == pytest.approx(0.8)

    def test_weight_0_is_generation_only(self):
        scorer = CombinedHeuristicScorer(strategy="weighted", weight=0.0)
        result = scorer.score(retrieval_confidence=0.8, generation_confidence=0.2)
        assert result.confidence == pytest.approx(0.2)

    def test_hand_computed(self):
        """w=0.3: conf = 0.3*0.8 + 0.7*0.6 = 0.24 + 0.42 = 0.66."""
        scorer = CombinedHeuristicScorer(strategy="weighted", weight=0.3)
        result = scorer.score(retrieval_confidence=0.8, generation_confidence=0.6)
        assert result.confidence == pytest.approx(0.66)

    def test_bounded_0_1(self):
        scorer = CombinedHeuristicScorer(strategy="weighted", weight=0.7)
        result = scorer.score(retrieval_confidence=0.9, generation_confidence=0.1)
        assert 0.0 <= result.confidence <= 1.0

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="[Ww]eight"):
            CombinedHeuristicScorer(strategy="weighted", weight=1.5)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="[Ww]eight"):
            CombinedHeuristicScorer(strategy="weighted", weight=-0.1)


# =============================================================================
# General scorer tests
# =============================================================================


class TestCombinedHeuristicScorer:
    """General tests for the CombinedHeuristicScorer wrapper."""

    def test_is_uq_scorer(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        assert isinstance(scorer, UQScorer)

    def test_returns_uq_score(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        result = scorer.score(retrieval_confidence=0.5, generation_confidence=0.5)
        assert isinstance(result, UQScore)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="[Ss]trategy"):
            CombinedHeuristicScorer(strategy="invalid")

    def test_name_contains_strategy(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        assert "product" in scorer.name.lower()

    def test_cost_category(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        assert scorer.cost_category == "free"

    def test_metadata_present(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        result = scorer.score(retrieval_confidence=0.8, generation_confidence=0.6)
        assert "retrieval_confidence" in result.metadata
        assert "generation_confidence" in result.metadata
        assert "strategy" in result.metadata

    def test_none_retrieval_raises(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        with pytest.raises((ValueError, TypeError)):
            scorer.score(retrieval_confidence=None, generation_confidence=0.5)

    def test_none_generation_raises(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        with pytest.raises((ValueError, TypeError)):
            scorer.score(retrieval_confidence=0.5, generation_confidence=None)

    def test_score_batch(self):
        scorer = CombinedHeuristicScorer(strategy="product")
        batch = [
            {"retrieval_confidence": 0.9, "generation_confidence": 0.8},
            {"retrieval_confidence": 0.5, "generation_confidence": 0.5},
            {"retrieval_confidence": 0.1, "generation_confidence": 0.2},
        ]
        results = scorer.score_batch(batch)
        assert isinstance(results, np.ndarray)
        assert len(results) == 3
        assert results[0] == pytest.approx(0.72)
        assert results[1] == pytest.approx(0.25)
        assert results[2] == pytest.approx(0.02)

    def test_product_leq_mean(self):
        """For inputs in [0,1]: product ≤ mean (AM-GM inequality)."""
        prod_scorer = CombinedHeuristicScorer(strategy="product")
        mean_scorer = CombinedHeuristicScorer(strategy="mean")
        for r, g in [(0.9, 0.3), (0.5, 0.5), (0.1, 0.8), (0.7, 0.7)]:
            p = prod_scorer.score(retrieval_confidence=r, generation_confidence=g)
            m = mean_scorer.score(retrieval_confidence=r, generation_confidence=g)
            assert p.confidence <= m.confidence + 1e-12, (
                f"Product {p.confidence} > Mean {m.confidence} for r={r}, g={g}"
            )
