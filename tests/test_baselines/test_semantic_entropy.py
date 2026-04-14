"""Tests for baselines/semantic_entropy.py.

Semantic Entropy (Kuhn, Gal, Farquhar — ICLR 2023, Nature 2024).
Cost: N × generation + O(N²) NLI calls.

Algorithm:
    1. Generate N answers with temperature > 0
    2. Cluster by bidirectional NLI entailment (semantic equivalence)
    3. Entropy over cluster distribution: SE = -Σ p_c · log p_c
    4. Confidence = exp(-SE) or 1/(1+SE)

Core functions tested with mocks (no real models needed):
    - cluster_by_equivalence(answers, equiv_fn) → clusters
    - semantic_entropy_from_clusters(clusters, n_total) → SE
    - SemanticEntropyScorer.score(query, passages) → UQScore
"""

import math

import numpy as np
import pytest

from xrag.baselines.base import UQScore, UQScorer
from xrag.baselines.semantic_entropy import (
    cluster_by_equivalence,
    semantic_entropy_from_clusters,
    SemanticEntropyScorer,
)


# =============================================================================
# cluster_by_equivalence
# =============================================================================


class TestClusterByEquivalence:
    """Cluster answers by semantic equivalence."""

    def _exact_match_equiv(self, a: str, b: str) -> bool:
        """Trivial equivalence: exact string match."""
        return a.strip().lower() == b.strip().lower()

    def test_all_same(self):
        """All identical → 1 cluster."""
        answers = ["Paris", "Paris", "Paris"]
        clusters = cluster_by_equivalence(answers, self._exact_match_equiv)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_all_different(self):
        """All distinct → N clusters."""
        answers = ["Paris", "London", "Berlin"]
        clusters = cluster_by_equivalence(answers, self._exact_match_equiv)
        assert len(clusters) == 3

    def test_two_clusters(self):
        """Two groups of equivalent answers."""
        answers = ["Paris", "paris", "London", "london"]
        clusters = cluster_by_equivalence(answers, self._exact_match_equiv)
        assert len(clusters) == 2
        sizes = sorted([len(c) for c in clusters])
        assert sizes == [2, 2]

    def test_uneven_clusters(self):
        """3 same + 1 different → 2 clusters of sizes [3, 1]."""
        answers = ["Paris", "Paris", "Paris", "London"]
        clusters = cluster_by_equivalence(answers, self._exact_match_equiv)
        assert len(clusters) == 2
        sizes = sorted([len(c) for c in clusters])
        assert sizes == [1, 3]

    def test_single_answer(self):
        clusters = cluster_by_equivalence(["Paris"], self._exact_match_equiv)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cluster_by_equivalence([], self._exact_match_equiv)

    def test_preserves_all_answers(self):
        """Total elements across clusters equals input length."""
        answers = ["a", "b", "a", "c", "b"]
        clusters = cluster_by_equivalence(answers, self._exact_match_equiv)
        total = sum(len(c) for c in clusters)
        assert total == 5

    def test_custom_equivalence_function(self):
        """Equivalence by first character."""
        def first_char_equiv(a: str, b: str) -> bool:
            return a[0].lower() == b[0].lower()

        answers = ["Paris", "Prague", "London", "Lyon"]
        clusters = cluster_by_equivalence(answers, first_char_equiv)
        assert len(clusters) == 2  # P-group and L-group

    def test_returns_list_of_lists(self):
        clusters = cluster_by_equivalence(["a", "b"], self._exact_match_equiv)
        assert isinstance(clusters, list)
        assert all(isinstance(c, list) for c in clusters)

    def test_cluster_elements_are_strings(self):
        clusters = cluster_by_equivalence(["hello"], self._exact_match_equiv)
        assert all(isinstance(s, str) for c in clusters for s in c)


# =============================================================================
# semantic_entropy_from_clusters
# =============================================================================


class TestSemanticEntropyFromClusters:
    """SE = -Σ (n_c/N) · log(n_c/N) over clusters."""

    def test_single_cluster(self):
        """All answers in one cluster → SE = 0 (complete agreement)."""
        clusters = [["Paris", "Paris", "Paris"]]
        se = semantic_entropy_from_clusters(clusters, n_total=3)
        assert se == pytest.approx(0.0)

    def test_all_singletons(self):
        """N singletons → SE = log(N) (maximum disagreement)."""
        clusters = [["Paris"], ["London"], ["Berlin"]]
        se = semantic_entropy_from_clusters(clusters, n_total=3)
        assert se == pytest.approx(math.log(3))

    def test_two_equal_clusters(self):
        """Two clusters of equal size → SE = log(2)."""
        clusters = [["Paris", "Paris"], ["London", "London"]]
        se = semantic_entropy_from_clusters(clusters, n_total=4)
        assert se == pytest.approx(math.log(2))

    def test_hand_computed(self):
        """
        clusters: [3, 1, 1] out of 5 total
        p = [3/5, 1/5, 1/5]
        SE = -(3/5)·log(3/5) - (1/5)·log(1/5) - (1/5)·log(1/5)
           = -(0.6·log(0.6) + 0.4·log(0.2))
           wait, let me recompute:
           = -(0.6·(-0.5108) + 0.2·(-1.6094) + 0.2·(-1.6094))
           = -(−0.3065 − 0.3219 − 0.3219)
           = 0.9503
        """
        clusters = [["a", "a", "a"], ["b"], ["c"]]
        se = semantic_entropy_from_clusters(clusters, n_total=5)
        p = np.array([3 / 5, 1 / 5, 1 / 5])
        expected = float(-np.sum(p * np.log(p)))
        assert se == pytest.approx(expected)

    def test_nonnegative(self):
        clusters = [["a", "a"], ["b"]]
        se = semantic_entropy_from_clusters(clusters, n_total=3)
        assert se >= 0.0

    def test_returns_float(self):
        result = semantic_entropy_from_clusters([["a"]], n_total=1)
        assert isinstance(result, float)

    def test_n_total_must_match(self):
        """n_total must equal sum of cluster sizes."""
        with pytest.raises(ValueError, match="[Mm]ismatch|[Tt]otal"):
            semantic_entropy_from_clusters([["a"], ["b"]], n_total=5)


# =============================================================================
# SemanticEntropyScorer — integration with mocked generator
# =============================================================================


class TestSemanticEntropyScorer:
    """Full scorer with mocked generator and equivalence function."""

    def _make_mock_generator(self, answers: list[str]):
        """Create a mock generator that returns pre-defined answers."""
        call_count = [0]

        class MockGenerator:
            def generate(self, query, passages, max_new_tokens=128,
                         return_logprobs=False, temperature=None, **kwargs):
                from xrag.generation.generator import GenerationResult
                idx = call_count[0] % len(answers)
                call_count[0] += 1
                return GenerationResult(
                    answer=answers[idx],
                    prompt=f"mock prompt for {query}",
                )

        return MockGenerator()

    def _exact_equiv(self, a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()

    def test_is_uq_scorer(self):
        gen = self._make_mock_generator(["Paris"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=3
        )
        assert isinstance(scorer, UQScorer)

    def test_cost_category(self):
        gen = self._make_mock_generator(["Paris"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=5
        )
        assert scorer.cost_category == "n_samples"

    def test_name_contains_semantic(self):
        gen = self._make_mock_generator(["Paris"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=3
        )
        assert "semantic" in scorer.name.lower()

    def test_all_agree_high_confidence(self):
        """All samples produce same answer → SE=0 → high confidence."""
        gen = self._make_mock_generator(["Paris"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=5
        )
        result = scorer.score(query="What is the capital of France?", passages=[])
        assert isinstance(result, UQScore)
        assert result.confidence == pytest.approx(1.0)

    def test_all_disagree_low_confidence(self):
        """All samples produce different answers → high SE → low confidence."""
        gen = self._make_mock_generator(["Paris", "London", "Berlin", "Rome", "Madrid"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=5
        )
        result = scorer.score(query="test", passages=[])
        assert result.confidence < 0.3

    def test_partial_agreement(self):
        """3 same + 2 different → moderate confidence."""
        gen = self._make_mock_generator(["Paris", "Paris", "Paris", "London", "Berlin"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=5
        )
        result = scorer.score(query="test", passages=[])
        assert 0.1 < result.confidence < 0.9

    def test_confidence_bounded_0_1(self):
        gen = self._make_mock_generator(["a", "b", "c"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=3
        )
        result = scorer.score(query="test", passages=[])
        assert 0.0 <= result.confidence <= 1.0

    def test_n_samples_controls_generation_count(self):
        """Should call generator exactly n_samples times."""
        call_count = [0]

        class CountingGenerator:
            def generate(self, *args, **kwargs):
                from xrag.generation.generator import GenerationResult
                call_count[0] += 1
                return GenerationResult(answer="Paris", prompt="p")

        scorer = SemanticEntropyScorer(
            generator=CountingGenerator(),
            equivalence_fn=self._exact_equiv,
            n_samples=7,
        )
        scorer.score(query="test", passages=[])
        assert call_count[0] == 7

    def test_metadata_contains_clusters(self):
        gen = self._make_mock_generator(["Paris", "Paris", "London"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=3
        )
        result = scorer.score(query="test", passages=[])
        assert "n_clusters" in result.metadata
        assert "semantic_entropy" in result.metadata
        assert result.metadata["n_clusters"] == 2

    def test_metadata_contains_answers(self):
        gen = self._make_mock_generator(["Paris", "London"])
        scorer = SemanticEntropyScorer(
            generator=gen, equivalence_fn=self._exact_equiv, n_samples=2
        )
        result = scorer.score(query="test", passages=[])
        assert "answers" in result.metadata
        assert len(result.metadata["answers"]) == 2
