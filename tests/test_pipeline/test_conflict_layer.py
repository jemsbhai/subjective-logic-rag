"""Tests for the real SL conflict detection layer implementation.

Covers pairwise conflict matrix construction, threshold-based flagging,
ConflictResult metadata, mathematical properties of pairwise_conflict(),
and edge cases.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion, pairwise_conflict, conflict_metric

from xrag.pipeline.conflict_layer import (
    ConflictLayer,
    NoOpConflictLayer,
    SLConflictLayer,
    ConflictResult,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def three_agreeing():
    """Three opinions that mostly agree (high belief)."""
    return [
        Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5),
        Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17, base_rate=0.5),
    ]


@pytest.fixture
def two_conflicting():
    """Two opinions that strongly disagree."""
    return [
        Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05, base_rate=0.5),
        Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05, base_rate=0.5),
    ]


@pytest.fixture
def mixed_with_conflict():
    """Three agreeing + one contradicting → one conflicting pair cluster."""
    return [
        Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5),
        Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17, base_rate=0.5),
        Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05, base_rate=0.5),  # contradicts
    ]


# ════════════════════════════════════════════════════════════════════
# SLConflictLayer — construction
# ════════════════════════════════════════════════════════════════════


class TestSLConflictLayerConstruction:

    def test_is_subclass(self):
        layer = SLConflictLayer(threshold=0.3)
        assert isinstance(layer, ConflictLayer)

    def test_stores_threshold(self):
        layer = SLConflictLayer(threshold=0.25)
        assert layer.threshold == 0.25

    def test_default_threshold(self):
        layer = SLConflictLayer()
        assert layer.threshold == 0.3  # sensible default

    def test_last_result_initially_none(self):
        layer = SLConflictLayer()
        assert layer.last_result is None


# ════════════════════════════════════════════════════════════════════
# Conflict matrix construction
# ════════════════════════════════════════════════════════════════════


class TestConflictMatrix:
    """Verify the NxN pairwise conflict matrix is correct."""

    def test_matrix_dimensions(self, three_agreeing):
        layer = SLConflictLayer()
        layer.detect(three_agreeing)
        matrix = layer.last_result.conflict_matrix
        n = len(three_agreeing)
        assert len(matrix) == n
        assert all(len(row) == n for row in matrix)

    def test_diagonal_is_zero(self, three_agreeing):
        layer = SLConflictLayer()
        layer.detect(three_agreeing)
        matrix = layer.last_result.conflict_matrix
        for i in range(len(three_agreeing)):
            assert matrix[i][i] == 0.0

    def test_symmetric(self, mixed_with_conflict):
        layer = SLConflictLayer()
        layer.detect(mixed_with_conflict)
        matrix = layer.last_result.conflict_matrix
        n = len(mixed_with_conflict)
        for i in range(n):
            for j in range(i + 1, n):
                assert matrix[i][j] == pytest.approx(matrix[j][i])

    def test_matches_jsonld_ex_pairwise_conflict(self, mixed_with_conflict):
        """Each matrix cell matches a direct pairwise_conflict() call."""
        layer = SLConflictLayer()
        layer.detect(mixed_with_conflict)
        matrix = layer.last_result.conflict_matrix
        n = len(mixed_with_conflict)
        for i in range(n):
            for j in range(i + 1, n):
                expected = pairwise_conflict(mixed_with_conflict[i], mixed_with_conflict[j])
                assert matrix[i][j] == pytest.approx(expected)

    def test_values_in_unit_interval(self, mixed_with_conflict):
        layer = SLConflictLayer()
        layer.detect(mixed_with_conflict)
        matrix = layer.last_result.conflict_matrix
        for row in matrix:
            for val in row:
                assert 0.0 <= val <= 1.0 + 1e-9


# ════════════════════════════════════════════════════════════════════
# Conflict detection — agreeing opinions
# ════════════════════════════════════════════════════════════════════


class TestAgreeingOpinions:

    def test_no_conflict_detected(self, three_agreeing):
        layer = SLConflictLayer(threshold=0.3)
        result = layer.detect(three_agreeing)
        assert result.conflict_detected is False

    def test_no_conflict_pairs(self, three_agreeing):
        layer = SLConflictLayer(threshold=0.3)
        result = layer.detect(three_agreeing)
        assert result.conflict_pairs == []

    def test_low_max_conflict(self, three_agreeing):
        layer = SLConflictLayer()
        result = layer.detect(three_agreeing)
        assert result.max_conflict_score < 0.3


# ════════════════════════════════════════════════════════════════════
# Conflict detection — conflicting opinions
# ════════════════════════════════════════════════════════════════════


class TestConflictingOpinions:

    def test_conflict_detected(self, two_conflicting):
        layer = SLConflictLayer(threshold=0.3)
        result = layer.detect(two_conflicting)
        assert result.conflict_detected is True

    def test_conflict_pair_identified(self, two_conflicting):
        layer = SLConflictLayer(threshold=0.3)
        result = layer.detect(two_conflicting)
        assert (0, 1) in result.conflict_pairs

    def test_high_max_conflict(self, two_conflicting):
        layer = SLConflictLayer()
        result = layer.detect(two_conflicting)
        # b_A*d_B + d_A*b_B = 0.9*0.9 + 0.05*0.05 = 0.81 + 0.0025 = 0.8125
        assert result.max_conflict_score > 0.8

    def test_mixed_identifies_rogue_pairs(self, mixed_with_conflict):
        """In a group of 3 agreeing + 1 contradicting, the contradicting
        doc should have high conflict with all 3 agreeing docs."""
        layer = SLConflictLayer(threshold=0.3)
        result = layer.detect(mixed_with_conflict)
        assert result.conflict_detected is True
        # Index 3 (the contradicting doc) should appear in conflict pairs
        rogue_pairs = [p for p in result.conflict_pairs if 3 in p]
        assert len(rogue_pairs) >= 1  # at least one pair involving doc 3


# ════════════════════════════════════════════════════════════════════
# Threshold behavior
# ════════════════════════════════════════════════════════════════════


class TestThresholdBehavior:

    def test_very_high_threshold_no_detection(self, two_conflicting):
        """Threshold above maximum possible conflict → nothing flagged."""
        layer = SLConflictLayer(threshold=0.99)
        result = layer.detect(two_conflicting)
        # pairwise conflict max is ~0.81, so threshold 0.99 misses it
        assert result.conflict_detected is False
        assert result.conflict_pairs == []

    def test_very_low_threshold_catches_everything(self, three_agreeing):
        """Threshold near zero → even minor disagreement flagged."""
        layer = SLConflictLayer(threshold=0.001)
        result = layer.detect(three_agreeing)
        # Agreeing docs still have nonzero pairwise conflict
        # (b_A*d_B + d_A*b_B > 0 unless one is vacuous)
        if result.max_conflict_score > 0.001:
            assert result.conflict_detected is True

    def test_threshold_is_exclusive(self, mixed_with_conflict):
        """A pair with conflict == threshold should NOT be flagged
        (strictly greater than threshold)."""
        layer = SLConflictLayer()
        result_ref = layer.detect(mixed_with_conflict)
        # Set threshold to exactly the max → should not flag
        layer2 = SLConflictLayer(threshold=result_ref.max_conflict_score)
        result = layer2.detect(mixed_with_conflict)
        # Pairs with conflict == threshold are NOT included
        assert all(
            layer2.last_result.conflict_matrix[i][j] > result_ref.max_conflict_score
            for i, j in result.conflict_pairs
        ) or result.conflict_pairs == []


# ════════════════════════════════════════════════════════════════════
# ConflictResult enriched fields
# ════════════════════════════════════════════════════════════════════


class TestConflictResultFields:
    """Test enriched ConflictResult fields beyond the ABC minimum."""

    def test_conflict_matrix_in_result(self, mixed_with_conflict):
        layer = SLConflictLayer()
        layer.detect(mixed_with_conflict)
        assert layer.last_result.conflict_matrix is not None

    def test_mean_conflict_in_result(self, mixed_with_conflict):
        layer = SLConflictLayer()
        layer.detect(mixed_with_conflict)
        lr = layer.last_result
        assert lr.mean_conflict is not None
        assert 0.0 <= lr.mean_conflict <= 1.0

    def test_mean_conflict_correct(self, mixed_with_conflict):
        """Mean conflict should be the mean of upper-triangle entries."""
        layer = SLConflictLayer()
        layer.detect(mixed_with_conflict)
        lr = layer.last_result
        n = len(mixed_with_conflict)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += lr.conflict_matrix[i][j]
                count += 1
        expected_mean = total / count
        assert lr.mean_conflict == pytest.approx(expected_mean)

    def test_per_doc_discord_scores(self, mixed_with_conflict):
        """Per-doc discord: mean pairwise conflict for each doc."""
        layer = SLConflictLayer()
        layer.detect(mixed_with_conflict)
        lr = layer.last_result
        assert lr.discord_scores is not None
        assert len(lr.discord_scores) == len(mixed_with_conflict)
        # Doc 3 (the rogue) should have highest discord
        assert lr.discord_scores[3] == max(lr.discord_scores)

    def test_discord_scores_nonnegative(self, three_agreeing):
        layer = SLConflictLayer()
        layer.detect(three_agreeing)
        for d in layer.last_result.discord_scores:
            assert d >= 0.0


# ════════════════════════════════════════════════════════════════════
# Mathematical properties
# ════════════════════════════════════════════════════════════════════


class TestConflictMathProperties:

    def test_vacuous_opinions_zero_conflict(self):
        """Vacuous opinions have zero pairwise conflict."""
        vac = [
            Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0),
            Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0),
        ]
        layer = SLConflictLayer()
        result = layer.detect(vac)
        assert result.max_conflict_score == pytest.approx(0.0)

    def test_identical_opinions_nonzero_conflict(self):
        """Identical non-vacuous opinions can have nonzero pairwise conflict.

        pairwise_conflict measures b_A*d_B + d_A*b_B.
        For identical opinions with b>0 and d>0: 2*b*d > 0.
        This is NOT a bug — it reflects that the evidence contains
        internal tension, which is a different concept from disagreement.
        """
        op = Opinion(belief=0.5, disbelief=0.4, uncertainty=0.1)
        layer = SLConflictLayer()
        result = layer.detect([op, op])
        expected = 2 * op.belief * op.disbelief  # 2 * 0.5 * 0.4 = 0.4
        assert result.max_conflict_score == pytest.approx(expected)

    def test_symmetry(self, two_conflicting):
        """pairwise_conflict(a, b) == pairwise_conflict(b, a)."""
        layer = SLConflictLayer()
        layer.detect(two_conflicting)
        matrix = layer.last_result.conflict_matrix
        assert matrix[0][1] == pytest.approx(matrix[1][0])

    def test_max_conflict_between_dogmatic_opposites(self):
        """Maximum conflict (1.0) between full belief and full disbelief."""
        full_b = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        full_d = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        layer = SLConflictLayer()
        result = layer.detect([full_b, full_d])
        assert result.max_conflict_score == pytest.approx(1.0)

    def test_conflict_formula(self, two_conflicting):
        """Verify con(A,B) = b_A*d_B + d_A*b_B."""
        a, b = two_conflicting
        expected = a.belief * b.disbelief + a.disbelief * b.belief
        layer = SLConflictLayer()
        result = layer.detect(two_conflicting)
        assert result.max_conflict_score == pytest.approx(expected)


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════


class TestConflictEdgeCases:

    def test_empty_list(self):
        layer = SLConflictLayer()
        result = layer.detect([])
        assert result.conflict_detected is False
        assert result.conflict_pairs == []
        assert result.max_conflict_score == 0.0

    def test_single_opinion(self):
        op = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        layer = SLConflictLayer()
        result = layer.detect([op])
        assert result.conflict_detected is False
        assert result.conflict_pairs == []
        assert result.max_conflict_score == 0.0

    def test_two_identical_opinions(self):
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        layer = SLConflictLayer(threshold=0.3)
        result = layer.detect([op, op])
        # b_A*d_B + d_A*b_B = 2*0.7*0.1 = 0.14 < 0.3
        assert result.conflict_detected is False

    def test_last_result_updated_each_call(self):
        layer = SLConflictLayer()
        op1 = [Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)]
        op2 = [
            Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05),
            Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05),
        ]
        layer.detect(op1)
        assert layer.last_result.conflict_detected is False
        layer.detect(op2)
        assert layer.last_result.conflict_detected is True


# ════════════════════════════════════════════════════════════════════
# NoOpConflictLayer still works
# ════════════════════════════════════════════════════════════════════


class TestNoOpConflictLayerStillWorks:

    def test_no_conflict(self, three_agreeing):
        layer = NoOpConflictLayer()
        result = layer.detect(three_agreeing)
        assert result.conflict_detected is False
        assert result.conflict_pairs == []
        assert result.max_conflict_score == 0.0

    def test_is_conflict_layer_subclass(self):
        assert isinstance(NoOpConflictLayer(), ConflictLayer)
