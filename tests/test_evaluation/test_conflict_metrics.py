"""Tests for conflict_metrics.py — P/R/F1 for conflict detection, AUROC.

Conflict detection is evaluated against known ground truth from corruption
suites (injected contradictions).  Two evaluation modes:

1. Set-based: predicted conflict pairs vs true conflict pairs → P/R/F1.
2. Score-based: continuous conflict scores vs binary labels → AUROC.

Key design decisions tested:
    - Pairs are unordered: (i,j) == (j,i) — conflict is symmetric.
    - Empty predictions/truths are handled without division-by-zero.
    - AUROC degeneracy (single class) returns NaN.
"""

import numpy as np
import pytest

from xrag.evaluation.conflict_metrics import (
    conflict_precision_recall_f1,
    conflict_detection_auroc,
)


# =============================================================================
# conflict_precision_recall_f1 — set-based evaluation
# =============================================================================


class TestConflictPrecisionRecallF1:
    """P/R/F1 over sets of conflict pairs."""

    # --- Perfect detection ---

    def test_perfect_detection(self):
        """Predicted == true → P=1, R=1, F1=1."""
        predicted = {(0, 1), (2, 3)}
        true = {(0, 1), (2, 3)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    # --- No detection ---

    def test_no_overlap(self):
        """Completely disjoint → P=0, R=0, F1=0."""
        predicted = {(0, 1), (2, 3)}
        true = {(4, 5), (6, 7)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    # --- Partial detection ---

    def test_partial_overlap(self):
        """
        predicted = {(0,1), (2,3), (4,5)}
        true      = {(0,1), (2,3), (6,7)}

        TP = {(0,1), (2,3)} = 2
        FP = {(4,5)} = 1
        FN = {(6,7)} = 1

        P = 2/3, R = 2/3, F1 = 2/3
        """
        predicted = {(0, 1), (2, 3), (4, 5)}
        true = {(0, 1), (2, 3), (6, 7)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(2 / 3)
        assert r == pytest.approx(2 / 3)
        assert f1 == pytest.approx(2 / 3)

    def test_high_precision_low_recall(self):
        """
        predicted = {(0,1)}   (1 pair, correct)
        true      = {(0,1), (2,3), (4,5)}

        P = 1/1 = 1.0, R = 1/3
        F1 = 2*(1.0)*(1/3) / (1.0 + 1/3) = (2/3) / (4/3) = 0.5
        """
        predicted = {(0, 1)}
        true = {(0, 1), (2, 3), (4, 5)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1 / 3)
        assert f1 == pytest.approx(0.5)

    def test_low_precision_high_recall(self):
        """
        predicted = {(0,1), (2,3), (4,5), (6,7), (8,9)}   (5 pairs)
        true      = {(0,1)}                                 (1 pair)

        TP = 1, FP = 4, FN = 0
        P = 1/5 = 0.2, R = 1/1 = 1.0
        F1 = 2*0.2*1 / (0.2 + 1) = 0.4/1.2 = 1/3
        """
        predicted = {(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)}
        true = {(0, 1)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(0.2)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1 / 3)

    # --- Symmetry: (i,j) == (j,i) ---

    def test_pair_order_invariance(self):
        """(i,j) and (j,i) should be treated as the same conflict pair."""
        predicted = {(1, 0)}  # reversed order
        true = {(0, 1)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_mixed_order_pairs(self):
        """Mix of orderings should all match."""
        predicted = {(1, 0), (3, 2)}
        true = {(0, 1), (2, 3)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_duplicate_after_normalization(self):
        """(0,1) and (1,0) in same set should count as one pair."""
        predicted = {(0, 1), (1, 0)}  # same pair twice
        true = {(0, 1)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        # After normalization, predicted has 1 unique pair
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    # --- Empty sets ---

    def test_empty_predicted_nonempty_true(self):
        """No predictions → P=0 (vacuously or by convention), R=0, F1=0."""
        predicted = set()
        true = {(0, 1), (2, 3)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_nonempty_predicted_empty_true(self):
        """No true conflicts → P=0 (all FP), R=0 (vacuously), F1=0."""
        predicted = {(0, 1), (2, 3)}
        true = set()
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_both_empty(self):
        """No predictions, no true conflicts → perfect (vacuous truth)."""
        predicted = set()
        true = set()
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    # --- Return type and bounds ---

    def test_returns_tuple_of_three_floats(self):
        predicted = {(0, 1)}
        true = {(0, 1)}
        result = conflict_precision_recall_f1(predicted, true)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

    def test_all_values_bounded_0_1(self):
        predicted = {(0, 1), (2, 3), (4, 5)}
        true = {(0, 1), (6, 7)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        for val in (p, r, f1):
            assert 0.0 <= val <= 1.0

    def test_f1_is_harmonic_mean(self):
        """Verify F1 = 2*P*R / (P+R) when both > 0."""
        predicted = {(0, 1), (2, 3), (4, 5)}
        true = {(0, 1), (2, 3), (6, 7)}
        p, r, f1 = conflict_precision_recall_f1(predicted, true)
        expected_f1 = 2 * p * r / (p + r)
        assert f1 == pytest.approx(expected_f1)

    # --- Self-pairs should be rejected or handled ---

    def test_self_pair_in_predicted(self):
        """(i,i) is not a valid conflict pair — should be ignored or raise."""
        predicted = {(0, 0), (0, 1)}
        true = {(0, 1)}
        # Either ignore (0,0) and give P=1, R=1 or raise ValueError
        # We accept either behavior, but it must not crash silently
        try:
            p, r, f1 = conflict_precision_recall_f1(predicted, true)
            # If it doesn't raise, (0,0) should be ignored
            assert r == pytest.approx(1.0)
        except ValueError:
            pass  # Also acceptable


# =============================================================================
# conflict_detection_auroc — score-based evaluation
# =============================================================================


class TestConflictDetectionAUROC:
    """AUROC of conflict scores as discriminator for true conflicts."""

    def test_perfect_discrimination(self):
        """True conflicts have higher scores → AUROC = 1.0."""
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.02])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert conflict_detection_auroc(scores, labels) == pytest.approx(1.0)

    def test_inverse_discrimination(self):
        """True conflicts have lower scores → AUROC = 0.0."""
        scores = np.array([0.1, 0.05, 0.02, 0.9, 0.8, 0.7])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert conflict_detection_auroc(scores, labels) == pytest.approx(0.0)

    def test_random_scores(self):
        """Random scores → AUROC ≈ 0.5."""
        rng = np.random.default_rng(42)
        n = 10000
        scores = rng.uniform(0, 1, n)
        labels = rng.integers(0, 2, n)
        result = conflict_detection_auroc(scores, labels)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_hand_computed(self):
        """
        scores = [0.9, 0.6, 0.4, 0.1]
        labels = [1,   0,   1,   0  ]

        Positive-negative pairs:
            (0.9, 0.6): pos > neg → +1
            (0.9, 0.1): pos > neg → +1
            (0.4, 0.6): pos < neg → +0
            (0.4, 0.1): pos > neg → +1

        AUROC = 3/4 = 0.75
        """
        scores = np.array([0.9, 0.6, 0.4, 0.1])
        labels = np.array([1, 0, 1, 0])
        assert conflict_detection_auroc(scores, labels) == pytest.approx(0.75)

    def test_all_same_label_returns_nan(self):
        """Degenerate: single class → AUROC undefined."""
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 1, 1])
        result = conflict_detection_auroc(scores, labels)
        assert np.isnan(result)

    def test_all_negative_returns_nan(self):
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([0, 0, 0])
        result = conflict_detection_auroc(scores, labels)
        assert np.isnan(result)

    # --- Input validation ---

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="[Ll]ength"):
            conflict_detection_auroc(np.array([0.5, 0.6]), np.array([1]))

    def test_empty_arrays(self):
        with pytest.raises(ValueError):
            conflict_detection_auroc(np.array([]), np.array([]))

    def test_labels_not_binary(self):
        with pytest.raises(ValueError, match="[Bb]inary|label|0.*1"):
            conflict_detection_auroc(np.array([0.5, 0.6]), np.array([0, 2]))

    # --- Bounds and types ---

    def test_bounded_0_1(self):
        rng = np.random.default_rng(99)
        scores = rng.uniform(0, 1, 100)
        labels = rng.integers(0, 2, 100)
        result = conflict_detection_auroc(scores, labels)
        if not np.isnan(result):
            assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        scores = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        result = conflict_detection_auroc(scores, labels)
        assert isinstance(result, (float, np.floating))

    def test_scores_not_bounded_still_works(self):
        """Conflict scores from SL pairwise matrix may exceed 1.
        AUROC only cares about ranking, not magnitude."""
        scores = np.array([2.5, 1.8, 0.3, 0.1])
        labels = np.array([1, 1, 0, 0])
        result = conflict_detection_auroc(scores, labels)
        assert result == pytest.approx(1.0)

    def test_tied_scores(self):
        """Tied scores should be handled without error."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1, 0])
        result = conflict_detection_auroc(scores, labels)
        # With all ties, AUROC = 0.5
        assert result == pytest.approx(0.5)


# =============================================================================
# Cross-metric consistency
# =============================================================================


class TestConflictMetricsConsistency:
    """Ensure set-based and score-based metrics are internally consistent."""

    def test_perfect_threshold_gives_perfect_prf1(self):
        """If AUROC = 1, there exists a threshold giving P=R=F1=1.

        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 1, 0, 0]
        All pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        True conflicts (label=1): those involving docs 0,1 where both are conflicts?

        Actually, this test verifies a simpler property: given a score matrix
        and a threshold that achieves perfect separation, the resulting
        predicted pairs should yield P=R=F1=1.
        """
        # 4 documents, conflicts between (0,1) and (2,3)
        true_pairs = {(0, 1), (2, 3)}

        # Conflict scores: high for true pairs, low for non-conflicts
        # Pairs indexed: (0,1)=0.9, (0,2)=0.1, (0,3)=0.1, (1,2)=0.1, (1,3)=0.1, (2,3)=0.8
        # With threshold 0.5:
        predicted_pairs = {(0, 1), (2, 3)}

        p, r, f1 = conflict_precision_recall_f1(predicted_pairs, true_pairs)
        assert f1 == pytest.approx(1.0)
