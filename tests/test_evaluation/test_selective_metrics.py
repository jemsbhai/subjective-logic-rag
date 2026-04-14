"""Tests for selective_metrics.py — AUROC, AUPRC, risk-coverage, AURC, E-AURC.

These metrics evaluate the quality of selective prediction (abstention).
The core idea: if the model's confidence ranking is good, it should be able
to abstain on hard/wrong examples and achieve lower risk at lower coverage.

References:
    Geifman & El-Yaniv (2017). "Selective Classification for Deep Neural Networks."
    AURC / E-AURC are standard in the selective prediction literature.
"""

import numpy as np
import pytest

from xrag.evaluation.selective_metrics import (
    auroc_selective,
    auprc_selective,
    risk_coverage_curve,
    aurc,
    oracle_aurc,
    e_aurc,
    RiskCoverageCurve,
)


# =============================================================================
# Input validation
# =============================================================================


class TestSelectiveInputValidation:
    """All selective metrics must validate inputs consistently."""

    def test_auroc_mismatched_lengths(self):
        with pytest.raises(ValueError, match="[Ll]ength"):
            auroc_selective(np.array([0.5, 0.6]), np.array([1]))

    def test_auprc_mismatched_lengths(self):
        with pytest.raises(ValueError, match="[Ll]ength"):
            auprc_selective(np.array([0.5]), np.array([1, 0]))

    def test_aurc_mismatched_lengths(self):
        with pytest.raises(ValueError, match="[Ll]ength"):
            aurc(np.array([0.5, 0.6]), np.array([0.0]))

    def test_auroc_empty(self):
        with pytest.raises(ValueError):
            auroc_selective(np.array([]), np.array([]))

    def test_risk_coverage_empty(self):
        with pytest.raises(ValueError):
            risk_coverage_curve(np.array([]), np.array([]))

    def test_auroc_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="[Cc]onfidence|[Rr]ange|bound"):
            auroc_selective(np.array([-0.1, 0.5]), np.array([0, 1]))

    def test_auroc_outcomes_not_binary(self):
        with pytest.raises(ValueError, match="[Bb]inary|outcome|0.*1"):
            auroc_selective(np.array([0.5, 0.6]), np.array([0, 2]))

    def test_aurc_losses_negative(self):
        """Losses should be non-negative."""
        with pytest.raises(ValueError, match="[Ll]oss|[Nn]egative|[Rr]ange"):
            aurc(np.array([0.5, 0.6]), np.array([-0.1, 0.0]))


# =============================================================================
# AUROC — selective prediction
# =============================================================================


class TestAUROCSelective:
    """AUROC: can the confidence score distinguish correct from incorrect?"""

    def test_perfect_ranking(self):
        """All correct have higher confidence than all incorrect → AUROC = 1.0."""
        confs = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        outcomes = np.array([1, 1, 1, 0, 0, 0])
        assert auroc_selective(confs, outcomes) == pytest.approx(1.0)

    def test_inverse_ranking(self):
        """All incorrect have higher confidence → AUROC = 0.0."""
        confs = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        outcomes = np.array([0, 0, 0, 1, 1, 1])
        assert auroc_selective(confs, outcomes) == pytest.approx(0.0)

    def test_random_ranking(self):
        """Random confidence → AUROC ≈ 0.5."""
        rng = np.random.default_rng(42)
        n = 10000
        outcomes = rng.integers(0, 2, n)
        confs = rng.uniform(0, 1, n)  # random, unrelated to outcomes
        result = auroc_selective(confs, outcomes)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_all_correct_returns_nan_or_handles(self):
        """Degenerate: all outcomes are 1 → AUROC undefined.
        Should return NaN or a documented fallback, not crash."""
        confs = np.array([0.9, 0.8, 0.7])
        outcomes = np.array([1, 1, 1])
        result = auroc_selective(confs, outcomes)
        assert np.isnan(result) or isinstance(result, float)

    def test_all_wrong_returns_nan_or_handles(self):
        confs = np.array([0.9, 0.8, 0.7])
        outcomes = np.array([0, 0, 0])
        result = auroc_selective(confs, outcomes)
        assert np.isnan(result) or isinstance(result, float)

    def test_hand_computed(self):
        """
        confs    = [0.9, 0.6, 0.4, 0.1]
        outcomes = [1,   0,   1,   0  ]

        Pairs (correct, incorrect):
            (0.9, 0.6): correct > incorrect → +1
            (0.9, 0.1): correct > incorrect → +1
            (0.4, 0.6): correct < incorrect → +0
            (0.4, 0.1): correct > incorrect → +1

        AUROC = 3/4 = 0.75
        """
        confs = np.array([0.9, 0.6, 0.4, 0.1])
        outcomes = np.array([1, 0, 1, 0])
        assert auroc_selective(confs, outcomes) == pytest.approx(0.75)

    def test_bounded_0_1(self):
        rng = np.random.default_rng(99)
        confs = rng.uniform(0, 1, 100)
        outcomes = rng.integers(0, 2, 100)
        result = auroc_selective(confs, outcomes)
        if not np.isnan(result):
            assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        confs = np.array([0.9, 0.1])
        outcomes = np.array([1, 0])
        result = auroc_selective(confs, outcomes)
        assert isinstance(result, (float, np.floating))


# =============================================================================
# AUPRC — selective prediction (precision-recall)
# =============================================================================


class TestAUPRCSelective:
    """AUPRC: important when correct/incorrect classes are imbalanced."""

    def test_perfect_ranking(self):
        confs = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        outcomes = np.array([1, 1, 1, 0, 0, 0])
        assert auprc_selective(confs, outcomes) == pytest.approx(1.0)

    def test_random_baseline(self):
        """Random confidence → AUPRC ≈ base_rate (prevalence of positive class)."""
        rng = np.random.default_rng(42)
        n = 10000
        base_rate = 0.7
        outcomes = (rng.uniform(0, 1, n) < base_rate).astype(int)
        confs = rng.uniform(0, 1, n)
        result = auprc_selective(confs, outcomes)
        assert result == pytest.approx(base_rate, abs=0.05)

    def test_all_correct_returns_nan_or_handles(self):
        confs = np.array([0.9, 0.8])
        outcomes = np.array([1, 1])
        result = auprc_selective(confs, outcomes)
        # All positive → AUPRC should be 1.0 or NaN
        assert np.isnan(result) or result == pytest.approx(1.0)

    def test_all_wrong_returns_nan_or_handles(self):
        confs = np.array([0.9, 0.8])
        outcomes = np.array([0, 0])
        result = auprc_selective(confs, outcomes)
        assert np.isnan(result) or isinstance(result, float)

    def test_bounded_0_1(self):
        rng = np.random.default_rng(77)
        confs = rng.uniform(0, 1, 100)
        outcomes = rng.integers(0, 2, 100)
        result = auprc_selective(confs, outcomes)
        if not np.isnan(result):
            assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        confs = np.array([0.9, 0.1])
        outcomes = np.array([1, 0])
        result = auprc_selective(confs, outcomes)
        assert isinstance(result, (float, np.floating))


# =============================================================================
# Risk-coverage curve
# =============================================================================


class TestRiskCoverageCurve:
    """Risk-coverage: risk at each coverage level when sorted by confidence."""

    def test_returns_dataclass(self):
        confs = np.array([0.9, 0.6, 0.3])
        losses = np.array([0.0, 1.0, 1.0])
        result = risk_coverage_curve(confs, losses)
        assert isinstance(result, RiskCoverageCurve)

    def test_has_required_fields(self):
        confs = np.array([0.9, 0.6, 0.3])
        losses = np.array([0.0, 1.0, 1.0])
        result = risk_coverage_curve(confs, losses)
        assert hasattr(result, "coverages")
        assert hasattr(result, "risks")

    def test_coverage_starts_at_0_ends_at_1(self):
        """Coverages should span from near-0 to 1.0."""
        confs = np.array([0.9, 0.6, 0.3])
        losses = np.array([0.0, 1.0, 1.0])
        result = risk_coverage_curve(confs, losses)
        # First coverage should be 1/n (answering only the top-1)
        assert result.coverages[0] == pytest.approx(1 / 3)
        assert result.coverages[-1] == pytest.approx(1.0)

    def test_coverages_monotonically_increasing(self):
        rng = np.random.default_rng(42)
        confs = rng.uniform(0, 1, 50)
        losses = rng.integers(0, 2, 50).astype(float)
        result = risk_coverage_curve(confs, losses)
        for i in range(len(result.coverages) - 1):
            assert result.coverages[i] <= result.coverages[i + 1]

    def test_hand_computed(self):
        """
        confs  = [0.9, 0.7, 0.5, 0.3]
        losses = [0.0, 0.0, 1.0, 1.0]

        Sorted by descending confidence (already sorted):
        Coverage 1/4: answer [0.9] → risk = 0/1 = 0.0
        Coverage 2/4: answer [0.9, 0.7] → risk = 0/2 = 0.0
        Coverage 3/4: answer [0.9, 0.7, 0.5] → risk = 1/3 ≈ 0.333
        Coverage 4/4: answer all → risk = 2/4 = 0.5
        """
        confs = np.array([0.9, 0.7, 0.5, 0.3])
        losses = np.array([0.0, 0.0, 1.0, 1.0])
        result = risk_coverage_curve(confs, losses)

        assert len(result.coverages) == 4
        assert result.coverages[0] == pytest.approx(0.25)
        assert result.coverages[1] == pytest.approx(0.50)
        assert result.coverages[2] == pytest.approx(0.75)
        assert result.coverages[3] == pytest.approx(1.00)

        assert result.risks[0] == pytest.approx(0.0)
        assert result.risks[1] == pytest.approx(0.0)
        assert result.risks[2] == pytest.approx(1 / 3)
        assert result.risks[3] == pytest.approx(0.5)

    def test_perfect_confidence_gives_monotonic_risk(self):
        """If confidence perfectly separates correct/wrong, risk should increase."""
        confs = np.array([0.99, 0.95, 0.90, 0.10, 0.05, 0.01])
        losses = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        result = risk_coverage_curve(confs, losses)
        # Risk should be non-decreasing
        for i in range(len(result.risks) - 1):
            assert result.risks[i] <= result.risks[i + 1] + 1e-12

    def test_unsorted_input(self):
        """Should work even if input is not pre-sorted by confidence."""
        confs = np.array([0.3, 0.9, 0.5, 0.7])
        losses = np.array([1.0, 0.0, 1.0, 0.0])
        result = risk_coverage_curve(confs, losses)
        # Same as sorted case: answer order is [0.9, 0.7, 0.5, 0.3]
        assert result.risks[0] == pytest.approx(0.0)
        assert result.risks[-1] == pytest.approx(0.5)

    def test_single_sample(self):
        confs = np.array([0.8])
        losses = np.array([0.0])
        result = risk_coverage_curve(confs, losses)
        assert len(result.coverages) == 1
        assert result.coverages[0] == pytest.approx(1.0)
        assert result.risks[0] == pytest.approx(0.0)

    def test_all_correct(self):
        """All losses = 0 → risk = 0 at all coverages."""
        confs = np.array([0.9, 0.7, 0.5])
        losses = np.zeros(3)
        result = risk_coverage_curve(confs, losses)
        assert np.all(result.risks == pytest.approx(0.0))

    def test_all_wrong(self):
        """All losses = 1 → risk = 1 at all coverages."""
        confs = np.array([0.9, 0.7, 0.5])
        losses = np.ones(3)
        result = risk_coverage_curve(confs, losses)
        assert np.all(result.risks == pytest.approx(1.0))

    def test_fractional_losses(self):
        """Losses can be fractional (e.g. 1 - F1)."""
        confs = np.array([0.9, 0.5, 0.1])
        losses = np.array([0.1, 0.5, 0.9])
        result = risk_coverage_curve(confs, losses)
        assert result.risks[0] == pytest.approx(0.1)
        assert result.risks[-1] == pytest.approx(0.5)  # mean(0.1, 0.5, 0.9)


# =============================================================================
# AURC — Area Under Risk-Coverage Curve
# =============================================================================


class TestAURC:
    """AURC: single scalar summary of selective prediction quality."""

    def test_perfect_confidence_zero_aurc(self):
        """
        Perfect: high-confidence samples are all correct.
        confs  = [0.9, 0.8, 0.7, 0.1, 0.05]
        losses = [0.0, 0.0, 0.0, 1.0, 1.0]

        Risk-coverage: [0, 0, 0, 0.25, 0.4]
        AURC = area under this curve.

        Trapezoidal: coverages = [0.2, 0.4, 0.6, 0.8, 1.0]
        Segments:
            [0.2, 0.4]: (0+0)/2 * 0.2 = 0.0
            [0.4, 0.6]: (0+0)/2 * 0.2 = 0.0
            [0.6, 0.8]: (0+0.25)/2 * 0.2 = 0.025
            [0.8, 1.0]: (0.25+0.4)/2 * 0.2 = 0.065
        AURC = 0.025 + 0.065 = 0.09
        """
        confs = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
        losses = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        result = aurc(confs, losses)
        assert result == pytest.approx(0.09, abs=1e-10)

    def test_all_correct_zero_aurc(self):
        """No losses → AURC = 0."""
        confs = np.array([0.9, 0.7, 0.5])
        losses = np.zeros(3)
        assert aurc(confs, losses) == pytest.approx(0.0)

    def test_all_wrong_max_aurc(self):
        """All losses = 1 → risk = 1 everywhere → AURC ≈ 1.
        
        With n=3: coverages = [1/3, 2/3, 1]
        risks = [1, 1, 1]
        Trap area = (1+1)/2 * 1/3 + (1+1)/2 * 1/3 = 1/3 + 1/3 = 2/3
        But we also need to include area from 0 to 1/3.
        
        Actually, the curve starts at coverage=1/n, not 0. 
        The AURC integrates from the first coverage to 1.
        """
        confs = np.array([0.9, 0.7, 0.5])
        losses = np.ones(3)
        result = aurc(confs, losses)
        # Risk is 1 everywhere, coverage goes from 1/3 to 1
        # Trap area = 1.0 * (1 - 1/3) = 2/3
        assert result == pytest.approx(2 / 3, abs=1e-10)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        confs = rng.uniform(0, 1, 100)
        losses = rng.integers(0, 2, 100).astype(float)
        assert aurc(confs, losses) >= 0.0

    def test_returns_float(self):
        confs = np.array([0.9, 0.1])
        losses = np.array([0.0, 1.0])
        result = aurc(confs, losses)
        assert isinstance(result, (float, np.floating))

    def test_worse_confidence_higher_aurc(self):
        """Inverting confidence ranking should increase AURC."""
        losses = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        good_confs = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        bad_confs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert aurc(good_confs, losses) < aurc(bad_confs, losses)


# =============================================================================
# Oracle AURC
# =============================================================================


class TestOracleAURC:
    """Oracle AURC: optimal risk-coverage (sort by actual loss)."""

    def test_all_correct(self):
        """No losses → oracle AURC = 0."""
        losses = np.zeros(5)
        assert oracle_aurc(losses) == pytest.approx(0.0)

    def test_all_wrong(self):
        """All wrong → oracle still has risk=1 everywhere."""
        losses = np.ones(3)
        result = oracle_aurc(losses)
        assert result == pytest.approx(2 / 3, abs=1e-10)

    def test_hand_computed(self):
        """
        losses = [0, 0, 0, 1, 1]  (3 correct, 2 wrong)
        Oracle ordering: answer correct first → [0, 0, 0, 1, 1]

        coverages = [0.2, 0.4, 0.6, 0.8, 1.0]
        risks     = [0.0, 0.0, 0.0, 0.25, 0.4]

        Trap:
            [0.2, 0.4]: 0 * 0.2 = 0
            [0.4, 0.6]: 0 * 0.2 = 0
            [0.6, 0.8]: (0+0.25)/2 * 0.2 = 0.025
            [0.8, 1.0]: (0.25+0.4)/2 * 0.2 = 0.065
        Oracle AURC = 0.09
        """
        losses = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        result = oracle_aurc(losses)
        assert result == pytest.approx(0.09, abs=1e-10)

    def test_oracle_leq_any_aurc(self):
        """Oracle AURC should be <= AURC for any confidence ranking."""
        rng = np.random.default_rng(42)
        losses = rng.integers(0, 2, 100).astype(float)
        confs = rng.uniform(0, 1, 100)
        assert oracle_aurc(losses) <= aurc(confs, losses) + 1e-12

    def test_nonnegative(self):
        losses = np.array([0.0, 1.0, 0.0, 1.0])
        assert oracle_aurc(losses) >= 0.0


# =============================================================================
# E-AURC — Excess AURC (normalized for task difficulty)
# =============================================================================


class TestEAURC:
    """E-AURC = AURC - oracle_AURC. Measures confidence quality."""

    def test_perfect_confidence_zero_eaurc(self):
        """If confidence perfectly ranks correct above incorrect, E-AURC = 0."""
        confs = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
        losses = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        result = e_aurc(confs, losses)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_all_correct_zero_eaurc(self):
        """No losses → E-AURC = 0 (trivially)."""
        confs = np.array([0.9, 0.5, 0.1])
        losses = np.zeros(3)
        assert e_aurc(confs, losses) == pytest.approx(0.0)

    def test_nonnegative(self):
        """E-AURC >= 0 always (AURC >= oracle_AURC)."""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0, 1, 200)
        losses = rng.integers(0, 2, 200).astype(float)
        result = e_aurc(confs, losses)
        assert result >= -1e-12  # allow tiny numerical error

    def test_bad_confidence_positive_eaurc(self):
        """Inverted confidence → positive E-AURC."""
        confs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        losses = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        result = e_aurc(confs, losses)
        assert result > 0.0

    def test_eaurc_equals_aurc_minus_oracle(self):
        """Verify E-AURC = AURC - oracle_AURC explicitly."""
        rng = np.random.default_rng(77)
        confs = rng.uniform(0, 1, 50)
        losses = rng.integers(0, 2, 50).astype(float)
        expected = aurc(confs, losses) - oracle_aurc(losses)
        result = e_aurc(confs, losses)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_returns_float(self):
        confs = np.array([0.9, 0.1])
        losses = np.array([0.0, 1.0])
        result = e_aurc(confs, losses)
        assert isinstance(result, (float, np.floating))

    def test_normalizes_for_difficulty(self):
        """
        Two tasks with different base error rates but same confidence quality
        should have similar E-AURC.

        Task A: 80% correct, perfect confidence
        Task B: 50% correct, perfect confidence
        Both have E-AURC = 0 because confidence is perfect.
        """
        # Task A: 80% correct
        confs_a = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.1, 0.05])
        losses_a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=float)

        # Task B: 50% correct
        confs_b = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.1, 0.08, 0.06, 0.04, 0.02])
        losses_b = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)

        # Both should have E-AURC ≈ 0 (perfect ranking)
        assert e_aurc(confs_a, losses_a) == pytest.approx(0.0, abs=1e-10)
        assert e_aurc(confs_b, losses_b) == pytest.approx(0.0, abs=1e-10)

        # But raw AURC differs (task B has higher AURC due to more errors)
        assert aurc(confs_a, losses_a) < aurc(confs_b, losses_b)
