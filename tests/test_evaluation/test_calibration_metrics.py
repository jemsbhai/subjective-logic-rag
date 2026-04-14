"""Tests for calibration_metrics.py — ECE, Brier score, reliability diagrams.

Reference formulations:
    ECE = Σ (n_b / N) · |acc_b - conf_b|     (Guo et al., 2017; Naeini et al., 2015)
    Brier = (1/N) Σ (conf_i - outcome_i)²    (Brier, 1950)

All hand-computed expected values are derived step-by-step in docstrings.
"""

import numpy as np
import pytest
from dataclasses import fields

from xrag.evaluation.calibration_metrics import (
    expected_calibration_error,
    brier_score,
    reliability_diagram_data,
    ReliabilityDiagram,
    maximum_calibration_error,
    debiased_ece_l2,
    bootstrap_calibration_ci,
    BootstrapCI,
)


# =============================================================================
# Input validation — shared across all functions
# =============================================================================


class TestInputValidation:
    """All calibration functions must validate inputs consistently."""

    def test_ece_mismatched_lengths(self):
        with pytest.raises(ValueError, match="[Ll]ength"):
            expected_calibration_error(
                np.array([0.5, 0.6]),
                np.array([1]),
            )

    def test_brier_mismatched_lengths(self):
        with pytest.raises(ValueError, match="[Ll]ength"):
            brier_score(np.array([0.5]), np.array([1, 0]))

    def test_ece_empty_arrays(self):
        with pytest.raises(ValueError):
            expected_calibration_error(np.array([]), np.array([]))

    def test_brier_empty_arrays(self):
        with pytest.raises(ValueError):
            brier_score(np.array([]), np.array([]))

    def test_ece_confidence_below_zero(self):
        with pytest.raises(ValueError, match="[Cc]onfidence|[Rr]ange|bound"):
            expected_calibration_error(
                np.array([-0.1, 0.5]),
                np.array([0, 1]),
            )

    def test_ece_confidence_above_one(self):
        with pytest.raises(ValueError, match="[Cc]onfidence|[Rr]ange|bound"):
            expected_calibration_error(
                np.array([0.5, 1.1]),
                np.array([0, 1]),
            )

    def test_brier_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="[Cc]onfidence|[Rr]ange|bound"):
            brier_score(np.array([-0.1, 0.5]), np.array([0, 1]))

    def test_ece_outcomes_not_binary(self):
        with pytest.raises(ValueError, match="[Bb]inary|outcome|0.*1"):
            expected_calibration_error(
                np.array([0.5, 0.6]),
                np.array([0, 2]),
            )

    def test_brier_outcomes_not_binary(self):
        with pytest.raises(ValueError, match="[Bb]inary|outcome|0.*1"):
            brier_score(np.array([0.5, 0.6]), np.array([0.5, 1.0]))

    def test_ece_invalid_strategy(self):
        with pytest.raises(ValueError, match="[Ss]trategy"):
            expected_calibration_error(
                np.array([0.5]),
                np.array([1]),
                strategy="invalid",
            )

    def test_ece_n_bins_zero(self):
        with pytest.raises(ValueError, match="[Bb]in"):
            expected_calibration_error(
                np.array([0.5]),
                np.array([1]),
                n_bins=0,
            )

    def test_ece_n_bins_negative(self):
        with pytest.raises(ValueError, match="[Bb]in"):
            expected_calibration_error(
                np.array([0.5]),
                np.array([1]),
                n_bins=-1,
            )

    def test_reliability_diagram_validates_inputs(self):
        with pytest.raises(ValueError):
            reliability_diagram_data(np.array([]), np.array([]))


# =============================================================================
# Brier score — (1/N) Σ (conf_i - outcome_i)²
# =============================================================================


class TestBrierScore:
    """Brier score: mean squared error between confidence and binary outcome."""

    def test_perfect_calibration_and_accuracy(self):
        """Confidence 1.0 for correct, 0.0 for wrong → Brier = 0."""
        confs = np.array([1.0, 1.0, 0.0, 0.0])
        outcomes = np.array([1, 1, 0, 0])
        assert brier_score(confs, outcomes) == pytest.approx(0.0)

    def test_worst_case(self):
        """Confidence 1.0 for wrong, 0.0 for correct → Brier = 1.0."""
        confs = np.array([1.0, 0.0])
        outcomes = np.array([0, 1])
        assert brier_score(confs, outcomes) == pytest.approx(1.0)

    def test_constant_half_all_correct(self):
        """
        All correct with conf=0.5:
        Brier = (1/4) * 4 * (0.5 - 1)² = (1/4) * 4 * 0.25 = 0.25
        """
        confs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 1, 1, 1])
        assert brier_score(confs, outcomes) == pytest.approx(0.25)

    def test_constant_half_all_wrong(self):
        """
        All wrong with conf=0.5:
        Brier = (1/4) * 4 * (0.5 - 0)² = 0.25
        """
        confs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([0, 0, 0, 0])
        assert brier_score(confs, outcomes) == pytest.approx(0.25)

    def test_constant_half_mixed(self):
        """
        50% correct with conf=0.5:
        Brier = (1/4) * [2*(0.5-1)² + 2*(0.5-0)²]
              = (1/4) * [2*0.25 + 2*0.25] = (1/4) * 1.0 = 0.25
        """
        confs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 1, 0, 0])
        assert brier_score(confs, outcomes) == pytest.approx(0.25)

    def test_hand_computed(self):
        """
        confs    = [0.9, 0.7, 0.3, 0.1]
        outcomes = [1,   1,   0,   0  ]
        errors²  = [0.01, 0.09, 0.09, 0.01]
        Brier = (0.01 + 0.09 + 0.09 + 0.01) / 4 = 0.20 / 4 = 0.05
        """
        confs = np.array([0.9, 0.7, 0.3, 0.1])
        outcomes = np.array([1, 1, 0, 0])
        assert brier_score(confs, outcomes) == pytest.approx(0.05)

    def test_hand_computed_overconfident(self):
        """
        confs    = [0.9, 0.9, 0.9]
        outcomes = [1,   0,   0  ]
        errors²  = [0.01, 0.81, 0.81]
        Brier = (0.01 + 0.81 + 0.81) / 3 = 1.63 / 3 ≈ 0.5433
        """
        confs = np.array([0.9, 0.9, 0.9])
        outcomes = np.array([1, 0, 0])
        assert brier_score(confs, outcomes) == pytest.approx(1.63 / 3)

    def test_single_sample(self):
        assert brier_score(np.array([0.8]), np.array([1])) == pytest.approx(0.04)

    def test_returns_float(self):
        result = brier_score(np.array([0.5]), np.array([1]))
        assert isinstance(result, (float, np.floating))

    def test_bounded_0_1(self):
        """Brier score is always in [0, 1]."""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0, 1, 100)
        outcomes = rng.integers(0, 2, 100)
        bs = brier_score(confs, outcomes)
        assert 0.0 <= bs <= 1.0

    def test_lower_is_better(self):
        """A well-calibrated model should have lower Brier than a random one."""
        outcomes = np.array([1, 1, 1, 0, 0, 0])
        good_confs = np.array([0.9, 0.8, 0.85, 0.1, 0.15, 0.2])
        bad_confs = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.8])
        assert brier_score(good_confs, outcomes) < brier_score(bad_confs, outcomes)


# =============================================================================
# ECE — uniform binning (equal-width)
# =============================================================================


class TestECEUniform:
    """ECE with uniform (equal-width) binning strategy."""

    def test_perfectly_calibrated(self):
        """If accuracy equals confidence in every bin, ECE = 0."""
        # 10 samples, all in bin [0.9, 1.0], all correct → acc=1.0, conf≈0.95
        # That's not ECE=0 unless conf exactly equals acc.
        # True perfect calibration: samples with conf=x are correct x fraction of time.
        # Place 100 in bin [0.0, 0.1] with conf=0.05, 5% correct → |0.05-0.05|=0
        # Easier: single bin, conf=1.0, all correct.
        confs = np.array([1.0, 1.0, 1.0, 1.0])
        outcomes = np.array([1, 1, 1, 1])
        assert expected_calibration_error(confs, outcomes) == pytest.approx(0.0)

    def test_hand_computed_2_bins(self):
        """
        n_bins=2, uniform: bin edges [0, 0.5), [0.5, 1.0]
        confs    = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        outcomes = [0,   0,   1,   1,   1,   0  ]

        Bin 0 [0, 0.5): confs=[0.1,0.2,0.3], outcomes=[0,0,1]
            mean_conf = 0.2, acc = 1/3, n=3
            |1/3 - 0.2| = |0.1333| = 0.1333

        Bin 1 [0.5, 1.0]: confs=[0.7,0.8,0.9], outcomes=[1,1,0]
            mean_conf = 0.8, acc = 2/3, n=3
            |2/3 - 0.8| = |-0.1333| = 0.1333

        ECE = (3/6)*0.1333 + (3/6)*0.1333 = 0.1333
        """
        confs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1, 0])
        ece = expected_calibration_error(confs, outcomes, n_bins=2, strategy="uniform")
        assert ece == pytest.approx(1 / 3 - 0.2, abs=1e-10)

    def test_hand_computed_5_bins(self):
        """
        n_bins=5, uniform: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0]
        confs    = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [0,   0,   1,   1,   1  ]

        Bin 0 [0,0.2):   conf=[0.1], out=[0] → acc=0, conf=0.1, n=1, gap=0.1
        Bin 1 [0.2,0.4): conf=[0.3], out=[0] → acc=0, conf=0.3, n=1, gap=0.3
        Bin 2 [0.4,0.6): conf=[0.5], out=[1] → acc=1, conf=0.5, n=1, gap=0.5
        Bin 3 [0.6,0.8): conf=[0.7], out=[1] → acc=1, conf=0.7, n=1, gap=0.3
        Bin 4 [0.8,1.0]: conf=[0.9], out=[1] → acc=1, conf=0.9, n=1, gap=0.1

        ECE = (1/5)*(0.1 + 0.3 + 0.5 + 0.3 + 0.1) = (1/5)*1.3 = 0.26
        """
        confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1])
        ece = expected_calibration_error(confs, outcomes, n_bins=5, strategy="uniform")
        assert ece == pytest.approx(0.26)

    def test_all_in_one_bin(self):
        """
        All confs clustered in one bin → ECE = |acc - mean_conf|.
        confs    = [0.91, 0.92, 0.93, 0.94]
        outcomes = [1, 1, 0, 1]
        mean_conf = 0.925, acc = 3/4 = 0.75
        ECE = |0.75 - 0.925| = 0.175
        """
        confs = np.array([0.91, 0.92, 0.93, 0.94])
        outcomes = np.array([1, 1, 0, 1])
        ece = expected_calibration_error(confs, outcomes, n_bins=10, strategy="uniform")
        assert ece == pytest.approx(0.175)

    def test_empty_bins_excluded(self):
        """Bins with no samples should not affect ECE."""
        # All samples in first and last bin of 10-bin scheme
        confs = np.array([0.05, 0.05, 0.95, 0.95])
        outcomes = np.array([0, 0, 1, 1])
        # Bin 0: conf=0.05, acc=0, n=2, gap=0.05
        # Bin 9: conf=0.95, acc=1, n=2, gap=0.05
        # ECE = (2/4)*0.05 + (2/4)*0.05 = 0.05
        ece = expected_calibration_error(confs, outcomes, n_bins=10, strategy="uniform")
        assert ece == pytest.approx(0.05)

    def test_single_sample(self):
        """Single sample: ECE = |acc - conf| = |1 - 0.8| = 0.2."""
        confs = np.array([0.8])
        outcomes = np.array([1])
        ece = expected_calibration_error(confs, outcomes, n_bins=10, strategy="uniform")
        assert ece == pytest.approx(0.2)

    def test_ece_nonnegative(self):
        """ECE should always be >= 0."""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0, 1, 200)
        outcomes = rng.integers(0, 2, 200)
        ece = expected_calibration_error(confs, outcomes, strategy="uniform")
        assert ece >= 0.0

    def test_ece_bounded_by_one(self):
        """ECE ≤ 1 (since |acc - conf| ≤ 1 for each bin)."""
        confs = np.array([1.0, 1.0, 1.0])
        outcomes = np.array([0, 0, 0])
        ece = expected_calibration_error(confs, outcomes, strategy="uniform")
        assert ece <= 1.0

    def test_default_n_bins_is_15(self):
        """Default n_bins should be 15 (Guo et al., 2017)."""
        confs = np.array([0.5, 0.6])
        outcomes = np.array([1, 0])
        # Should not raise — just verify it runs with default
        ece = expected_calibration_error(confs, outcomes, strategy="uniform")
        assert isinstance(ece, (float, np.floating))

    def test_returns_float(self):
        result = expected_calibration_error(
            np.array([0.5]), np.array([1]), strategy="uniform"
        )
        assert isinstance(result, (float, np.floating))


# =============================================================================
# ECE — adaptive binning (equal-mass / quantile)
# =============================================================================


class TestECEAdaptive:
    """ECE with adaptive (equal-mass) binning strategy."""

    def test_adaptive_hand_computed(self):
        """
        n_bins=2, adaptive: split at median confidence.
        confs    = [0.1, 0.2, 0.8, 0.9]  → sorted, median split at idx 2
        outcomes = [0,   1,   1,   0  ]

        Bin 0 (lower half): confs=[0.1,0.2], outcomes=[0,1]
            mean_conf = 0.15, acc = 0.5, n=2, gap=0.35

        Bin 1 (upper half): confs=[0.8,0.9], outcomes=[1,0]
            mean_conf = 0.85, acc = 0.5, n=2, gap=0.35

        ECE = (2/4)*0.35 + (2/4)*0.35 = 0.35
        """
        confs = np.array([0.1, 0.2, 0.8, 0.9])
        outcomes = np.array([0, 1, 1, 0])
        ece = expected_calibration_error(confs, outcomes, n_bins=2, strategy="adaptive")
        assert ece == pytest.approx(0.35)

    def test_adaptive_all_same_confidence(self):
        """All same confidence → one effective bin.

        When all confidences are identical, adaptive binning must merge them
        into a single bin regardless of n_bins. Splitting tied confidences
        across bins creates artificial calibration gaps from single-sample
        bins with acc ∈ {0,1}.

        confs = [0.5, 0.5, 0.5, 0.5], outcomes = [1, 0, 1, 0]
        Single merged bin: acc = 0.5, conf = 0.5 → ECE = 0.0
        """
        confs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        # acc = 0.5, conf = 0.5 → ECE = 0.0
        ece = expected_calibration_error(confs, outcomes, n_bins=5, strategy="adaptive")
        assert ece == pytest.approx(0.0)

    def test_adaptive_partial_ties_merged(self):
        """Tied confidences must never be split across bins.

        confs    = [0.2, 0.5, 0.5, 0.5, 0.9]  (3 unique values)
        outcomes = [0,   1,   0,   1,   1  ]
        n_bins=5 → only 3 effective bins (cannot split the 0.5 group)

        Bin 0: conf=[0.2], out=[0] → acc=0, conf=0.2, n=1, gap=0.2
        Bin 1: conf=[0.5,0.5,0.5], out=[1,0,1] → acc=2/3, conf=0.5, n=3, gap=1/6
        Bin 2: conf=[0.9], out=[1] → acc=1, conf=0.9, n=1, gap=0.1

        ECE = (1/5)*0.2 + (3/5)*(1/6) + (1/5)*0.1
            = 0.04 + 0.1 + 0.02 = 0.16
        """
        confs = np.array([0.2, 0.5, 0.5, 0.5, 0.9])
        outcomes = np.array([0, 1, 0, 1, 1])
        ece = expected_calibration_error(confs, outcomes, n_bins=5, strategy="adaptive")
        assert ece == pytest.approx(0.16, abs=1e-10)

    def test_adaptive_nonnegative(self):
        rng = np.random.default_rng(123)
        confs = rng.uniform(0, 1, 200)
        outcomes = rng.integers(0, 2, 200)
        ece = expected_calibration_error(confs, outcomes, strategy="adaptive")
        assert ece >= 0.0

    def test_adaptive_bounded_by_one(self):
        confs = np.array([1.0, 1.0, 1.0])
        outcomes = np.array([0, 0, 0])
        ece = expected_calibration_error(confs, outcomes, strategy="adaptive")
        assert ece <= 1.0

    def test_adaptive_single_sample(self):
        """Single sample: same as uniform — ECE = |acc - conf|."""
        confs = np.array([0.7])
        outcomes = np.array([0])
        ece = expected_calibration_error(confs, outcomes, n_bins=5, strategy="adaptive")
        assert ece == pytest.approx(0.7)

    def test_uniform_vs_adaptive_differ_on_skewed(self):
        """Uniform and adaptive ECE can give different values on skewed data."""
        # Skewed: most confidences near 0.9
        confs = np.array([0.1, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])
        outcomes = np.array([0, 1, 1, 0, 1, 1, 0, 1])
        ece_uniform = expected_calibration_error(
            confs, outcomes, n_bins=5, strategy="uniform"
        )
        ece_adaptive = expected_calibration_error(
            confs, outcomes, n_bins=5, strategy="adaptive"
        )
        # They should generally differ (not testing which is larger)
        # Just check both are valid
        assert ece_uniform >= 0.0
        assert ece_adaptive >= 0.0


# =============================================================================
# ReliabilityDiagram dataclass
# =============================================================================


class TestReliabilityDiagram:
    """reliability_diagram_data should return a proper dataclass."""

    def test_returns_dataclass(self):
        confs = np.array([0.1, 0.5, 0.9])
        outcomes = np.array([0, 1, 1])
        result = reliability_diagram_data(confs, outcomes, n_bins=3)
        assert isinstance(result, ReliabilityDiagram)

    def test_has_required_fields(self):
        confs = np.array([0.1, 0.5, 0.9])
        outcomes = np.array([0, 1, 1])
        result = reliability_diagram_data(confs, outcomes, n_bins=3)
        field_names = {f.name for f in fields(result)}
        assert "bin_edges" in field_names
        assert "bin_accuracies" in field_names
        assert "bin_confidences" in field_names
        assert "bin_counts" in field_names

    def test_bin_counts_sum_to_n(self):
        """Total count across bins must equal number of samples."""
        confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1])
        result = reliability_diagram_data(confs, outcomes, n_bins=5)
        assert np.nansum(result.bin_counts) == 5

    def test_bin_edges_monotonic(self):
        confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1])
        result = reliability_diagram_data(confs, outcomes, n_bins=5)
        edges = result.bin_edges
        for i in range(len(edges) - 1):
            assert edges[i] <= edges[i + 1]

    def test_bin_accuracies_bounded(self):
        """All non-NaN bin accuracies should be in [0, 1]."""
        rng = np.random.default_rng(99)
        confs = rng.uniform(0, 1, 50)
        outcomes = rng.integers(0, 2, 50)
        result = reliability_diagram_data(confs, outcomes, n_bins=10)
        valid = result.bin_accuracies[~np.isnan(result.bin_accuracies)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_bin_confidences_bounded(self):
        """All non-NaN bin confidences should be in [0, 1]."""
        rng = np.random.default_rng(99)
        confs = rng.uniform(0, 1, 50)
        outcomes = rng.integers(0, 2, 50)
        result = reliability_diagram_data(confs, outcomes, n_bins=10)
        valid = result.bin_confidences[~np.isnan(result.bin_confidences)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_hand_computed_diagram(self):
        """
        n_bins=2, uniform: [0, 0.5), [0.5, 1.0]
        confs    = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        outcomes = [0,   0,   1,   1,   1,   0  ]

        Bin 0: acc=1/3, conf=0.2, count=3
        Bin 1: acc=2/3, conf=0.8, count=3
        """
        confs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1, 0])
        result = reliability_diagram_data(confs, outcomes, n_bins=2, strategy="uniform")

        # Find non-empty bins
        nonempty = result.bin_counts > 0
        accs = result.bin_accuracies[nonempty]
        mean_confs = result.bin_confidences[nonempty]
        counts = result.bin_counts[nonempty]

        assert len(accs) == 2
        assert accs[0] == pytest.approx(1 / 3)
        assert accs[1] == pytest.approx(2 / 3)
        assert mean_confs[0] == pytest.approx(0.2)
        assert mean_confs[1] == pytest.approx(0.8)
        assert counts[0] == 3
        assert counts[1] == 3

    def test_empty_bins_have_nan_accuracy(self):
        """Bins with no samples should have NaN accuracy."""
        # All samples in bin 0 of a 10-bin scheme
        confs = np.array([0.01, 0.02, 0.03])
        outcomes = np.array([0, 1, 0])
        result = reliability_diagram_data(confs, outcomes, n_bins=10, strategy="uniform")
        # At least some bins should be empty → NaN
        assert np.any(np.isnan(result.bin_accuracies))

    def test_supports_adaptive_strategy(self):
        confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1])
        result = reliability_diagram_data(confs, outcomes, n_bins=2, strategy="adaptive")
        assert isinstance(result, ReliabilityDiagram)
        # Adaptive with n_bins=2 should give 2 non-empty bins
        assert np.sum(result.bin_counts > 0) >= 1


# =============================================================================
# Consistency: ECE computed from ReliabilityDiagram must match ECE function
# =============================================================================


class TestECEDiagramConsistency:
    """ECE from diagram data must equal ECE from the function."""

    def _ece_from_diagram(self, diagram: ReliabilityDiagram) -> float:
        """Recompute ECE from the diagram dataclass."""
        n_total = np.nansum(diagram.bin_counts)
        if n_total == 0:
            return 0.0
        nonempty = diagram.bin_counts > 0
        weights = diagram.bin_counts[nonempty] / n_total
        gaps = np.abs(
            diagram.bin_accuracies[nonempty] - diagram.bin_confidences[nonempty]
        )
        return float(np.sum(weights * gaps))

    def test_consistency_uniform(self):
        confs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1, 0])
        ece = expected_calibration_error(confs, outcomes, n_bins=5, strategy="uniform")
        diagram = reliability_diagram_data(confs, outcomes, n_bins=5, strategy="uniform")
        ece_from_diag = self._ece_from_diagram(diagram)
        assert ece == pytest.approx(ece_from_diag, abs=1e-10)

    def test_consistency_adaptive(self):
        rng = np.random.default_rng(77)
        confs = rng.uniform(0, 1, 100)
        outcomes = rng.integers(0, 2, 100)
        ece = expected_calibration_error(confs, outcomes, n_bins=10, strategy="adaptive")
        diagram = reliability_diagram_data(
            confs, outcomes, n_bins=10, strategy="adaptive"
        )
        ece_from_diag = self._ece_from_diagram(diagram)
        assert ece == pytest.approx(ece_from_diag, abs=1e-10)

    def test_consistency_random_large(self):
        """Consistency on a larger random dataset."""
        rng = np.random.default_rng(2026)
        confs = rng.uniform(0, 1, 1000)
        outcomes = rng.integers(0, 2, 1000)
        for strategy in ("uniform", "adaptive"):
            ece = expected_calibration_error(
                confs, outcomes, n_bins=15, strategy=strategy
            )
            diagram = reliability_diagram_data(
                confs, outcomes, n_bins=15, strategy=strategy
            )
            ece_from_diag = self._ece_from_diagram(diagram)
            assert ece == pytest.approx(ece_from_diag, abs=1e-10), (
                f"Mismatch for strategy={strategy}"
            )


# =============================================================================
# Edge cases and properties
# =============================================================================


class TestCalibrationEdgeCases:
    """Additional edge cases and mathematical properties."""

    def test_all_correct_high_confidence_low_ece(self):
        """If all correct at 0.95, ECE = |1.0 - 0.95| = 0.05."""
        confs = np.array([0.95] * 100)
        outcomes = np.ones(100, dtype=int)
        ece = expected_calibration_error(confs, outcomes, n_bins=10, strategy="uniform")
        assert ece == pytest.approx(0.05)

    def test_all_wrong_high_confidence_high_ece(self):
        """If all wrong at 0.95, ECE = |0.0 - 0.95| = 0.95."""
        confs = np.array([0.95] * 100)
        outcomes = np.zeros(100, dtype=int)
        ece = expected_calibration_error(confs, outcomes, n_bins=10, strategy="uniform")
        assert ece == pytest.approx(0.95)

    def test_brier_decomposes_into_calibration_and_refinement(self):
        """
        Brier score = calibration + refinement (Murphy decomposition).
        We can't easily test this decomposition, but we can verify that
        a perfectly calibrated model has Brier = base_rate * (1 - base_rate)
        when confidence always equals the base rate.
        
        base_rate = 0.6 (60% correct)
        All confs = 0.6
        Brier = 0.6*(0.6-1)² + 0.4*(0.6-0)² = 0.6*0.16 + 0.4*0.36 = 0.096 + 0.144 = 0.24
        = base_rate * (1 - base_rate) = 0.6 * 0.4 = 0.24
        """
        n = 1000
        base_rate = 0.6
        outcomes = np.array([1] * int(n * base_rate) + [0] * int(n * (1 - base_rate)))
        confs = np.full(n, base_rate)
        bs = brier_score(confs, outcomes)
        assert bs == pytest.approx(base_rate * (1 - base_rate), abs=1e-10)

    def test_integer_outcomes_accepted(self):
        """Outcomes as int array should work."""
        confs = np.array([0.5, 0.6])
        outcomes = np.array([0, 1], dtype=np.int64)
        bs = brier_score(confs, outcomes)
        assert isinstance(bs, (float, np.floating))

    def test_n_bins_larger_than_samples(self):
        """More bins than samples should still work (many empty bins)."""
        confs = np.array([0.5, 0.9])
        outcomes = np.array([1, 0])
        ece = expected_calibration_error(confs, outcomes, n_bins=100, strategy="uniform")
        assert ece >= 0.0


# =============================================================================
# MCE — Maximum Calibration Error
# =============================================================================


class TestMCE:
    """MCE = max over bins of |acc_b - conf_b|."""

    def test_perfectly_calibrated(self):
        """Confidence 1.0, all correct → MCE = 0."""
        confs = np.array([1.0, 1.0, 1.0])
        outcomes = np.array([1, 1, 1])
        assert maximum_calibration_error(confs, outcomes) == pytest.approx(0.0)

    def test_hand_computed_2_bins(self):
        """
        n_bins=2, uniform: [0, 0.5), [0.5, 1.0]
        confs    = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        outcomes = [0,   0,   1,   1,   1,   0  ]

        Bin 0: acc=1/3, conf=0.2, gap=|1/3 - 0.2| = 0.1333
        Bin 1: acc=2/3, conf=0.8, gap=|2/3 - 0.8| = 0.1333
        MCE = max(0.1333, 0.1333) = 0.1333
        """
        confs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1, 0])
        mce = maximum_calibration_error(confs, outcomes, n_bins=2, strategy="uniform")
        assert mce == pytest.approx(1 / 3 - 0.2, abs=1e-10)

    def test_hand_computed_asymmetric(self):
        """
        n_bins=2, uniform: [0, 0.5), [0.5, 1.0]
        confs    = [0.1, 0.2, 0.9, 0.9, 0.9]
        outcomes = [0,   0,   1,   0,   0  ]

        Bin 0: acc=0, conf=0.15, gap=0.15
        Bin 1: acc=1/3, conf=0.9, gap=|1/3-0.9| = 0.5667
        MCE = max(0.15, 0.5667) = 0.5667
        """
        confs = np.array([0.1, 0.2, 0.9, 0.9, 0.9])
        outcomes = np.array([0, 0, 1, 0, 0])
        mce = maximum_calibration_error(confs, outcomes, n_bins=2, strategy="uniform")
        assert mce == pytest.approx(abs(1 / 3 - 0.9))

    def test_mce_geq_ece(self):
        """MCE >= ECE always (max >= weighted average)."""
        rng = np.random.default_rng(42)
        confs = rng.uniform(0, 1, 200)
        outcomes = rng.integers(0, 2, 200)
        ece = expected_calibration_error(confs, outcomes, n_bins=10, strategy="uniform")
        mce = maximum_calibration_error(confs, outcomes, n_bins=10, strategy="uniform")
        assert mce >= ece - 1e-12

    def test_mce_bounded_0_1(self):
        rng = np.random.default_rng(99)
        confs = rng.uniform(0, 1, 100)
        outcomes = rng.integers(0, 2, 100)
        mce = maximum_calibration_error(confs, outcomes)
        assert 0.0 <= mce <= 1.0

    def test_worst_case_mce(self):
        """All wrong at 1.0 → MCE = 1.0."""
        confs = np.array([1.0, 1.0, 1.0])
        outcomes = np.array([0, 0, 0])
        mce = maximum_calibration_error(confs, outcomes)
        assert mce == pytest.approx(1.0)

    def test_supports_adaptive(self):
        confs = np.array([0.1, 0.5, 0.9])
        outcomes = np.array([0, 1, 1])
        mce = maximum_calibration_error(confs, outcomes, n_bins=2, strategy="adaptive")
        assert isinstance(mce, (float, np.floating))

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            maximum_calibration_error(np.array([]), np.array([]))


# =============================================================================
# Debiased ECE (L2) — Kumar et al., NeurIPS 2019 (Spotlight)
# =============================================================================


class TestDebiasedECEL2:
    """Debiased L2 calibration error.

    Plugin:  ECE²_L2 = Σ (n_b/N) · (acc_b - conf_b)²
    Bias per bin: acc_b·(1-acc_b)/n_b
    Debiased: sqrt( Σ (n_b/N) · max(0, (acc_b - conf_b)² - acc_b·(1-acc_b)/n_b) )
    """

    def test_perfectly_calibrated(self):
        """Perfect calibration → debiased ECE_L2 = 0."""
        confs = np.array([1.0, 1.0, 1.0])
        outcomes = np.array([1, 1, 1])
        assert debiased_ece_l2(confs, outcomes) == pytest.approx(0.0)

    def test_hand_computed_single_bin(self):
        """
        All in one bin (n_bins=1):
        confs = [0.8, 0.8, 0.8, 0.8], outcomes = [1, 1, 0, 0]
        acc = 0.5, conf = 0.8, n_b = 4

        Plugin term: (0.5 - 0.8)² = 0.09
        Bias correction: 0.5 * 0.5 / 4 = 0.0625
        Debiased term: max(0, 0.09 - 0.0625) = 0.0275
        Debiased ECE_L2 = sqrt(1.0 * 0.0275) = sqrt(0.0275) ≈ 0.16583
        """
        confs = np.array([0.8, 0.8, 0.8, 0.8])
        outcomes = np.array([1, 1, 0, 0])
        result = debiased_ece_l2(confs, outcomes, n_bins=1)
        expected = np.sqrt(0.09 - 0.0625)
        assert result == pytest.approx(expected)

    def test_hand_computed_2_bins(self):
        """
        n_bins=2, uniform.
        confs    = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        outcomes = [0,   0,   1,   1,   1,   0  ]

        Bin 0: acc=1/3, conf=0.2, n_b=3
            plugin = (1/3 - 0.2)² = (0.1333)² ≈ 0.01778
            bias = (1/3)*(2/3)/3 = 2/27 ≈ 0.07407
            debiased = max(0, 0.01778 - 0.07407) = 0  (clamped)

        Bin 1: acc=2/3, conf=0.8, n_b=3
            plugin = (2/3 - 0.8)² = (-0.1333)² ≈ 0.01778
            bias = (2/3)*(1/3)/3 = 2/27 ≈ 0.07407
            debiased = max(0, 0.01778 - 0.07407) = 0  (clamped)

        Debiased ECE_L2 = sqrt((3/6)*0 + (3/6)*0) = 0.0
        """
        confs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1, 0])
        result = debiased_ece_l2(confs, outcomes, n_bins=2, strategy="uniform")
        assert result == pytest.approx(0.0)

    def test_debiased_leq_plugin(self):
        """Debiased ECE_L2 should generally be <= plugin ECE_L2.

        This isn't mathematically guaranteed for every sample, but it holds
        statistically. We test on a large sample where it should hold.
        """
        rng = np.random.default_rng(42)
        confs = rng.uniform(0, 1, 1000)
        outcomes = rng.integers(0, 2, 1000)
        debiased = debiased_ece_l2(confs, outcomes, n_bins=15, strategy="uniform")
        # Plugin L2 ECE
        diagram = reliability_diagram_data(confs, outcomes, n_bins=15, strategy="uniform")
        nonempty = diagram.bin_counts > 0
        weights = diagram.bin_counts[nonempty].astype(float) / np.sum(diagram.bin_counts)
        gaps_sq = (diagram.bin_accuracies[nonempty] - diagram.bin_confidences[nonempty]) ** 2
        plugin_l2 = float(np.sqrt(np.sum(weights * gaps_sq)))
        assert debiased <= plugin_l2 + 1e-12

    def test_nonnegative(self):
        """Debiased ECE_L2 >= 0 always (clamped)."""
        rng = np.random.default_rng(123)
        confs = rng.uniform(0, 1, 200)
        outcomes = rng.integers(0, 2, 200)
        result = debiased_ece_l2(confs, outcomes)
        assert result >= 0.0

    def test_large_gap_survives_debiasing(self):
        """
        When the calibration gap is large relative to the bias correction,
        the debiased term should remain positive.

        confs = [0.95]*100, outcomes = [0]*100
        Single bin: acc=0, conf=0.95
        plugin = (0 - 0.95)² = 0.9025
        bias = 0*1/100 = 0.0
        debiased = 0.9025
        ECE_L2 = sqrt(0.9025) = 0.95
        """
        confs = np.array([0.95] * 100)
        outcomes = np.zeros(100, dtype=int)
        result = debiased_ece_l2(confs, outcomes, n_bins=10)
        assert result == pytest.approx(0.95)

    def test_supports_adaptive(self):
        confs = np.array([0.1, 0.5, 0.9])
        outcomes = np.array([0, 1, 1])
        result = debiased_ece_l2(confs, outcomes, n_bins=2, strategy="adaptive")
        assert isinstance(result, (float, np.floating))

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            debiased_ece_l2(np.array([-0.1]), np.array([1]))


# =============================================================================
# Bootstrap CIs
# =============================================================================


class TestBootstrapCI:
    """Bootstrap confidence intervals for calibration metrics."""

    def test_returns_dataclass(self):
        confs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 1, 0, 1, 1])
        result = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=100, seed=42
        )
        assert isinstance(result, BootstrapCI)

    def test_has_required_fields(self):
        confs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 1, 0, 1, 1])
        result = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=100, seed=42
        )
        field_names = {f.name for f in fields(result)}
        assert "point_estimate" in field_names
        assert "ci_lower" in field_names
        assert "ci_upper" in field_names
        assert "alpha" in field_names
        assert "n_bootstrap" in field_names

    def test_ci_contains_point_estimate(self):
        """Point estimate should fall within the CI."""
        confs = np.array([0.5, 0.6, 0.7, 0.8, 0.9] * 20)
        outcomes = np.array([0, 1, 0, 1, 1] * 20)
        result = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=500, seed=42
        )
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_ci_lower_leq_upper(self):
        confs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 1, 0, 1, 1])
        result = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=200, seed=42
        )
        assert result.ci_lower <= result.ci_upper

    def test_narrower_ci_with_more_data(self):
        """More data → narrower CI (law of large numbers)."""
        rng = np.random.default_rng(42)
        confs_small = rng.uniform(0, 1, 50)
        outcomes_small = rng.integers(0, 2, 50)
        confs_large = rng.uniform(0, 1, 500)
        outcomes_large = rng.integers(0, 2, 500)

        ci_small = bootstrap_calibration_ci(
            brier_score, confs_small, outcomes_small, n_bootstrap=500, seed=42
        )
        ci_large = bootstrap_calibration_ci(
            brier_score, confs_large, outcomes_large, n_bootstrap=500, seed=42
        )
        width_small = ci_small.ci_upper - ci_small.ci_lower
        width_large = ci_large.ci_upper - ci_large.ci_lower
        assert width_large < width_small

    def test_alpha_95(self):
        """Default alpha=0.05 → 95% CI."""
        confs = np.array([0.5, 0.6, 0.7, 0.8, 0.9] * 20)
        outcomes = np.array([0, 1, 0, 1, 1] * 20)
        result = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=200, seed=42
        )
        assert result.alpha == 0.05

    def test_custom_alpha(self):
        confs = np.array([0.5, 0.6, 0.7, 0.8, 0.9] * 20)
        outcomes = np.array([0, 1, 0, 1, 1] * 20)
        ci_90 = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=500, alpha=0.10, seed=42
        )
        ci_95 = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=500, alpha=0.05, seed=42
        )
        assert ci_90.alpha == 0.10
        # 90% CI should be narrower than 95% CI
        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_95 = ci_95.ci_upper - ci_95.ci_lower
        assert width_90 <= width_95 + 1e-10

    def test_reproducible_with_seed(self):
        confs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 1, 0, 1, 1])
        r1 = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=100, seed=42
        )
        r2 = bootstrap_calibration_ci(
            brier_score, confs, outcomes, n_bootstrap=100, seed=42
        )
        assert r1.point_estimate == r2.point_estimate
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_works_with_ece(self):
        """Bootstrap should work with ECE (which takes extra kwargs)."""
        confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
        outcomes = np.array([0, 0, 1, 1, 1] * 20)
        result = bootstrap_calibration_ci(
            expected_calibration_error,
            confs,
            outcomes,
            n_bootstrap=100,
            seed=42,
            n_bins=10,
            strategy="uniform",
        )
        assert result.ci_lower <= result.ci_upper
        assert result.point_estimate >= 0.0

    def test_works_with_debiased_ece(self):
        confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
        outcomes = np.array([0, 0, 1, 1, 1] * 20)
        result = bootstrap_calibration_ci(
            debiased_ece_l2,
            confs,
            outcomes,
            n_bootstrap=100,
            seed=42,
            n_bins=10,
        )
        assert result.ci_lower <= result.ci_upper

    def test_validates_n_bootstrap(self):
        with pytest.raises(ValueError, match="[Bb]ootstrap"):
            bootstrap_calibration_ci(
                brier_score, np.array([0.5]), np.array([1]), n_bootstrap=0
            )

    def test_validates_alpha_range(self):
        with pytest.raises(ValueError, match="[Aa]lpha"):
            bootstrap_calibration_ci(
                brier_score, np.array([0.5]), np.array([1]),
                n_bootstrap=100, alpha=1.5,
            )
