"""Calibration metrics: ECE, Brier score, and reliability diagram data.

Formulations:
    ECE = Σ (n_b / N) · |acc_b - conf_b|     (Naeini et al., 2015; Guo et al., 2017)
    Brier = (1/N) Σ (conf_i - outcome_i)²    (Brier, 1950)

Supports two binning strategies for ECE:
    - "uniform": equal-width bins (standard)
    - "adaptive": equal-mass / quantile bins (robust to skewed confidence distributions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ReliabilityDiagram:
    """Data for plotting a reliability (calibration) diagram.

    Attributes:
        bin_edges: 1D array of bin boundary values (length = n_bins + 1 for
            uniform, or n_bins + 1 for adaptive).
        bin_accuracies: Mean accuracy in each bin. NaN for empty bins.
        bin_confidences: Mean confidence in each bin. NaN for empty bins.
        bin_counts: Number of samples in each bin.
    """

    bin_edges: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bins: int | None = None,
    strategy: str | None = None,
) -> None:
    """Validate inputs shared by all calibration functions.

    Raises:
        ValueError: On any invalid input.
    """
    if confidences.ndim != 1 or binary_outcomes.ndim != 1:
        raise ValueError("confidences and binary_outcomes must be 1-D arrays")

    if len(confidences) == 0 or len(binary_outcomes) == 0:
        raise ValueError("confidences and binary_outcomes must be non-empty")

    if len(confidences) != len(binary_outcomes):
        raise ValueError(
            f"Length mismatch: {len(confidences)} confidences "
            f"vs {len(binary_outcomes)} outcomes"
        )

    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("Confidence values must be in range [0, 1]")

    # Check binary: must be exactly 0 or 1
    unique_vals = np.unique(binary_outcomes)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"Binary outcomes must contain only 0 and 1, got unique values: {unique_vals}"
        )

    if n_bins is not None and n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    if strategy is not None and strategy not in ("uniform", "adaptive"):
        raise ValueError(
            f"Strategy must be 'uniform' or 'adaptive', got '{strategy}'"
        )


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------


def _bin_uniform(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bins: int,
) -> ReliabilityDiagram:
    """Compute reliability diagram data with equal-width (uniform) bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_accuracies = np.full(n_bins, np.nan)
    bin_confidences = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i < n_bins - 1:
            # [lower, upper) for all bins except the last
            mask = (confidences >= lower) & (confidences < upper)
        else:
            # [lower, upper] for the last bin (inclusive upper bound)
            mask = (confidences >= lower) & (confidences <= upper)

        count = int(np.sum(mask))
        bin_counts[i] = count

        if count > 0:
            bin_accuracies[i] = np.mean(binary_outcomes[mask].astype(float))
            bin_confidences[i] = np.mean(confidences[mask])

    return ReliabilityDiagram(
        bin_edges=bin_edges,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


def _bin_adaptive(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bins: int,
) -> ReliabilityDiagram:
    """Compute reliability diagram data with equal-mass (adaptive/quantile) bins.

    Tied confidence values are never split across bins — they form atomic
    groups that must stay together.  When ties reduce the number of valid
    split points below ``n_bins - 1``, fewer bins are returned.
    """
    n = len(confidences)
    sorted_indices = np.argsort(confidences, kind="stable")
    sorted_confs = confidences[sorted_indices]
    sorted_outcomes = binary_outcomes[sorted_indices].astype(float)

    # --- Build atomic tie groups (cannot be split) ---
    tie_groups: list[np.ndarray] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_confs[j] == sorted_confs[i]:
            j += 1
        tie_groups.append(np.arange(i, j))
        i = j

    k = len(tie_groups)
    effective_bins = min(n_bins, k)

    # --- Find optimal split points among group boundaries ---
    # Ideal cumulative split points: we want each bin to hold n/effective_bins samples.
    # We have effective_bins - 1 splits to place at group boundaries.
    group_sizes = np.array([len(g) for g in tie_groups])
    cum_sizes = np.cumsum(group_sizes)
    ideal_splits = [i * n / effective_bins for i in range(1, effective_bins)]

    bins: list[np.ndarray] = []
    start_group = 0
    for split in ideal_splits:
        # Find group boundary closest to this split
        best_idx = start_group
        best_dist = abs(cum_sizes[start_group] - split)
        for j in range(start_group + 1, k):
            dist = abs(cum_sizes[j] - split)
            if dist <= best_dist:
                best_dist = dist
                best_idx = j
            else:
                break  # cumulative sizes are monotonic; distances increasing
        # Bin covers groups [start_group, best_idx] inclusive
        bin_indices = np.concatenate(tie_groups[start_group : best_idx + 1])
        bins.append(bin_indices)
        start_group = best_idx + 1

    # Last bin gets everything remaining
    if start_group < k:
        bins.append(np.concatenate(tie_groups[start_group:]))

    # --- Compute per-bin statistics ---
    actual_bins = len(bins)
    bin_accuracies = np.full(actual_bins, np.nan)
    bin_confidences = np.full(actual_bins, np.nan)
    bin_counts = np.zeros(actual_bins, dtype=int)

    bin_edges_list: list[float] = []
    for b_idx, b_indices in enumerate(bins):
        bin_counts[b_idx] = len(b_indices)
        bin_accuracies[b_idx] = np.mean(sorted_outcomes[b_indices])
        bin_confidences[b_idx] = np.mean(sorted_confs[b_indices])
        bin_edges_list.append(float(sorted_confs[b_indices[0]]))
    # Upper edge of last bin
    bin_edges_list.append(float(sorted_confs[bins[-1][-1]]))
    bin_edges = np.array(bin_edges_list)

    return ReliabilityDiagram(
        bin_edges=bin_edges,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _ece_from_diagram(diagram: ReliabilityDiagram) -> float:
    """Compute ECE from a ReliabilityDiagram."""
    n_total = int(np.sum(diagram.bin_counts))
    if n_total == 0:
        return 0.0

    nonempty = diagram.bin_counts > 0
    weights = diagram.bin_counts[nonempty].astype(float) / n_total
    gaps = np.abs(diagram.bin_accuracies[nonempty] - diagram.bin_confidences[nonempty])
    return float(np.sum(weights * gaps))


def expected_calibration_error(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error.

    ECE = Σ (n_b / N) · |acc_b - conf_b|

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        binary_outcomes: 1D array of binary outcomes in {0, 1}.
        n_bins: Number of bins (default 15, per Guo et al. 2017).
        strategy: Binning strategy — "uniform" (equal-width) or
            "adaptive" (equal-mass / quantile).

    Returns:
        ECE value (float, non-negative).

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_inputs(confidences, binary_outcomes, n_bins=n_bins, strategy=strategy)
    diagram = reliability_diagram_data(
        confidences, binary_outcomes, n_bins=n_bins, strategy=strategy,
        _skip_validation=True,
    )
    return _ece_from_diagram(diagram)


def brier_score(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
) -> float:
    """Compute Brier score: mean squared error between confidence and outcome.

    BS = (1/N) Σ (conf_i - outcome_i)²

    Lower is better. Range: [0, 1].

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        binary_outcomes: 1D array of binary outcomes in {0, 1}.

    Returns:
        Brier score (float).

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_inputs(confidences, binary_outcomes)
    return float(np.mean((confidences - binary_outcomes.astype(float)) ** 2))


def reliability_diagram_data(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
    *,
    _skip_validation: bool = False,
) -> ReliabilityDiagram:
    """Compute data for a reliability (calibration) diagram.

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        binary_outcomes: 1D array of binary outcomes in {0, 1}.
        n_bins: Number of bins (default 15).
        strategy: "uniform" (equal-width) or "adaptive" (equal-mass).

    Returns:
        ReliabilityDiagram dataclass with bin data.

    Raises:
        ValueError: On invalid inputs.
    """
    if not _skip_validation:
        _validate_inputs(confidences, binary_outcomes, n_bins=n_bins, strategy=strategy)

    if strategy == "uniform":
        return _bin_uniform(confidences, binary_outcomes, n_bins)
    else:  # adaptive
        return _bin_adaptive(confidences, binary_outcomes, n_bins)


def maximum_calibration_error(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """Compute Maximum Calibration Error.

    MCE = max_b |acc_b - conf_b|

    Captures worst-case miscalibration across bins.

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        binary_outcomes: 1D array of binary outcomes in {0, 1}.
        n_bins: Number of bins (default 15).
        strategy: "uniform" or "adaptive".

    Returns:
        MCE value (float, non-negative).
    """
    _validate_inputs(confidences, binary_outcomes, n_bins=n_bins, strategy=strategy)
    diagram = reliability_diagram_data(
        confidences, binary_outcomes, n_bins=n_bins, strategy=strategy,
        _skip_validation=True,
    )
    nonempty = diagram.bin_counts > 0
    if not np.any(nonempty):
        return 0.0
    gaps = np.abs(diagram.bin_accuracies[nonempty] - diagram.bin_confidences[nonempty])
    return float(np.max(gaps))


def debiased_ece_l2(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """Compute debiased L2 calibration error (Kumar et al., NeurIPS 2019).

    The plugin estimator ECE²_L2 = Σ (n_b/N) · (acc_b - conf_b)² has positive
    bias due to finite-sample variance in per-bin accuracy estimates.  The
    debiased estimator subtracts the estimated bias acc_b·(1-acc_b)/n_b from
    each squared term, clamping to zero.

    Returns sqrt of the debiased sum for comparability with ECE units.

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        binary_outcomes: 1D array of binary outcomes in {0, 1}.
        n_bins: Number of bins (default 15).
        strategy: "uniform" or "adaptive".

    Returns:
        Debiased ECE_L2 (float, non-negative).

    References:
        Kumar, Liang, Ma. "Verified Uncertainty Calibration." NeurIPS 2019.
    """
    _validate_inputs(confidences, binary_outcomes, n_bins=n_bins, strategy=strategy)
    diagram = reliability_diagram_data(
        confidences, binary_outcomes, n_bins=n_bins, strategy=strategy,
        _skip_validation=True,
    )
    n_total = int(np.sum(diagram.bin_counts))
    if n_total == 0:
        return 0.0

    nonempty = diagram.bin_counts > 0
    weights = diagram.bin_counts[nonempty].astype(float) / n_total
    accs = diagram.bin_accuracies[nonempty]
    confs = diagram.bin_confidences[nonempty]
    counts = diagram.bin_counts[nonempty].astype(float)

    # Plugin squared gaps
    gaps_sq = (accs - confs) ** 2

    # Bias correction: Var[acc_b] ≈ acc_b * (1 - acc_b) / n_b
    bias_correction = accs * (1.0 - accs) / counts

    # Debiased per-bin term, clamped to non-negative
    debiased_terms = np.maximum(0.0, gaps_sq - bias_correction)

    # Weighted sum, then sqrt for same units as ECE
    debiased_l2_sq = float(np.sum(weights * debiased_terms))
    return float(np.sqrt(debiased_l2_sq))


@dataclass(frozen=True)
class BootstrapCI:
    """Result of a bootstrap confidence interval computation.

    Attributes:
        point_estimate: Metric computed on the original data.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        alpha: Significance level (e.g. 0.05 for 95% CI).
        n_bootstrap: Number of bootstrap resamples used.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_bootstrap: int


def bootstrap_calibration_ci(
    metric_fn: Callable[..., float],
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    **metric_kwargs: Any,
) -> BootstrapCI:
    """Compute bootstrap percentile confidence interval for a calibration metric.

    Resamples (confidence, outcome) pairs with replacement B times,
    computes the metric on each resample, and returns the percentile CI.

    Args:
        metric_fn: Callable with signature (confidences, binary_outcomes, **kwargs) -> float.
        confidences: 1D array of predicted confidences in [0, 1].
        binary_outcomes: 1D array of binary outcomes in {0, 1}.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.
        **metric_kwargs: Extra keyword arguments passed to metric_fn
            (e.g. n_bins, strategy).

    Returns:
        BootstrapCI with point estimate and percentile interval.

    Raises:
        ValueError: On invalid n_bootstrap or alpha.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

    # Point estimate on original data
    point_estimate = float(metric_fn(confidences, binary_outcomes, **metric_kwargs))

    # Bootstrap resampling
    rng = np.random.default_rng(seed)
    n = len(confidences)
    boot_estimates = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_confs = confidences[indices]
        boot_outcomes = binary_outcomes[indices]
        boot_estimates[i] = metric_fn(boot_confs, boot_outcomes, **metric_kwargs)

    # Percentile CI
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_estimates, lower_pct))
    ci_upper = float(np.percentile(boot_estimates, upper_pct))

    return BootstrapCI(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )
