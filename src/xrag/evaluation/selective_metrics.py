"""Selective prediction metrics: AUROC, AUPRC, risk-coverage, AURC, E-AURC.

These metrics evaluate whether a model's confidence scores can be used to
selectively abstain on uncertain predictions, reducing risk at the cost of
lower coverage.

References:
    Geifman & El-Yaniv (2017). "Selective Classification for Deep Neural Networks."
    Franc & Prusa (2023). "Optimal Visual Search with Highly Confident Learners."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Data for a risk-coverage plot.

    Sorted by descending confidence.  At coverage c, risk is the mean loss
    over the top-c fraction of samples (ranked by confidence).

    Attributes:
        coverages: 1D array of coverage values (fraction answered), ascending.
        risks: 1D array of risk values (mean loss at that coverage).
    """

    coverages: np.ndarray
    risks: np.ndarray


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_selective_inputs(
    confidences: np.ndarray,
    binary_outcomes_or_losses: np.ndarray,
    *,
    is_binary: bool = True,
) -> None:
    """Validate inputs for selective prediction metrics.

    Args:
        confidences: 1D array of confidences in [0, 1].
        binary_outcomes_or_losses: 1D array of outcomes or losses.
        is_binary: If True, validate outcomes are in {0, 1}.
            If False, validate losses are non-negative.
    """
    if confidences.ndim != 1 or binary_outcomes_or_losses.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays")

    if len(confidences) == 0 or len(binary_outcomes_or_losses) == 0:
        raise ValueError("Inputs must be non-empty")

    if len(confidences) != len(binary_outcomes_or_losses):
        raise ValueError(
            f"Length mismatch: {len(confidences)} confidences "
            f"vs {len(binary_outcomes_or_losses)} outcomes/losses"
        )

    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("Confidence values must be in range [0, 1]")

    if is_binary:
        unique_vals = np.unique(binary_outcomes_or_losses)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"Binary outcomes must contain only 0 and 1, "
                f"got unique values: {unique_vals}"
            )
    else:
        if np.any(binary_outcomes_or_losses < 0.0):
            raise ValueError("Loss values must be non-negative")


# ---------------------------------------------------------------------------
# AUROC / AUPRC
# ---------------------------------------------------------------------------


def auroc_selective(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
) -> float:
    """AUROC for selective prediction: can confidence distinguish correct from incorrect?

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        binary_outcomes: 1D array in {0, 1} (1 = correct).

    Returns:
        AUROC in [0, 1], or NaN if only one class is present.
    """
    _validate_selective_inputs(confidences, binary_outcomes, is_binary=True)

    unique = np.unique(binary_outcomes)
    if len(unique) < 2:
        return float("nan")

    return float(roc_auc_score(binary_outcomes, confidences))


def auprc_selective(
    confidences: np.ndarray,
    binary_outcomes: np.ndarray,
) -> float:
    """AUPRC for selective prediction.

    Important when correct/incorrect classes are imbalanced (which is
    typical — most RAG answers are correct).

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        binary_outcomes: 1D array in {0, 1} (1 = correct).

    Returns:
        AUPRC in [0, 1], or NaN if only one class is present.
    """
    _validate_selective_inputs(confidences, binary_outcomes, is_binary=True)

    unique = np.unique(binary_outcomes)
    if len(unique) < 2:
        # All positive → AUPRC = 1.0; all negative → undefined
        if unique[0] == 1:
            return 1.0
        return float("nan")

    return float(average_precision_score(binary_outcomes, confidences))


# ---------------------------------------------------------------------------
# Risk-coverage curve
# ---------------------------------------------------------------------------


def risk_coverage_curve(
    confidences: np.ndarray,
    losses: np.ndarray,
) -> RiskCoverageCurve:
    """Compute the risk-coverage curve for selective prediction.

    Sorts samples by descending confidence and computes cumulative risk
    (mean loss) at each coverage level.

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        losses: 1D array of per-sample losses (non-negative).

    Returns:
        RiskCoverageCurve with coverages and risks arrays.
    """
    _validate_selective_inputs(confidences, losses, is_binary=False)

    n = len(confidences)

    # Sort by descending confidence (most confident first)
    sorted_indices = np.argsort(-confidences, kind="stable")
    sorted_losses = losses[sorted_indices]

    # Cumulative mean loss at each coverage level
    cum_losses = np.cumsum(sorted_losses)
    counts = np.arange(1, n + 1, dtype=float)
    risks = cum_losses / counts
    coverages = counts / n

    return RiskCoverageCurve(coverages=coverages, risks=risks)


# ---------------------------------------------------------------------------
# AURC / Oracle AURC / E-AURC
# ---------------------------------------------------------------------------


def _compute_aurc_from_curve(rc: RiskCoverageCurve) -> float:
    """Compute area under a risk-coverage curve using trapezoidal rule."""
    return float(np.trapezoid(rc.risks, rc.coverages))


def aurc(
    confidences: np.ndarray,
    losses: np.ndarray,
) -> float:
    """Compute Area Under the Risk-Coverage Curve.

    Lower is better.  Measures how well the model can reduce risk
    by abstaining on low-confidence predictions.

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        losses: 1D array of per-sample losses (non-negative).

    Returns:
        AURC (float, non-negative).
    """
    _validate_selective_inputs(confidences, losses, is_binary=False)
    rc = risk_coverage_curve(confidences, losses)
    return _compute_aurc_from_curve(rc)


def oracle_aurc(losses: np.ndarray) -> float:
    """Compute the oracle (optimal) AURC.

    The oracle sorts by actual loss (ascending), answering correct
    samples first.  This gives the minimum achievable AURC for the
    given set of losses, regardless of confidence quality.

    Args:
        losses: 1D array of per-sample losses (non-negative).

    Returns:
        Oracle AURC (float, non-negative).
    """
    if losses.ndim != 1 or len(losses) == 0:
        raise ValueError("losses must be a non-empty 1-D array")
    if np.any(losses < 0.0):
        raise ValueError("Loss values must be non-negative")

    n = len(losses)
    # Oracle: sort by ascending loss (answer easy ones first)
    sorted_losses = np.sort(losses)
    cum_losses = np.cumsum(sorted_losses)
    counts = np.arange(1, n + 1, dtype=float)
    risks = cum_losses / counts
    coverages = counts / n

    rc = RiskCoverageCurve(coverages=coverages, risks=risks)
    return _compute_aurc_from_curve(rc)


def e_aurc(
    confidences: np.ndarray,
    losses: np.ndarray,
) -> float:
    """Compute Excess AURC: AURC - oracle_AURC.

    Normalizes AURC for task difficulty.  A model with perfect confidence
    ranking achieves E-AURC = 0 regardless of the base error rate.

    Args:
        confidences: 1D array of predicted confidences in [0, 1].
        losses: 1D array of per-sample losses (non-negative).

    Returns:
        E-AURC (float).  Non-negative when confidence is non-adversarial.

    References:
        Geifman & El-Yaniv (2017). "Selective Classification for DNNs."
    """
    _validate_selective_inputs(confidences, losses, is_binary=False)
    return aurc(confidences, losses) - oracle_aurc(losses)
