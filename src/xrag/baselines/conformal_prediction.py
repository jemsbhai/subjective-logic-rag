"""Split conformal prediction baseline for selective RAG.

Cherian, Gibbs, Candès (NeurIPS 2024); Quach et al. (ICLR 2024).
Cost: FREE at inference (calibration is a one-time offline step).

Provides distribution-free coverage guarantee:
    P(Y_correct | confidence ≥ τ) ≥ 1 - α

The conformal scorer wraps ANY base confidence signal (softmax, retrieval,
combined, or even SL-RAG's projected probability) and adds a calibrated
threshold with a finite-sample coverage guarantee.

Algorithm:
    1. On calibration set: collect confidences of CORRECT examples.
    2. Compute threshold τ = quantile of correct confidences at level α,
       with finite-sample correction: level = ⌈α(n+1)⌉ / n.
    3. At inference: if confidence ≥ τ → answer; else → abstain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xrag.baselines.base import UQScore, UQScorer


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def compute_conformal_threshold(
    cal_confidences: np.ndarray,
    cal_outcomes: np.ndarray,
    alpha: float,
) -> float:
    """Compute conformal threshold from calibration data.

    Finds the lowest threshold τ such that accuracy among calibration
    examples with confidence ≥ τ is at least 1-α.  This directly
    provides an accuracy guarantee on the answered subset.

    Semantics:
        - Lower α (stricter) → higher τ → fewer answers, higher accuracy.
        - Higher α (more tolerant) → lower τ → more answers, lower accuracy.

    Args:
        cal_confidences: 1D array of calibration confidences in [0, 1].
        cal_outcomes: 1D array of binary outcomes in {0, 1}.
        alpha: Desired maximum error rate among answered examples.
            Must be in (0, 1).

    Returns:
        Threshold τ (float).

    Raises:
        ValueError: On invalid inputs.
    """
    if cal_confidences.ndim != 1 or cal_outcomes.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays")

    if len(cal_confidences) == 0:
        raise ValueError("Calibration data must be non-empty")

    if len(cal_confidences) != len(cal_outcomes):
        raise ValueError(
            f"Length mismatch: {len(cal_confidences)} confidences "
            f"vs {len(cal_outcomes)} outcomes"
        )

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

    unique_vals = np.unique(cal_outcomes)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"Binary outcomes must contain only 0 and 1, "
            f"got unique values: {unique_vals}"
        )

    if not np.any(cal_outcomes == 1):
        raise ValueError(
            "No correct examples in calibration set — cannot calibrate"
        )

    # Find the lowest threshold τ such that accuracy among
    # examples with confidence ≥ τ is ≥ 1-α.
    # Iterate over unique confidence values in ascending order.
    sorted_unique = np.sort(np.unique(cal_confidences))
    target_accuracy = 1.0 - alpha

    for tau in sorted_unique:
        answered_mask = cal_confidences >= tau
        n_answered = int(np.sum(answered_mask))
        if n_answered == 0:
            continue
        accuracy = float(np.mean(cal_outcomes[answered_mask]))
        if accuracy >= target_accuracy:
            return float(tau)

    # No threshold achieves target accuracy → return above max confidence
    # (i.e., abstain on everything)
    return float(np.max(cal_confidences) + 1e-9)


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class SplitConformalScorer(UQScorer):
    """Split conformal prediction scorer for selective RAG.

    Wraps any base confidence signal and adds a calibrated threshold
    with a distribution-free coverage guarantee.

    Usage:
        scorer = SplitConformalScorer(alpha=0.1)
        scorer.calibrate(cal_confidences, cal_outcomes)
        result = scorer.score(confidence=0.85)
        if result.metadata["above_threshold"]:
            # answer
        else:
            # abstain

    Args:
        alpha: Desired miscoverage rate (default 0.1 for 90% coverage).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._threshold: float | None = None
        self._calibrated = False

    @property
    def name(self) -> str:
        return f"conformal_alpha{self._alpha:.2f}"

    @property
    def cost_category(self) -> str:
        return "free"

    @property
    def threshold(self) -> float | None:
        """Calibrated threshold τ, or None if not yet calibrated."""
        return self._threshold

    def calibrate(
        self,
        cal_confidences: np.ndarray,
        cal_outcomes: np.ndarray,
    ) -> None:
        """Calibrate the threshold from held-out calibration data.

        Args:
            cal_confidences: 1D array of calibration confidences.
            cal_outcomes: 1D array of binary outcomes.
        """
        self._threshold = compute_conformal_threshold(
            cal_confidences, cal_outcomes, self._alpha,
        )
        self._calibrated = True

    def score(
        self,
        *,
        confidence: float | None = None,
        **kwargs: Any,
    ) -> UQScore:
        """Score an example using the calibrated conformal threshold.

        The output confidence is the original confidence value, but
        metadata indicates whether it exceeds the threshold (i.e.,
        whether the model should answer or abstain).

        Args:
            confidence: Base confidence score from any scorer.

        Returns:
            UQScore with original confidence and threshold metadata.

        Raises:
            RuntimeError: If not yet calibrated.
            ValueError: If confidence is None.
        """
        if not self._calibrated:
            raise RuntimeError(
                "Scorer must be calibrated before scoring. "
                "Call calibrate(cal_confidences, cal_outcomes) first."
            )

        if confidence is None:
            raise ValueError("confidence is required")

        above = bool(confidence >= self._threshold)

        return UQScore(
            confidence=float(confidence),
            method=self.name,
            metadata={
                "threshold": self._threshold,
                "above_threshold": above,
                "alpha": self._alpha,
                "decision": "answer" if above else "abstain",
            },
        )

    def score_batch(self, inputs_list: list[dict[str, float]]) -> np.ndarray:
        """Score a batch of confidence values.

        Args:
            inputs_list: List of dicts with "confidence" key.

        Returns:
            1D numpy array of confidence values.
        """
        return np.array([
            self.score(**inp).confidence for inp in inputs_list
        ])
