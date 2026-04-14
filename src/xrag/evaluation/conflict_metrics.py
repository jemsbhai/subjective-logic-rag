"""Conflict detection metrics: set-based P/R/F1 and score-based AUROC.

Evaluates conflict detection against known ground truth from corruption
suites (injected contradictions).

Two evaluation modes:
    1. Set-based: predicted conflict pairs vs true conflict pairs → P/R/F1.
    2. Score-based: continuous conflict scores vs binary labels → AUROC.

Conflict pairs are unordered: (i, j) == (j, i) since conflict is symmetric.
Self-pairs (i, i) are silently discarded.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _normalize_pairs(pairs: set[tuple]) -> set[tuple[int, int]]:
    """Normalize conflict pairs to canonical form (min, max) and discard self-pairs.

    Args:
        pairs: Set of (i, j) tuples.

    Returns:
        Set of (min(i,j), max(i,j)) tuples with self-pairs removed.
    """
    normalized = set()
    for a, b in pairs:
        if a == b:
            continue  # discard self-pairs
        normalized.add((min(a, b), max(a, b)))
    return normalized


def conflict_precision_recall_f1(
    predicted_pairs: set[tuple],
    true_pairs: set[tuple],
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for conflict pair detection.

    Pairs are unordered: (i, j) and (j, i) are treated as the same pair.
    Self-pairs (i, i) are silently discarded.

    Empty-set semantics:
        - Both empty → (1.0, 1.0, 1.0) — vacuous truth, nothing to detect.
        - Predicted empty, true non-empty → (0.0, 0.0, 0.0) — missed everything.
        - Predicted non-empty, true empty → (0.0, 0.0, 0.0) — all false positives.

    Args:
        predicted_pairs: Set of predicted conflict pairs (i, j).
        true_pairs: Set of ground-truth conflict pairs (i, j).

    Returns:
        (precision, recall, f1) — each a float in [0, 1].
    """
    pred_norm = _normalize_pairs(predicted_pairs)
    true_norm = _normalize_pairs(true_pairs)

    # Both empty → vacuous truth
    if len(pred_norm) == 0 and len(true_norm) == 0:
        return (1.0, 1.0, 1.0)

    # One or both non-empty
    tp = len(pred_norm & true_norm)

    precision = tp / len(pred_norm) if len(pred_norm) > 0 else 0.0
    recall = tp / len(true_norm) if len(true_norm) > 0 else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return (float(precision), float(recall), float(f1))


def conflict_detection_auroc(
    conflict_scores: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """AUROC of conflict scores as a discriminator for true conflicts.

    Measures whether higher conflict scores correspond to genuinely
    conflicting document pairs.  Scores are not required to be in [0, 1]
    — only their ranking matters.

    Args:
        conflict_scores: 1D array of pairwise conflict scores.
        true_labels: 1D array of binary labels in {0, 1}
            (1 = true conflict, 0 = consistent).

    Returns:
        AUROC in [0, 1], or NaN if only one class is present.

    Raises:
        ValueError: On mismatched lengths, empty inputs, or non-binary labels.
    """
    if conflict_scores.ndim != 1 or true_labels.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays")

    if len(conflict_scores) == 0 or len(true_labels) == 0:
        raise ValueError("Inputs must be non-empty")

    if len(conflict_scores) != len(true_labels):
        raise ValueError(
            f"Length mismatch: {len(conflict_scores)} scores "
            f"vs {len(true_labels)} labels"
        )

    unique_vals = np.unique(true_labels)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"Binary labels must contain only 0 and 1, "
            f"got unique values: {unique_vals}"
        )

    if len(unique_vals) < 2:
        return float("nan")

    return float(roc_auc_score(true_labels, conflict_scores))
