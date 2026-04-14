"""Evaluation metrics for SL-RAG experiments."""

from xrag.evaluation.answer_metrics import (
    normalize_answer,
    exact_match,
    token_f1,
    batch_em,
    batch_f1,
)
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
from xrag.evaluation.selective_metrics import (
    auroc_selective,
    auprc_selective,
    risk_coverage_curve,
    aurc,
    oracle_aurc,
    e_aurc,
    RiskCoverageCurve,
)
from xrag.evaluation.conflict_metrics import (
    conflict_precision_recall_f1,
    conflict_detection_auroc,
)
from xrag.evaluation.faithfulness_metrics import (
    decompose_claims,
    faithfulness_precision,
    faithfulness_result,
    FaithfulnessResult,
)

__all__ = [
    # Answer correctness
    "normalize_answer",
    "exact_match",
    "token_f1",
    "batch_em",
    "batch_f1",
    # Calibration
    "expected_calibration_error",
    "brier_score",
    "reliability_diagram_data",
    "ReliabilityDiagram",
    "maximum_calibration_error",
    "debiased_ece_l2",
    "bootstrap_calibration_ci",
    "BootstrapCI",
    # Selective prediction
    "auroc_selective",
    "auprc_selective",
    "risk_coverage_curve",
    "aurc",
    "oracle_aurc",
    "e_aurc",
    "RiskCoverageCurve",
    # Conflict detection
    "conflict_precision_recall_f1",
    "conflict_detection_auroc",
    # Faithfulness
    "decompose_claims",
    "faithfulness_precision",
    "faithfulness_result",
    "FaithfulnessResult",
]
