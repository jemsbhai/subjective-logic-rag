"""Opinion estimation: NLI and LLM-as-judge → SL opinions."""

from xrag.opinion_estimation.base import BaseOpinionEstimator, EstimationResult
from xrag.opinion_estimation.llm_judge_estimator import (
    APIBackend,
    HuggingFaceBackend,
    LLMBackend,
    LLMJudgeEstimator,
    LLMJudgment,
    RelevanceLevel,
    SupportLevel,
)
from xrag.opinion_estimation.nli_estimator import (
    NLIEstimator,
    NLIFaithfulnessEstimator,
    NLIRelevanceEstimator,
)

__all__ = [
    "APIBackend",
    "BaseOpinionEstimator",
    "EstimationResult",
    "HuggingFaceBackend",
    "LLMBackend",
    "LLMJudgeEstimator",
    "LLMJudgment",
    "NLIEstimator",
    "NLIFaithfulnessEstimator",
    "NLIRelevanceEstimator",
    "RelevanceLevel",
    "SupportLevel",
]
