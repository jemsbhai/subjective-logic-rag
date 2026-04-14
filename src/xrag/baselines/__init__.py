"""UQ baseline methods for comparison with SL-RAG."""

from xrag.baselines.base import UQScore, UQScorer
from xrag.baselines.softmax_confidence import (
    mean_logprob,
    normalized_seq_prob,
    perplexity_to_conf,
    min_token_prob,
    SoftmaxConfidenceScorer,
)
from xrag.baselines.retrieval_confidence import (
    max_retrieval_score,
    mean_retrieval_score,
    score_gap,
    score_entropy,
    RetrievalConfidenceScorer,
)
from xrag.baselines.semantic_entropy import (
    cluster_by_equivalence,
    semantic_entropy_from_clusters,
    SemanticEntropyScorer,
)
from xrag.baselines.p_true import (
    format_p_true_prompt,
    p_true_from_logits,
    PTrueScorer,
)
from xrag.baselines.combined_heuristic import (
    CombinedHeuristicScorer,
)

__all__ = [
    "UQScore",
    "UQScorer",
    # Softmax confidence (free)
    "mean_logprob",
    "normalized_seq_prob",
    "perplexity_to_conf",
    "min_token_prob",
    "SoftmaxConfidenceScorer",
    # Retrieval confidence (free)
    "max_retrieval_score",
    "mean_retrieval_score",
    "score_gap",
    "score_entropy",
    "RetrievalConfidenceScorer",
    # Semantic entropy (n_samples)
    "cluster_by_equivalence",
    "semantic_entropy_from_clusters",
    "SemanticEntropyScorer",
    # P(True) (extra_call)
    "format_p_true_prompt",
    "p_true_from_logits",
    "PTrueScorer",
    # Combined heuristic (free)
    "CombinedHeuristicScorer",
]
