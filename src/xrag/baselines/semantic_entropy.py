"""Semantic Entropy baseline for UQ in RAG.

Kuhn, Gal, Farquhar — "Semantic Uncertainty" (ICLR 2023, Nature 2024).
Cost: N × generation + O(N²) equivalence checks.

Algorithm:
    1. Generate N answers with temperature > 0 (sampling).
    2. Cluster answers by semantic equivalence (bidirectional NLI).
    3. Compute entropy over cluster distribution: SE = -Σ p_c · log p_c.
    4. Map to confidence: exp(-SE) ∈ (0, 1].

The generator and equivalence function are injected, keeping this
module decoupled from specific models.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from xrag.baselines.base import UQScore, UQScorer


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def cluster_by_equivalence(
    answers: list[str],
    equivalence_fn: Callable[[str, str], bool],
) -> list[list[str]]:
    """Cluster answers by pairwise semantic equivalence.

    Uses a greedy single-linkage approach: each answer is assigned to
    the first existing cluster whose representative is equivalent.

    Args:
        answers: List of generated answer strings.
        equivalence_fn: Callable(a, b) -> bool. True if semantically equivalent.

    Returns:
        List of clusters, each a list of answer strings.

    Raises:
        ValueError: If answers is empty.
    """
    if not answers:
        raise ValueError("answers must be non-empty")

    clusters: list[list[str]] = []
    representatives: list[str] = []

    for answer in answers:
        assigned = False
        for i, rep in enumerate(representatives):
            if equivalence_fn(answer, rep):
                clusters[i].append(answer)
                assigned = True
                break
        if not assigned:
            clusters.append([answer])
            representatives.append(answer)

    return clusters


def semantic_entropy_from_clusters(
    clusters: list[list[str]],
    n_total: int,
) -> float:
    """Compute semantic entropy from cluster sizes.

    SE = -Σ (n_c / N) · log(n_c / N)

    Args:
        clusters: List of clusters (each a list of answers).
        n_total: Total number of answers (must equal sum of cluster sizes).

    Returns:
        Semantic entropy (float, non-negative). 0 = perfect agreement.

    Raises:
        ValueError: If n_total doesn't match sum of cluster sizes.
    """
    actual_total = sum(len(c) for c in clusters)
    if actual_total != n_total:
        raise ValueError(
            f"Total mismatch: sum of cluster sizes is {actual_total}, "
            f"but n_total is {n_total}"
        )

    if n_total == 0:
        return 0.0

    probs = np.array([len(c) / n_total for c in clusters])
    # Filter zeros (shouldn't happen with valid clusters, but be safe)
    nonzero = probs[probs > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))

    return entropy


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class SemanticEntropyScorer(UQScorer):
    """Semantic entropy scorer.

    Generates N answers via sampling, clusters by semantic equivalence,
    and returns confidence = exp(-SE).

    Args:
        generator: Any object with a .generate(query, passages, ...) method
            returning a GenerationResult.
        equivalence_fn: Callable(a, b) -> bool for semantic equivalence.
        n_samples: Number of sampled generations per query (default 5).
        temperature: Sampling temperature (default 0.7).
    """

    def __init__(
        self,
        generator: Any,
        equivalence_fn: Callable[[str, str], bool],
        n_samples: int = 5,
        temperature: float = 0.7,
    ) -> None:
        self._generator = generator
        self._equivalence_fn = equivalence_fn
        self._n_samples = n_samples
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"semantic_entropy_n{self._n_samples}"

    @property
    def cost_category(self) -> str:
        return "n_samples"

    def score(self, *, query: str, passages: list | None = None, **kwargs: Any) -> UQScore:
        """Compute semantic entropy confidence for a query.

        Args:
            query: The question string.
            passages: Retrieved passages (passed to generator).

        Returns:
            UQScore with confidence = exp(-SE) ∈ (0, 1].
        """
        if passages is None:
            passages = []

        # Step 1: Generate N answers via sampling
        answers: list[str] = []
        for _ in range(self._n_samples):
            result = self._generator.generate(
                query=query,
                passages=passages,
                temperature=self._temperature,
            )
            answers.append(result.answer)

        # Step 2: Cluster by equivalence
        clusters = cluster_by_equivalence(answers, self._equivalence_fn)

        # Step 3: Compute semantic entropy
        se = semantic_entropy_from_clusters(clusters, n_total=len(answers))

        # Step 4: Map to confidence
        confidence = float(math.exp(-se))

        return UQScore(
            confidence=confidence,
            method=self.name,
            metadata={
                "semantic_entropy": se,
                "n_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
                "answers": answers,
                "n_samples": self._n_samples,
                "temperature": self._temperature,
            },
        )
