"""Faithfulness metrics: claim decomposition and entailment-based scoring.

Evaluates whether generated answers are faithful to the retrieved evidence
by decomposing answers into claims and checking each claim against evidence.

Two decomposition methods:
    1. "sentence" — nltk sentence tokenizer (no model needed).
    2. "llm" — LLM-based atomic claim decomposition (callable interface).

The entailment function is injected as a callable, keeping this module
decoupled from any specific NLI model.

References:
    Min et al. (2023). "FActScore: Fine-grained Atomic Evaluation of Factual Precision."
    Es et al. (2024). "RAGAs: Automated Evaluation of Retrieval Augmented Generation."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import nltk


# Ensure punkt_tab tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


@dataclass(frozen=True)
class FaithfulnessResult:
    """Result of faithfulness evaluation.

    Attributes:
        claims: List of decomposed claim strings.
        per_claim_supported: List of booleans — True if claim is entailed
            by at least one evidence passage.
        precision: Fraction of claims that are supported (faithfulness precision).
        method: Decomposition method used ("sentence" or "llm").
    """

    claims: list[str]
    per_claim_supported: list[bool]
    precision: float
    method: str


def decompose_claims(
    answer: str,
    method: str = "sentence",
    *,
    decompose_fn: Callable[[str], list[str]] | None = None,
) -> list[str]:
    """Decompose an answer string into individual claims.

    Args:
        answer: The generated answer text.
        method: Decomposition method — "sentence" or "llm".
        decompose_fn: Required when method="llm". A callable that takes
            a string and returns a list of atomic claim strings.

    Returns:
        List of claim strings.

    Raises:
        ValueError: On invalid method or missing decompose_fn for "llm".
    """
    if method == "sentence":
        return _decompose_sentence(answer)
    elif method == "llm":
        if decompose_fn is None:
            raise ValueError("decompose_fn is required when method='llm'")
        return decompose_fn(answer)
    else:
        raise ValueError(f"Method must be 'sentence' or 'llm', got '{method}'")


def _decompose_sentence(answer: str) -> list[str]:
    """Split answer into sentences using nltk sentence tokenizer."""
    stripped = answer.strip()
    if not stripped:
        return []

    sentences = nltk.sent_tokenize(stripped)
    # Strip whitespace and filter empty
    return [s.strip() for s in sentences if s.strip()]


def faithfulness_precision(
    claims: list[str],
    evidence: list[str],
    entailment_fn: Callable[[str, str], bool],
) -> float:
    """Compute faithfulness precision: fraction of claims supported by evidence.

    A claim is "supported" if at least one evidence passage entails it.
    Short-circuits on the first entailing passage for each claim.

    Args:
        claims: List of claim strings (from decompose_claims).
        evidence: List of evidence passage strings.
        entailment_fn: Callable(premise, hypothesis) -> bool.
            Returns True if premise entails hypothesis.

    Returns:
        Precision in [0, 1]. Returns 1.0 if claims is empty (vacuous truth).
    """
    if len(claims) == 0:
        return 1.0

    if len(evidence) == 0:
        return 0.0

    supported_count = 0
    for claim in claims:
        for passage in evidence:
            if entailment_fn(passage, claim):
                supported_count += 1
                break  # short-circuit: claim is supported

    return float(supported_count / len(claims))


def faithfulness_result(
    answer: str,
    evidence: list[str],
    entailment_fn: Callable[[str, str], bool],
    method: str = "sentence",
    *,
    decompose_fn: Callable[[str], list[str]] | None = None,
) -> FaithfulnessResult:
    """Full faithfulness evaluation with per-claim verdicts.

    Decomposes the answer into claims, checks each against evidence,
    and returns detailed results.

    Args:
        answer: The generated answer text.
        evidence: List of evidence passage strings.
        entailment_fn: Callable(premise, hypothesis) -> bool.
        method: Decomposition method — "sentence" or "llm".
        decompose_fn: Required when method="llm".

    Returns:
        FaithfulnessResult with claims, per-claim verdicts, and precision.
    """
    claims = decompose_claims(answer, method=method, decompose_fn=decompose_fn)

    if len(claims) == 0:
        return FaithfulnessResult(
            claims=[],
            per_claim_supported=[],
            precision=1.0,
            method=method,
        )

    per_claim_supported: list[bool] = []
    for claim in claims:
        supported = False
        if len(evidence) > 0:
            for passage in evidence:
                if entailment_fn(passage, claim):
                    supported = True
                    break
        per_claim_supported.append(supported)

    precision = sum(per_claim_supported) / len(per_claim_supported)

    return FaithfulnessResult(
        claims=claims,
        per_claim_supported=per_claim_supported,
        precision=float(precision),
        method=method,
    )
