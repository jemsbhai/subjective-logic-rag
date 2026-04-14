"""SQuAD-canonical answer correctness metrics: EM and token F1.

Normalization follows the official SQuAD v1.1/v2.0 evaluation script exactly:
    lower → remove_punc → remove_articles → white_space_fix

References:
    - https://rajpurkar.github.io/SQuAD-explorer/
    - allenai/bi-att-flow/squad/evaluate-v1.1.py
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Matches the official SQuAD v1.1 evaluation script exactly.
    Order: lower → remove_punc → remove_articles → white_space_fix.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _get_tokens(s: str) -> list[str]:
    """Tokenize a normalized string by whitespace."""
    if not s:
        return []
    return normalize_answer(s).split()


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between a single prediction and ground truth.

    Follows the official SQuAD v1.1 evaluation script.
    """
    prediction_tokens = _get_tokens(prediction)
    ground_truth_tokens = _get_tokens(ground_truth)

    # SQuAD v2 convention: if either is empty, F1 is 1 if they agree, 0 otherwise
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    """Check if prediction exactly matches any gold answer after normalization.

    Args:
        prediction: Model-generated answer string.
        gold_answers: List of acceptable gold answer strings. Must be non-empty.

    Returns:
        True if normalized prediction equals any normalized gold answer.

    Raises:
        ValueError: If gold_answers is empty.
    """
    if not gold_answers:
        raise ValueError("gold_answers must be a non-empty list")

    normalized_pred = normalize_answer(prediction)
    return any(normalized_pred == normalize_answer(gold) for gold in gold_answers)


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    """Compute token-level F1, taking max over all gold answers.

    Uses bag-of-words precision and recall over normalized tokens,
    following the official SQuAD evaluation script.

    Args:
        prediction: Model-generated answer string.
        gold_answers: List of acceptable gold answer strings. Must be non-empty.

    Returns:
        Maximum F1 score across all gold answers, in [0.0, 1.0].

    Raises:
        ValueError: If gold_answers is empty.
    """
    if not gold_answers:
        raise ValueError("gold_answers must be a non-empty list")

    return float(max(_compute_f1(prediction, gold) for gold in gold_answers))


def batch_em(
    predictions: list[str],
    gold_answers_list: list[list[str]],
) -> float:
    """Compute mean Exact Match over a dataset.

    Args:
        predictions: List of predicted answer strings.
        gold_answers_list: List of gold answer lists (one per example).

    Returns:
        Mean EM score in [0.0, 1.0].

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if not predictions or not gold_answers_list:
        raise ValueError("predictions and gold_answers_list must be non-empty")
    if len(predictions) != len(gold_answers_list):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(gold_answers_list)} gold answer lists"
        )

    scores = [
        float(exact_match(pred, golds))
        for pred, golds in zip(predictions, gold_answers_list)
    ]
    return sum(scores) / len(scores)


def batch_f1(
    predictions: list[str],
    gold_answers_list: list[list[str]],
) -> float:
    """Compute mean token F1 over a dataset.

    Args:
        predictions: List of predicted answer strings.
        gold_answers_list: List of gold answer lists (one per example).

    Returns:
        Mean F1 score in [0.0, 1.0].

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if not predictions or not gold_answers_list:
        raise ValueError("predictions and gold_answers_list must be non-empty")
    if len(predictions) != len(gold_answers_list):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(gold_answers_list)} gold answer lists"
        )

    scores = [
        token_f1(pred, golds)
        for pred, golds in zip(predictions, gold_answers_list)
    ]
    return sum(scores) / len(scores)
