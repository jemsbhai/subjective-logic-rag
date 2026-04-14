"""P(True) baseline for UQ in RAG.

Kadavath et al. (2022) — "Language Models (Mostly) Know What They Know."
Cost: 1 extra LLM call per example.

Algorithm:
    1. Format self-evaluation prompt with query, answer, and optional context.
    2. Get LLM logits for "True" vs "False" tokens.
    3. P(True) = softmax(logit_True, logit_False)[True].
"""

from __future__ import annotations

import math
from typing import Any

from xrag.baselines.base import UQScore, UQScorer


# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

_DEFAULT_P_TRUE_TEMPLATE = (
    "Given the following context and question-answer pair, "
    "is the answer correct?\n\n"
    "{context}"
    "Question: {query}\n"
    "Proposed Answer: {answer}\n\n"
    "Is the proposed answer correct? Respond with True or False.\n"
    "Answer:"
)

_DEFAULT_P_TRUE_TEMPLATE_NO_CONTEXT = (
    "Given the following question-answer pair, "
    "is the answer correct?\n\n"
    "Question: {query}\n"
    "Proposed Answer: {answer}\n\n"
    "Is the proposed answer correct? Respond with True or False.\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def format_p_true_prompt(
    query: str,
    answer: str,
    passages: list[str] | None = None,
    template: str | None = None,
) -> str:
    """Format the self-evaluation prompt for P(True).

    Args:
        query: The question string.
        answer: The proposed answer.
        passages: Optional context passages.
        template: Custom template with {query}, {answer}, and optionally
            {context} placeholders.

    Returns:
        Formatted prompt string.
    """
    if template is not None:
        return template.format(query=query, answer=answer)

    if passages:
        context_str = "Context:\n"
        for i, p in enumerate(passages):
            context_str += f"[{i + 1}] {p}\n"
        context_str += "\n"
        return _DEFAULT_P_TRUE_TEMPLATE.format(
            query=query, answer=answer, context=context_str,
        )
    else:
        return _DEFAULT_P_TRUE_TEMPLATE_NO_CONTEXT.format(
            query=query, answer=answer,
        )


def p_true_from_logits(logit_true: float, logit_false: float) -> float:
    """Compute P(True) from logits via numerically stable softmax.

    P(True) = exp(logit_true) / (exp(logit_true) + exp(logit_false))

    Args:
        logit_true: Raw logit for the "True" token.
        logit_false: Raw logit for the "False" token.

    Returns:
        P(True) ∈ [0, 1].
    """
    # Numerically stable softmax: subtract max
    max_logit = max(logit_true, logit_false)
    exp_true = math.exp(logit_true - max_logit)
    exp_false = math.exp(logit_false - max_logit)
    return float(exp_true / (exp_true + exp_false))


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class PTrueScorer(UQScorer):
    """P(True) scorer using LLM self-evaluation.

    Requires a generator with a .get_token_logits(prompt, target_tokens)
    method that returns a dict mapping token strings to logit values.

    Args:
        generator: Object with get_token_logits(prompt, target_tokens) method.
        true_token: Token string for "True" (default "True").
        false_token: Token string for "False" (default "False").
        prompt_template: Optional custom prompt template.
    """

    def __init__(
        self,
        generator: Any,
        true_token: str = "True",
        false_token: str = "False",
        prompt_template: str | None = None,
    ) -> None:
        self._generator = generator
        self._true_token = true_token
        self._false_token = false_token
        self._prompt_template = prompt_template

    @property
    def name(self) -> str:
        return "p_true"

    @property
    def cost_category(self) -> str:
        return "extra_call"

    def score(
        self,
        *,
        query: str,
        answer: str,
        passages: list[str] | None = None,
        **kwargs: Any,
    ) -> UQScore:
        """Compute P(True) confidence for a query-answer pair.

        Args:
            query: The question string.
            answer: The proposed answer.
            passages: Optional context passages.

        Returns:
            UQScore with confidence = P(True).
        """
        prompt = format_p_true_prompt(
            query=query,
            answer=answer,
            passages=passages,
            template=self._prompt_template,
        )

        # Get logits for True/False tokens
        target_tokens = [self._true_token, self._false_token]
        logits = self._generator.get_token_logits(prompt, target_tokens)

        logit_true = logits[self._true_token]
        logit_false = logits[self._false_token]

        confidence = p_true_from_logits(logit_true, logit_false)

        return UQScore(
            confidence=confidence,
            method=self.name,
            metadata={
                "logit_true": logit_true,
                "logit_false": logit_false,
                "p_true": confidence,
                "prompt": prompt,
            },
        )
