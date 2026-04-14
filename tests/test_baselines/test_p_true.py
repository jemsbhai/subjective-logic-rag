"""Tests for baselines/p_true.py.

P(True) baseline (Kadavath et al., 2022 — "Language Models (Mostly) Know
What They Know").  Cost: 1 extra LLM call per example.

Algorithm:
    1. Format self-evaluation prompt: "Q: ... A: ... Is this correct? (True/False)"
    2. Get LLM logits for "True" vs "False" tokens
    3. P(True) = softmax(logit_True, logit_False)[True]

Core logic tested with mocks — no real LLM needed.
"""

import math

import numpy as np
import pytest

from xrag.baselines.base import UQScore, UQScorer
from xrag.baselines.p_true import (
    format_p_true_prompt,
    p_true_from_logits,
    PTrueScorer,
)


# =============================================================================
# format_p_true_prompt
# =============================================================================


class TestFormatPTruePrompt:
    """Prompt formatting for self-evaluation."""

    def test_basic_format(self):
        prompt = format_p_true_prompt(
            query="What is the capital of France?",
            answer="Paris",
        )
        assert "What is the capital of France?" in prompt
        assert "Paris" in prompt
        assert "True" in prompt or "true" in prompt
        assert "False" in prompt or "false" in prompt

    def test_with_passages(self):
        prompt = format_p_true_prompt(
            query="Who wrote Hamlet?",
            answer="Shakespeare",
            passages=["William Shakespeare wrote Hamlet in 1600."],
        )
        assert "Shakespeare wrote Hamlet" in prompt
        assert "Who wrote Hamlet?" in prompt

    def test_without_passages(self):
        prompt = format_p_true_prompt(
            query="test?",
            answer="answer",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_returns_string(self):
        result = format_p_true_prompt(query="q", answer="a")
        assert isinstance(result, str)

    def test_custom_template(self):
        template = "Q: {query}\nA: {answer}\nCorrect? "
        prompt = format_p_true_prompt(
            query="test", answer="ans", template=template
        )
        assert prompt == "Q: test\nA: ans\nCorrect? "


# =============================================================================
# p_true_from_logits
# =============================================================================


class TestPTrueFromLogits:
    """Extract P(True) from logits over True/False tokens."""

    def test_equal_logits(self):
        """Equal logits → P(True) = 0.5."""
        result = p_true_from_logits(logit_true=0.0, logit_false=0.0)
        assert result == pytest.approx(0.5)

    def test_high_true_logit(self):
        """logit_true >> logit_false → P(True) ≈ 1.0."""
        result = p_true_from_logits(logit_true=10.0, logit_false=0.0)
        assert result > 0.99

    def test_high_false_logit(self):
        """logit_false >> logit_true → P(True) ≈ 0.0."""
        result = p_true_from_logits(logit_true=0.0, logit_false=10.0)
        assert result < 0.01

    def test_hand_computed(self):
        """
        logit_true=2.0, logit_false=1.0
        softmax: exp(2)/(exp(2)+exp(1)) = 7.389/(7.389+2.718) = 7.389/10.107 ≈ 0.7311
        """
        result = p_true_from_logits(logit_true=2.0, logit_false=1.0)
        expected = math.exp(2.0) / (math.exp(2.0) + math.exp(1.0))
        assert result == pytest.approx(expected)

    def test_bounded_0_1(self):
        for lt, lf in [(5.0, -5.0), (-5.0, 5.0), (0.0, 0.0), (100.0, 100.0)]:
            result = p_true_from_logits(logit_true=lt, logit_false=lf)
            assert 0.0 <= result <= 1.0

    def test_symmetric(self):
        """P(True | lt, lf) = 1 - P(True | lf, lt)."""
        p1 = p_true_from_logits(logit_true=3.0, logit_false=1.0)
        p2 = p_true_from_logits(logit_true=1.0, logit_false=3.0)
        assert p1 + p2 == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(p_true_from_logits(0.0, 0.0), float)

    def test_numerical_stability_large_logits(self):
        """Should not overflow with very large logits."""
        result = p_true_from_logits(logit_true=1000.0, logit_false=0.0)
        assert result == pytest.approx(1.0)
        result2 = p_true_from_logits(logit_true=0.0, logit_false=1000.0)
        assert result2 == pytest.approx(0.0)


# =============================================================================
# PTrueScorer — integration with mocked generator
# =============================================================================


class TestPTrueScorer:
    """Full scorer with mocked LLM for self-evaluation."""

    def _make_mock_generator(self, logit_true: float, logit_false: float):
        """Mock generator that returns fixed logits for True/False."""

        class MockGenerator:
            def get_token_logits(self, prompt, target_tokens):
                return {t: (logit_true if t.lower().strip() == "true"
                           else logit_false)
                        for t in target_tokens}

        return MockGenerator()

    def test_is_uq_scorer(self):
        gen = self._make_mock_generator(1.0, 0.0)
        scorer = PTrueScorer(generator=gen)
        assert isinstance(scorer, UQScorer)

    def test_cost_category(self):
        gen = self._make_mock_generator(1.0, 0.0)
        scorer = PTrueScorer(generator=gen)
        assert scorer.cost_category == "extra_call"

    def test_name_contains_p_true(self):
        gen = self._make_mock_generator(1.0, 0.0)
        scorer = PTrueScorer(generator=gen)
        assert "true" in scorer.name.lower()

    def test_high_true_logit_high_confidence(self):
        gen = self._make_mock_generator(logit_true=5.0, logit_false=0.0)
        scorer = PTrueScorer(generator=gen)
        result = scorer.score(
            query="What is 2+2?", answer="4", passages=[]
        )
        assert isinstance(result, UQScore)
        assert result.confidence > 0.9

    def test_high_false_logit_low_confidence(self):
        gen = self._make_mock_generator(logit_true=0.0, logit_false=5.0)
        scorer = PTrueScorer(generator=gen)
        result = scorer.score(
            query="What is 2+2?", answer="5", passages=[]
        )
        assert result.confidence < 0.1

    def test_equal_logits_half_confidence(self):
        gen = self._make_mock_generator(logit_true=0.0, logit_false=0.0)
        scorer = PTrueScorer(generator=gen)
        result = scorer.score(query="q", answer="a", passages=[])
        assert result.confidence == pytest.approx(0.5)

    def test_confidence_bounded_0_1(self):
        gen = self._make_mock_generator(logit_true=3.0, logit_false=1.0)
        scorer = PTrueScorer(generator=gen)
        result = scorer.score(query="q", answer="a", passages=[])
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata_contains_logits(self):
        gen = self._make_mock_generator(logit_true=2.0, logit_false=1.0)
        scorer = PTrueScorer(generator=gen)
        result = scorer.score(query="q", answer="a", passages=[])
        assert "logit_true" in result.metadata
        assert "logit_false" in result.metadata
        assert "p_true" in result.metadata

    def test_metadata_contains_prompt(self):
        gen = self._make_mock_generator(logit_true=1.0, logit_false=1.0)
        scorer = PTrueScorer(generator=gen)
        result = scorer.score(query="q", answer="a", passages=[])
        assert "prompt" in result.metadata

    def test_returns_uq_score(self):
        gen = self._make_mock_generator(1.0, 0.0)
        scorer = PTrueScorer(generator=gen)
        result = scorer.score(query="q", answer="a", passages=[])
        assert isinstance(result, UQScore)
