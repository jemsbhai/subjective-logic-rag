"""TDD tests for LLM-as-judge opinion estimator.

Tests the LLMJudgeEstimator which prompts an LLM to assess document
relevance and support, then maps structured judgments to SL opinions.

Red phase: all tests should FAIL until implementation is written.
"""

from __future__ import annotations

import abc
import json
from dataclasses import fields
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Section 1: Enum tests
# ---------------------------------------------------------------------------


class TestRelevanceLevel:
    """Tests for the RelevanceLevel enum."""

    def test_import(self):
        from xrag.opinion_estimation.llm_judge_estimator import RelevanceLevel

        assert RelevanceLevel is not None

    def test_is_enum(self):
        from xrag.opinion_estimation.llm_judge_estimator import RelevanceLevel

        assert issubclass(RelevanceLevel, Enum)

    def test_members(self):
        from xrag.opinion_estimation.llm_judge_estimator import RelevanceLevel

        names = {m.name for m in RelevanceLevel}
        assert names == {"HIGH", "MEDIUM", "LOW", "NONE"}

    def test_scores(self):
        from xrag.opinion_estimation.llm_judge_estimator import RelevanceLevel

        assert RelevanceLevel.HIGH.score == pytest.approx(1.0)
        assert RelevanceLevel.MEDIUM.score == pytest.approx(0.67, abs=0.01)
        assert RelevanceLevel.LOW.score == pytest.approx(0.33, abs=0.01)
        assert RelevanceLevel.NONE.score == pytest.approx(0.0)

    def test_scores_are_normalized(self):
        """All scores must be in [0, 1]."""
        from xrag.opinion_estimation.llm_judge_estimator import RelevanceLevel

        for member in RelevanceLevel:
            assert 0.0 <= member.score <= 1.0


class TestSupportLevel:
    """Tests for the SupportLevel enum."""

    def test_import(self):
        from xrag.opinion_estimation.llm_judge_estimator import SupportLevel

        assert SupportLevel is not None

    def test_is_enum(self):
        from xrag.opinion_estimation.llm_judge_estimator import SupportLevel

        assert issubclass(SupportLevel, Enum)

    def test_members(self):
        from xrag.opinion_estimation.llm_judge_estimator import SupportLevel

        names = {m.name for m in SupportLevel}
        assert names == {"SUPPORTS", "PARTIALLY_SUPPORTS", "NEUTRAL", "CONTRADICTS"}

    def test_scores(self):
        from xrag.opinion_estimation.llm_judge_estimator import SupportLevel

        assert SupportLevel.SUPPORTS.score == pytest.approx(1.0)
        assert SupportLevel.PARTIALLY_SUPPORTS.score == pytest.approx(0.67, abs=0.01)
        assert SupportLevel.NEUTRAL.score == pytest.approx(0.33, abs=0.01)
        assert SupportLevel.CONTRADICTS.score == pytest.approx(0.0)

    def test_scores_are_normalized(self):
        from xrag.opinion_estimation.llm_judge_estimator import SupportLevel

        for member in SupportLevel:
            assert 0.0 <= member.score <= 1.0


# ---------------------------------------------------------------------------
# Section 2: LLMJudgment dataclass tests
# ---------------------------------------------------------------------------


class TestLLMJudgment:
    """Tests for the LLMJudgment dataclass."""

    def test_import(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgment

        assert LLMJudgment is not None

    def test_construction(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMJudgment,
            RelevanceLevel,
            SupportLevel,
        )

        j = LLMJudgment(
            relevance=RelevanceLevel.HIGH,
            relevance_score=1.0,
            support=SupportLevel.SUPPORTS,
            support_score=1.0,
            raw_response='{"relevance": "high", "support": "supports"}',
        )
        assert j.relevance == RelevanceLevel.HIGH
        assert j.relevance_score == 1.0
        assert j.support == SupportLevel.SUPPORTS
        assert j.support_score == 1.0
        assert isinstance(j.raw_response, str)

    def test_field_names(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgment

        names = {f.name for f in fields(LLMJudgment)}
        assert names == {
            "relevance", "relevance_score",
            "support", "support_score",
            "raw_response",
        }


# ---------------------------------------------------------------------------
# Section 3: LLMBackend abstraction tests
# ---------------------------------------------------------------------------


class TestLLMBackend:
    """Tests for the abstract LLMBackend base class."""

    def test_import(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMBackend

        assert LLMBackend is not None

    def test_is_abstract(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMBackend

        assert abc.ABC in LLMBackend.__mro__

    def test_cannot_instantiate(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMBackend

        with pytest.raises(TypeError):
            LLMBackend()

    def test_has_generate_method(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMBackend

        assert hasattr(LLMBackend, "generate")
        assert getattr(LLMBackend.generate, "__isabstractmethod__", False)


class TestHuggingFaceBackend:
    """Tests for the HuggingFace backend stub."""

    def test_import(self):
        from xrag.opinion_estimation.llm_judge_estimator import HuggingFaceBackend

        assert HuggingFaceBackend is not None

    def test_is_llm_backend(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            HuggingFaceBackend,
            LLMBackend,
        )

        assert issubclass(HuggingFaceBackend, LLMBackend)

    def test_raises_not_implemented(self):
        from xrag.opinion_estimation.llm_judge_estimator import HuggingFaceBackend

        backend = HuggingFaceBackend(model_name="meta-llama/Llama-3.1-8B-Instruct")
        with pytest.raises(NotImplementedError):
            backend.generate("test prompt")


class TestAPIBackend:
    """Tests for the API backend stub."""

    def test_import(self):
        from xrag.opinion_estimation.llm_judge_estimator import APIBackend

        assert APIBackend is not None

    def test_is_llm_backend(self):
        from xrag.opinion_estimation.llm_judge_estimator import APIBackend, LLMBackend

        assert issubclass(APIBackend, LLMBackend)

    def test_raises_not_implemented(self):
        from xrag.opinion_estimation.llm_judge_estimator import APIBackend

        backend = APIBackend(provider="openai", model_name="gpt-4o-mini")
        with pytest.raises(NotImplementedError):
            backend.generate("test prompt")


# ---------------------------------------------------------------------------
# Section 4: LLMJudgeEstimator class basics
# ---------------------------------------------------------------------------


class TestLLMJudgeEstimatorBasics:
    """Tests for LLMJudgeEstimator class structure."""

    def test_import(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        assert LLMJudgeEstimator is not None

    def test_is_base_opinion_estimator(self):
        from xrag.opinion_estimation.base import BaseOpinionEstimator
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        assert issubclass(LLMJudgeEstimator, BaseOpinionEstimator)

    def test_instantiate_with_mock_backend(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        estimator = LLMJudgeEstimator(backend=mock_backend)
        assert estimator is not None

    def test_evidence_weight_default(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        estimator = LLMJudgeEstimator(backend=mock_backend)
        assert estimator.evidence_weight == 10.0

    def test_evidence_weight_custom(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        estimator = LLMJudgeEstimator(backend=mock_backend, evidence_weight=5.0)
        assert estimator.evidence_weight == 5.0

    def test_evidence_weight_must_be_positive(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        with pytest.raises(ValueError):
            LLMJudgeEstimator(backend=mock_backend, evidence_weight=0.0)
        with pytest.raises(ValueError):
            LLMJudgeEstimator(backend=mock_backend, evidence_weight=-1.0)


# ---------------------------------------------------------------------------
# Section 5: Prompt construction tests
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Tests for the prompt template used to query the LLM."""

    def test_build_prompt_returns_string(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        estimator = LLMJudgeEstimator(backend=mock_backend)
        prompt = estimator.build_prompt("What is X?", "X is a thing.")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_query(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        estimator = LLMJudgeEstimator(backend=mock_backend)
        prompt = estimator.build_prompt("What is X?", "X is a thing.")
        assert "What is X?" in prompt

    def test_prompt_contains_document(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        estimator = LLMJudgeEstimator(backend=mock_backend)
        prompt = estimator.build_prompt("What is X?", "X is a thing.")
        assert "X is a thing." in prompt

    def test_prompt_requests_json_output(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        estimator = LLMJudgeEstimator(backend=mock_backend)
        prompt = estimator.build_prompt("Q", "D")
        # Prompt should instruct the LLM to output JSON
        assert "json" in prompt.lower() or "JSON" in prompt


# ---------------------------------------------------------------------------
# Section 6: Response parsing tests
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """Tests for parsing the LLM's structured response into LLMJudgment."""

    def _make_estimator(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        return LLMJudgeEstimator(backend=mock_backend)

    def test_parse_valid_json(self):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMJudgment,
            RelevanceLevel,
            SupportLevel,
        )

        estimator = self._make_estimator()
        response = json.dumps({
            "relevance": "high",
            "support": "supports",
        })
        judgment = estimator.parse_response(response)
        assert isinstance(judgment, LLMJudgment)
        assert judgment.relevance == RelevanceLevel.HIGH
        assert judgment.support == SupportLevel.SUPPORTS

    def test_parse_medium_neutral(self):
        from xrag.opinion_estimation.llm_judge_estimator import RelevanceLevel, SupportLevel

        estimator = self._make_estimator()
        response = json.dumps({"relevance": "medium", "support": "neutral"})
        judgment = estimator.parse_response(response)
        assert judgment.relevance == RelevanceLevel.MEDIUM
        assert judgment.support == SupportLevel.NEUTRAL

    def test_parse_contradicts(self):
        from xrag.opinion_estimation.llm_judge_estimator import SupportLevel

        estimator = self._make_estimator()
        response = json.dumps({"relevance": "high", "support": "contradicts"})
        judgment = estimator.parse_response(response)
        assert judgment.support == SupportLevel.CONTRADICTS

    def test_parse_partially_supports(self):
        from xrag.opinion_estimation.llm_judge_estimator import SupportLevel

        estimator = self._make_estimator()
        response = json.dumps({"relevance": "high", "support": "partially_supports"})
        judgment = estimator.parse_response(response)
        assert judgment.support == SupportLevel.PARTIALLY_SUPPORTS

    def test_relevance_score_matches_enum(self):
        estimator = self._make_estimator()
        response = json.dumps({"relevance": "high", "support": "supports"})
        judgment = estimator.parse_response(response)
        assert judgment.relevance_score == pytest.approx(1.0)

    def test_support_score_matches_enum(self):
        estimator = self._make_estimator()
        response = json.dumps({"relevance": "high", "support": "contradicts"})
        judgment = estimator.parse_response(response)
        assert judgment.support_score == pytest.approx(0.0)

    def test_raw_response_preserved(self):
        estimator = self._make_estimator()
        response = json.dumps({"relevance": "low", "support": "neutral"})
        judgment = estimator.parse_response(response)
        assert judgment.raw_response == response

    def test_parse_with_json_markdown_fences(self):
        """LLMs often wrap JSON in ```json ... ``` blocks."""
        estimator = self._make_estimator()
        response = '```json\n{"relevance": "high", "support": "supports"}\n```'
        judgment = estimator.parse_response(response)
        assert judgment.relevance_score == pytest.approx(1.0)

    def test_parse_case_insensitive(self):
        estimator = self._make_estimator()
        response = json.dumps({"relevance": "HIGH", "support": "Supports"})
        judgment = estimator.parse_response(response)
        assert judgment.relevance_score == pytest.approx(1.0)
        assert judgment.support_score == pytest.approx(1.0)

    def test_parse_invalid_json_returns_fallback(self):
        """Unparseable response should return a vacuous judgment (max uncertainty)."""
        from xrag.opinion_estimation.llm_judge_estimator import RelevanceLevel, SupportLevel

        estimator = self._make_estimator()
        judgment = estimator.parse_response("I cannot parse this gibberish")
        assert judgment.relevance == RelevanceLevel.NONE
        assert judgment.support == SupportLevel.NEUTRAL

    def test_parse_missing_fields_returns_fallback(self):
        from xrag.opinion_estimation.llm_judge_estimator import RelevanceLevel, SupportLevel

        estimator = self._make_estimator()
        judgment = estimator.parse_response(json.dumps({"relevance": "high"}))
        # Missing support → fallback to NEUTRAL
        assert judgment.support == SupportLevel.NEUTRAL


# ---------------------------------------------------------------------------
# Section 7: Opinion mapping tests
# ---------------------------------------------------------------------------


class TestJudgmentToOpinion:
    """Tests for mapping LLMJudgment → SL Opinion."""

    def _make_estimator(self, evidence_weight: float = 10.0):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMBackend,
            LLMJudgeEstimator,
        )

        mock_backend = MagicMock(spec=LLMBackend)
        return LLMJudgeEstimator(backend=mock_backend, evidence_weight=evidence_weight)

    def _make_judgment(self, relevance_score: float, support_score: float):
        from xrag.opinion_estimation.llm_judge_estimator import (
            LLMJudgment,
            RelevanceLevel,
            SupportLevel,
        )

        return LLMJudgment(
            relevance=RelevanceLevel.HIGH,  # enum doesn't matter for mapping
            relevance_score=relevance_score,
            support=SupportLevel.SUPPORTS,
            support_score=support_score,
            raw_response="",
        )

    def test_returns_opinion(self):
        from jsonld_ex.confidence_algebra import Opinion

        estimator = self._make_estimator()
        judgment = self._make_judgment(1.0, 1.0)
        opinion = estimator.judgment_to_opinion(judgment)
        assert isinstance(opinion, Opinion)

    def test_simplex_constraint(self):
        """b + d + u must equal 1.0 for all inputs."""
        estimator = self._make_estimator()
        test_cases = [
            (1.0, 1.0),    # fully relevant, fully supports
            (1.0, 0.0),    # fully relevant, fully contradicts
            (0.0, 0.5),    # not relevant at all
            (0.5, 0.5),    # medium relevant, ambiguous support
            (0.33, 0.67),  # low relevant, partial support
        ]
        for rel, sup in test_cases:
            judgment = self._make_judgment(rel, sup)
            op = estimator.judgment_to_opinion(judgment)
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9, \
                f"Simplex violated for rel={rel}, sup={sup}: b={op.belief}, d={op.disbelief}, u={op.uncertainty}"

    def test_high_relevance_high_support_gives_high_belief(self):
        estimator = self._make_estimator()
        judgment = self._make_judgment(1.0, 1.0)
        op = estimator.judgment_to_opinion(judgment)
        assert op.belief > 0.8
        assert op.disbelief < 0.05

    def test_high_relevance_contradicts_gives_high_disbelief(self):
        estimator = self._make_estimator()
        judgment = self._make_judgment(1.0, 0.0)
        op = estimator.judgment_to_opinion(judgment)
        assert op.disbelief > 0.8
        assert op.belief < 0.05

    def test_no_relevance_gives_high_uncertainty(self):
        estimator = self._make_estimator()
        judgment = self._make_judgment(0.0, 0.5)
        op = estimator.judgment_to_opinion(judgment)
        assert op.uncertainty > 0.9
        assert op.belief < 0.05
        assert op.disbelief < 0.05

    def test_uncertainty_floor(self):
        """Even with perfect relevance + support, uncertainty >= base_u."""
        estimator = self._make_estimator(evidence_weight=10.0)
        judgment = self._make_judgment(1.0, 1.0)
        op = estimator.judgment_to_opinion(judgment)
        base_u = 1.0 / 11.0
        assert op.uncertainty >= base_u - 1e-9

    def test_higher_evidence_weight_lower_uncertainty(self):
        est_low_w = self._make_estimator(evidence_weight=2.0)
        est_high_w = self._make_estimator(evidence_weight=20.0)
        judgment = self._make_judgment(0.8, 0.8)
        op_low = est_low_w.judgment_to_opinion(judgment)
        op_high = est_high_w.judgment_to_opinion(judgment)
        assert op_low.uncertainty > op_high.uncertainty


# ---------------------------------------------------------------------------
# Section 8: End-to-end estimate tests (mocked backend)
# ---------------------------------------------------------------------------


def _mock_backend_response(relevance: str, support: str):
    """Create a mock backend that returns a fixed JSON response."""
    from xrag.opinion_estimation.llm_judge_estimator import LLMBackend

    mock = MagicMock(spec=LLMBackend)
    mock.generate.return_value = json.dumps({
        "relevance": relevance,
        "support": support,
    })
    return mock


class TestEstimateSingle:
    """Tests for LLMJudgeEstimator.estimate() with mocked backend."""

    def test_returns_estimation_result(self):
        from xrag.opinion_estimation.base import EstimationResult
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        result = estimator.estimate("What is X?", "X is a great thing.")
        assert isinstance(result, EstimationResult)

    def test_opinion_is_valid(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        result = estimator.estimate("Q", "D")
        op = result.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_high_support_gives_belief(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        result = estimator.estimate("Q", "D")
        assert result.opinion.belief > 0.8

    def test_contradicts_gives_disbelief(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "contradicts")
        estimator = LLMJudgeEstimator(backend=backend)
        result = estimator.estimate("Q", "D")
        assert result.opinion.disbelief > 0.8

    def test_irrelevant_gives_uncertainty(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("none", "neutral")
        estimator = LLMJudgeEstimator(backend=backend)
        result = estimator.estimate("Q", "D")
        assert result.opinion.uncertainty > 0.9

    def test_metadata_contains_judgment(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        result = estimator.estimate("Q", "D")
        assert result.metadata is not None
        assert "judgment" in result.metadata

    def test_nli_scores_is_none(self):
        """LLM judge does not produce NLI scores."""
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        result = estimator.estimate("Q", "D")
        assert result.nli_scores is None

    def test_backend_called_with_prompt(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        estimator.estimate("What is X?", "X is Y.")
        backend.generate.assert_called_once()
        prompt = backend.generate.call_args[0][0]
        assert "What is X?" in prompt
        assert "X is Y." in prompt


class TestEstimateBatch:
    """Tests for LLMJudgeEstimator.estimate_batch() with mocked backend."""

    def test_returns_correct_count(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        results = estimator.estimate_batch(
            queries=["Q1", "Q2", "Q3"],
            documents=["D1", "D2", "D3"],
        )
        assert len(results) == 3

    def test_mismatched_lengths_raises(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        with pytest.raises(ValueError):
            estimator.estimate_batch(queries=["Q1"], documents=["D1", "D2"])

    def test_empty_batch(self):
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("high", "supports")
        estimator = LLMJudgeEstimator(backend=backend)
        results = estimator.estimate_batch(queries=[], documents=[])
        assert results == []

    def test_all_results_valid(self):
        from xrag.opinion_estimation.base import EstimationResult
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator

        backend = _mock_backend_response("medium", "neutral")
        estimator = LLMJudgeEstimator(backend=backend)
        results = estimator.estimate_batch(
            queries=["Q1", "Q2"],
            documents=["D1", "D2"],
        )
        for r in results:
            assert isinstance(r, EstimationResult)
            op = r.opinion
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
