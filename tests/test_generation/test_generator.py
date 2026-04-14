"""Tests for the generation module.

Unit tests use a mock model. Integration tests use a real local LLM
and are marked with @pytest.mark.gpu and @pytest.mark.slow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from xrag.benchmarks.retriever import RetrievedPassage
from xrag.generation.generator import (
    Generator,
    GenerationResult,
    HuggingFaceGenerator,
    RAGPromptTemplate,
    DEFAULT_RAG_PROMPT_TEMPLATE,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_passages() -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            id="p1", title="France", text="Paris is the capital of France.",
            score=0.95, has_answer=True,
        ),
        RetrievedPassage(
            id="p2", title="Europe", text="France is in Western Europe.",
            score=0.8, has_answer=False,
        ),
    ]


@pytest.fixture
def sample_query() -> str:
    return "What is the capital of France?"


# ════════════════════════════════════════════════════════════════════
# GenerationResult
# ════════════════════════════════════════════════════════════════════


class TestGenerationResult:

    def test_basic_construction(self):
        result = GenerationResult(
            answer="Paris",
            prompt="Given context...",
            token_logprobs=[-0.1, -0.3],
            metadata={"model": "test"},
        )
        assert result.answer == "Paris"
        assert result.prompt == "Given context..."
        assert result.token_logprobs == [-0.1, -0.3]
        assert result.metadata["model"] == "test"

    def test_optional_fields(self):
        result = GenerationResult(answer="Paris", prompt="test")
        assert result.token_logprobs is None
        assert result.metadata == {} or result.metadata is not None


# ════════════════════════════════════════════════════════════════════
# RAGPromptTemplate
# ════════════════════════════════════════════════════════════════════


class TestRAGPromptTemplate:

    def test_default_template_formats(self, sample_query, sample_passages):
        template = DEFAULT_RAG_PROMPT_TEMPLATE
        prompt = template.format(query=sample_query, passages=sample_passages)
        assert sample_query in prompt
        assert "Paris is the capital of France" in prompt
        assert "France is in Western Europe" in prompt

    def test_template_includes_passage_titles(self, sample_query, sample_passages):
        template = DEFAULT_RAG_PROMPT_TEMPLATE
        prompt = template.format(query=sample_query, passages=sample_passages)
        assert "France" in prompt
        assert "Europe" in prompt

    def test_template_numbers_passages(self, sample_query, sample_passages):
        template = DEFAULT_RAG_PROMPT_TEMPLATE
        prompt = template.format(query=sample_query, passages=sample_passages)
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_empty_passages(self, sample_query):
        template = DEFAULT_RAG_PROMPT_TEMPLATE
        prompt = template.format(query=sample_query, passages=[])
        assert sample_query in prompt

    def test_custom_template(self, sample_query, sample_passages):
        custom = RAGPromptTemplate(
            template="Q: {query}\nContext: {context}\nA:",
            passage_format="{text}",
            passage_separator=" | ",
        )
        prompt = custom.format(query=sample_query, passages=sample_passages)
        assert "Q: What is the capital of France?" in prompt
        assert "Paris is the capital of France." in prompt


# ════════════════════════════════════════════════════════════════════
# HuggingFaceGenerator — construction (unit, no model loading)
# ════════════════════════════════════════════════════════════════════


class TestHuggingFaceGeneratorConstruction:

    def test_is_generator_subclass(self):
        """Just verify the class structure without loading a model."""
        assert issubclass(HuggingFaceGenerator, Generator)

    def test_stores_model_path(self):
        gen = HuggingFaceGenerator.__new__(HuggingFaceGenerator)
        gen.model_path = "D:\\cc\\models\\test"
        assert gen.model_path == "D:\\cc\\models\\test"


# ════════════════════════════════════════════════════════════════════
# HuggingFaceGenerator — generate (mocked model)
# ════════════════════════════════════════════════════════════════════


class TestHuggingFaceGeneratorMocked:
    """Tests with a mocked model — no GPU required."""

    def _make_generator_with_mock(self):
        """Create a generator with mocked model and tokenizer."""
        gen = HuggingFaceGenerator.__new__(HuggingFaceGenerator)
        gen.model_path = "mock_model"
        gen.device = "cpu"
        gen.prompt_template = DEFAULT_RAG_PROMPT_TEMPLATE
        gen.max_new_tokens = 128

        # Mock tokenizer
        gen.tokenizer = MagicMock()
        gen.tokenizer.return_value = {"input_ids": MagicMock()}
        gen.tokenizer.decode.return_value = "Paris"
        gen.tokenizer.eos_token_id = 2

        # Mock model
        gen.model = MagicMock()
        mock_output = MagicMock()
        mock_output.sequences = MagicMock()
        mock_output.sequences.__getitem__ = MagicMock(return_value=[1, 2, 3])
        mock_output.scores = None
        gen.model.generate.return_value = mock_output

        gen._loaded = True
        return gen

    def test_generate_returns_result(self, sample_query, sample_passages):
        gen = self._make_generator_with_mock()
        result = gen.generate(sample_query, sample_passages)
        assert isinstance(result, GenerationResult)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_generate_includes_prompt(self, sample_query, sample_passages):
        gen = self._make_generator_with_mock()
        result = gen.generate(sample_query, sample_passages)
        assert isinstance(result.prompt, str)
        assert sample_query in result.prompt

    def test_generate_with_empty_passages(self, sample_query):
        gen = self._make_generator_with_mock()
        result = gen.generate(sample_query, [])
        assert isinstance(result, GenerationResult)

    def test_generate_records_model_in_metadata(self, sample_query, sample_passages):
        gen = self._make_generator_with_mock()
        result = gen.generate(sample_query, sample_passages)
        assert "model_path" in result.metadata

    def test_abstain_returns_none_answer(self, sample_query, sample_passages):
        """When decision is abstain, generator should return empty/None answer."""
        gen = self._make_generator_with_mock()
        result = gen.generate(sample_query, sample_passages, abstain=True)
        assert result.answer == "" or result.answer is None


# ════════════════════════════════════════════════════════════════════
# Integration: real model (marked slow + gpu)
# ════════════════════════════════════════════════════════════════════


_AVAILABLE_MODELS = [
    ("Qwen2.5-7B", "D:\\cc\\models\\Qwen2.5-7B-Instruct"),
    ("Mistral-7B-v0.3", "D:\\cc\\models\\Mistral-7B-Instruct-v0.3"),
    ("Llama-3.1-8B", "D:\\cc\\models\\Llama-3.1-8B-Instruct"),
]


def _find_available_model() -> tuple[str, str] | None:
    """Find the first available model on disk."""
    import os
    for name, path in _AVAILABLE_MODELS:
        if os.path.exists(path):
            return name, path
    return None


@pytest.mark.slow
@pytest.mark.gpu
class TestHuggingFaceGeneratorIntegration:
    """Integration tests with a real local LLM.

    Requires one of:
        - Qwen2.5-7B-Instruct at D:\\cc\\models\\Qwen2.5-7B-Instruct
        - Mistral-7B-Instruct-v0.3 at D:\\cc\\models\\Mistral-7B-Instruct-v0.3
    Run with: python -m pytest -m "slow and gpu" tests/test_generation/
    """

    @pytest.fixture(scope="class")
    def model_info(self):
        info = _find_available_model()
        if info is None:
            pytest.skip("No local model found at D:\\cc\\models\\")
        return info

    @pytest.fixture(scope="class")
    def generator(self, model_info):
        """Load model once for all tests in this class."""
        _, model_path = model_info
        return HuggingFaceGenerator(
            model_path=model_path,
            device="cuda",
            load_in_4bit=True,
        )

    def test_generates_answer(self, generator, sample_query, sample_passages):
        result = generator.generate(sample_query, sample_passages)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_answer_mentions_paris(self, generator, sample_query, sample_passages):
        result = generator.generate(sample_query, sample_passages)
        assert "paris" in result.answer.lower() or "Paris" in result.answer

    def test_token_logprobs_available(self, generator, sample_query, sample_passages):
        result = generator.generate(
            sample_query, sample_passages, return_logprobs=True,
        )
        assert result.token_logprobs is not None
        assert len(result.token_logprobs) > 0

    def test_generation_result_complete(self, generator, model_info, sample_query, sample_passages):
        _, model_path = model_info
        result = generator.generate(sample_query, sample_passages)
        assert result.prompt is not None
        assert result.metadata.get("model_path") == model_path
