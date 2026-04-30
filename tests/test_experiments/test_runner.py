"""Tests for experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from xrag.experiment.config import (
    ExperimentConfig,
    DatasetConfig,
    RetrievalConfig,
    CorruptionConfig,
    EstimationConfig,
    PipelineConfig,
    BaselinesConfig,
    GenerationConfig,
    OutputConfig,
)
from xrag.experiment.runner import (
    ExperimentResult,
    run_experiment,
    save_result,
    load_result,
)
from xrag.benchmarks.data_loader import QAExample, Paragraph
from xrag.benchmarks.retriever import RetrievedPassage, RetrievalResult
from xrag.opinion_estimation.base import EstimationResult
from xrag.pipeline.sl_rag_pipeline import PipelineResult
from xrag.generation.generator import GenerationResult
from jsonld_ex.confidence_algebra import Opinion


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def hotpotqa_examples():
    """Minimal HotpotQA examples with gold paragraphs."""
    return [
        QAExample(
            id="hp-1",
            question="What is the capital of France?",
            answers=["Paris"],
            paragraphs=[
                Paragraph(title="France", text="Paris is the capital of France.", is_supporting=True),
                Paragraph(title="Germany", text="Berlin is the capital of Germany.", is_supporting=False),
                Paragraph(title="Europe", text="Europe has many countries.", is_supporting=False),
            ],
            metadata={"type": "bridge", "level": "easy"},
        ),
        QAExample(
            id="hp-2",
            question="Who wrote Romeo and Juliet?",
            answers=["William Shakespeare", "Shakespeare"],
            paragraphs=[
                Paragraph(title="Shakespeare", text="William Shakespeare wrote Romeo and Juliet.", is_supporting=True),
                Paragraph(title="Theatre", text="The Globe Theatre was in London.", is_supporting=False),
            ],
            metadata={"type": "comparison", "level": "medium"},
        ),
        QAExample(
            id="hp-3",
            question="What year did World War 2 end?",
            answers=["1945"],
            paragraphs=[
                Paragraph(title="WW2", text="World War 2 ended in 1945.", is_supporting=True),
                Paragraph(title="History", text="Many wars have been fought.", is_supporting=False),
            ],
            metadata={"type": "bridge", "level": "easy"},
        ),
    ]


@pytest.fixture
def minimal_config(tmp_path):
    """Minimal valid experiment config for testing."""
    return ExperimentConfig(
        name="test_run",
        seed=42,
        dataset=DatasetConfig(name="hotpotqa", split="validation", max_samples=3),
        retrieval=RetrievalConfig(method="gold", top_k=10),
        corruption=CorruptionConfig(type=None, level=0.0),
        estimation=EstimationConfig(method="nli", batch_size=32),
        pipeline=PipelineConfig(),
        baselines=BaselinesConfig(
            run_softmax=True,
            run_retrieval_conf=True,
            run_semantic_entropy=False,
            run_p_true=False,
            run_combined_heuristic=False,
            run_conformal=False,
        ),
        generation=GenerationConfig(model_path="/fake/model"),
        output=OutputConfig(results_dir=str(tmp_path / "results"), save_predictions=True),
    )


@pytest.fixture
def mock_opinion():
    return Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)


# ════════════════════════════════════════════════════════════════════
# ExperimentResult
# ════════════════════════════════════════════════════════════════════


class TestExperimentResult:

    def test_basic_construction(self, minimal_config):
        result = ExperimentResult(
            config=minimal_config,
            metrics={"sl_rag": {"em": 0.8, "f1": 0.85}},
            predictions=[],
            timing={"total": 10.5},
        )
        assert result.config.name == "test_run"
        assert result.metrics["sl_rag"]["em"] == 0.8
        assert result.timing["total"] == 10.5

    def test_predictions_optional(self, minimal_config):
        result = ExperimentResult(
            config=minimal_config,
            metrics={},
            predictions=None,
            timing={},
        )
        assert result.predictions is None

    def test_to_dict(self, minimal_config):
        result = ExperimentResult(
            config=minimal_config,
            metrics={"sl_rag": {"em": 0.8}},
            predictions=[{"id": "hp-1", "answer": "Paris", "correct": True}],
            timing={"load_data": 1.0, "generate": 5.0},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["config"]["name"] == "test_run"
        assert d["metrics"]["sl_rag"]["em"] == 0.8
        assert len(d["predictions"]) == 1
        assert d["timing"]["load_data"] == 1.0

    def test_from_dict_roundtrip(self, minimal_config):
        original = ExperimentResult(
            config=minimal_config,
            metrics={"sl_rag": {"em": 0.75, "f1": 0.82}},
            predictions=[{"id": "hp-1", "answer": "Paris"}],
            timing={"total": 42.0},
        )
        d = original.to_dict()
        loaded = ExperimentResult.from_dict(d)
        assert loaded.config.name == original.config.name
        assert loaded.metrics == original.metrics
        assert loaded.predictions == original.predictions
        assert loaded.timing == original.timing


# ════════════════════════════════════════════════════════════════════
# Result serialization
# ════════════════════════════════════════════════════════════════════


class TestResultSerialization:

    def test_save_creates_file(self, minimal_config, tmp_path):
        result = ExperimentResult(
            config=minimal_config,
            metrics={"sl_rag": {"em": 0.8}},
            predictions=None,
            timing={"total": 1.0},
        )
        filepath = tmp_path / "result.json"
        save_result(result, str(filepath))
        assert filepath.exists()

    def test_save_load_roundtrip(self, minimal_config, tmp_path):
        original = ExperimentResult(
            config=minimal_config,
            metrics={"sl_rag": {"em": 0.8, "f1": 0.85}},
            predictions=[{"id": "hp-1", "pred": "Paris", "gold": ["Paris"]}],
            timing={"load": 0.5, "estimate": 2.0, "generate": 5.0, "total": 7.5},
        )
        filepath = tmp_path / "result.json"
        save_result(original, str(filepath))
        loaded = load_result(str(filepath))

        assert loaded.config.name == original.config.name
        assert loaded.config.seed == original.config.seed
        assert loaded.metrics == original.metrics
        assert loaded.predictions == original.predictions
        assert loaded.timing == original.timing

    def test_saved_json_is_valid(self, minimal_config, tmp_path):
        result = ExperimentResult(
            config=minimal_config,
            metrics={"sl_rag": {"em": 0.8}},
            predictions=None,
            timing={"total": 1.0},
        )
        filepath = tmp_path / "result.json"
        save_result(result, str(filepath))

        with open(filepath) as f:
            raw = json.load(f)
        assert isinstance(raw, dict)
        assert "config" in raw
        assert "metrics" in raw
        assert "timing" in raw


# ════════════════════════════════════════════════════════════════════
# Runner wiring (mocked components)
# ════════════════════════════════════════════════════════════════════


class TestRunnerWiring:
    """Tests that run_experiment wires components correctly.

    All heavy components (dataset loader, NLI model, LLM) are mocked.
    """

    def _mock_loader(self, examples):
        """Create a mock dataset loader returning given examples."""
        loader = MagicMock()
        loader.load.return_value = examples
        return loader

    def _mock_estimator(self, opinion):
        """Create a mock NLI estimator returning a fixed opinion."""
        estimator = MagicMock()
        estimator.estimate.return_value = EstimationResult(opinion=opinion)
        estimator.estimate_batch.return_value = [
            EstimationResult(opinion=opinion)
        ]
        return estimator

    def _mock_generator(self):
        """Create a mock LLM generator."""
        generator = MagicMock()
        generator.generate.return_value = GenerationResult(
            answer="Paris",
            prompt="test prompt",
            token_logprobs=[-0.1, -0.2, -0.3],
            metadata={"model_path": "/fake", "abstained": False,
                       "input_tokens": 50, "output_tokens": 3},
        )
        return generator

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_returns_experiment_result(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        minimal_config, hotpotqa_examples, mock_opinion,
    ):
        mock_loader_fn.return_value = self._mock_loader(hotpotqa_examples)
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        result = run_experiment(minimal_config)

        assert isinstance(result, ExperimentResult)
        assert result.config.name == "test_run"

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_metrics_contain_sl_rag(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        minimal_config, hotpotqa_examples, mock_opinion,
    ):
        mock_loader_fn.return_value = self._mock_loader(hotpotqa_examples)
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        result = run_experiment(minimal_config)

        assert "sl_rag" in result.metrics
        assert "em" in result.metrics["sl_rag"]
        assert "f1" in result.metrics["sl_rag"]

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_enabled_baselines_in_metrics(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        minimal_config, hotpotqa_examples, mock_opinion,
    ):
        mock_loader_fn.return_value = self._mock_loader(hotpotqa_examples)
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        result = run_experiment(minimal_config)

        # softmax and retrieval_conf are enabled in minimal_config
        assert "softmax" in result.metrics
        assert "retrieval_conf" in result.metrics
        # semantic_entropy, p_true, combined, conformal are disabled
        assert "semantic_entropy" not in result.metrics

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_predictions_saved_when_enabled(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        minimal_config, hotpotqa_examples, mock_opinion,
    ):
        mock_loader_fn.return_value = self._mock_loader(hotpotqa_examples)
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        result = run_experiment(minimal_config)

        assert result.predictions is not None
        assert len(result.predictions) == len(hotpotqa_examples)

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_predictions_contain_required_fields(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        minimal_config, hotpotqa_examples, mock_opinion,
    ):
        mock_loader_fn.return_value = self._mock_loader(hotpotqa_examples)
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        result = run_experiment(minimal_config)

        pred = result.predictions[0]
        assert "id" in pred
        assert "question" in pred
        assert "gold_answers" in pred
        assert "predicted_answer" in pred
        assert "sl_rag_confidence" in pred
        assert "sl_rag_decision" in pred
        assert "em" in pred
        assert "f1" in pred

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_timing_has_stages(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        minimal_config, hotpotqa_examples, mock_opinion,
    ):
        mock_loader_fn.return_value = self._mock_loader(hotpotqa_examples)
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        result = run_experiment(minimal_config)

        assert "load_data" in result.timing
        assert "estimate_opinions" in result.timing
        assert "run_pipeline" in result.timing
        assert "generate" in result.timing
        assert "evaluate" in result.timing
        assert "total" in result.timing

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_no_predictions_when_disabled(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        hotpotqa_examples, mock_opinion, tmp_path,
    ):
        config = ExperimentConfig(
            name="no_preds",
            seed=42,
            dataset=DatasetConfig(name="hotpotqa", split="validation", max_samples=3),
            retrieval=RetrievalConfig(method="gold", top_k=10),
            corruption=CorruptionConfig(type=None, level=0.0),
            estimation=EstimationConfig(method="nli", batch_size=32),
            pipeline=PipelineConfig(),
            baselines=BaselinesConfig(
                run_softmax=False, run_retrieval_conf=False,
                run_p_true=False, run_combined_heuristic=False,
                run_conformal=False,
            ),
            generation=GenerationConfig(model_path="/fake"),
            output=OutputConfig(
                results_dir=str(tmp_path / "results"),
                save_predictions=False,
            ),
        )
        mock_loader_fn.return_value = self._mock_loader(hotpotqa_examples)
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        result = run_experiment(config)
        assert result.predictions is None

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_loader_called_with_config_args(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        minimal_config, hotpotqa_examples, mock_opinion,
    ):
        mock_loader = self._mock_loader(hotpotqa_examples)
        mock_loader_fn.return_value = mock_loader
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        run_experiment(minimal_config)

        mock_loader.load.assert_called_once_with(
            split="validation", max_samples=3,
        )

    @patch("xrag.experiment.runner._build_loader")
    @patch("xrag.experiment.runner._build_estimator")
    @patch("xrag.experiment.runner._build_generator")
    def test_seed_is_set(
        self, mock_gen_fn, mock_est_fn, mock_loader_fn,
        minimal_config, hotpotqa_examples, mock_opinion,
    ):
        mock_loader_fn.return_value = self._mock_loader(hotpotqa_examples)
        mock_est_fn.return_value = self._mock_estimator(mock_opinion)
        mock_gen_fn.return_value = self._mock_generator()

        with patch("xrag.experiment.runner._set_seed") as mock_seed:
            run_experiment(minimal_config)
            mock_seed.assert_called_once_with(42)
