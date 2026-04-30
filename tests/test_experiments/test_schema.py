"""Tests for experiment configuration schema."""

from __future__ import annotations

import pytest
import yaml

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
    load_config,
    save_config,
)


# ════════════════════════════════════════════════════════════════════
# DatasetConfig
# ════════════════════════════════════════════════════════════════════


class TestDatasetConfig:

    def test_valid_hotpotqa(self):
        cfg = DatasetConfig(name="hotpotqa", split="validation", max_samples=100)
        assert cfg.name == "hotpotqa"
        assert cfg.split == "validation"
        assert cfg.max_samples == 100

    def test_valid_musique(self):
        cfg = DatasetConfig(name="musique", split="validation", max_samples=None)
        assert cfg.name == "musique"
        assert cfg.max_samples is None

    def test_valid_nq(self):
        cfg = DatasetConfig(name="nq", split="validation")
        assert cfg.name == "nq"

    def test_valid_popqa(self):
        cfg = DatasetConfig(name="popqa", split="test")
        assert cfg.name == "popqa"

    def test_invalid_dataset_name(self):
        with pytest.raises(ValueError, match="dataset"):
            DatasetConfig(name="invalid_dataset", split="validation")

    def test_max_samples_must_be_positive(self):
        with pytest.raises(ValueError, match="max_samples"):
            DatasetConfig(name="hotpotqa", split="validation", max_samples=0)

    def test_max_samples_negative_rejected(self):
        with pytest.raises(ValueError, match="max_samples"):
            DatasetConfig(name="hotpotqa", split="validation", max_samples=-5)


# ════════════════════════════════════════════════════════════════════
# RetrievalConfig
# ════════════════════════════════════════════════════════════════════


class TestRetrievalConfig:

    def test_gold_retrieval(self):
        cfg = RetrievalConfig(method="gold", top_k=10)
        assert cfg.method == "gold"
        assert cfg.top_k == 10

    def test_precomputed_retrieval(self):
        cfg = RetrievalConfig(
            method="precomputed", top_k=10,
            precomputed_path="/path/to/retrieval.jsonl.gz",
        )
        assert cfg.method == "precomputed"
        assert cfg.precomputed_path == "/path/to/retrieval.jsonl.gz"

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            RetrievalConfig(method="live", top_k=10)

    def test_precomputed_requires_path(self):
        with pytest.raises(ValueError, match="precomputed_path"):
            RetrievalConfig(method="precomputed", top_k=10)

    def test_top_k_must_be_positive(self):
        with pytest.raises(ValueError, match="top_k"):
            RetrievalConfig(method="gold", top_k=0)


# ════════════════════════════════════════════════════════════════════
# CorruptionConfig
# ════════════════════════════════════════════════════════════════════


class TestCorruptionConfig:

    def test_no_corruption(self):
        cfg = CorruptionConfig(type=None, level=0.0)
        assert cfg.type is None

    def test_distractor_corruption(self):
        cfg = CorruptionConfig(type="distractor", level=0.5, seed=42)
        assert cfg.type == "distractor"
        assert cfg.level == 0.5
        assert cfg.seed == 42

    def test_contradiction_corruption(self):
        cfg = CorruptionConfig(type="contradiction", level=0.25)
        assert cfg.type == "contradiction"

    def test_missing_corruption(self):
        cfg = CorruptionConfig(type="missing", level=0.75)
        assert cfg.type == "missing"

    def test_stale_corruption(self):
        cfg = CorruptionConfig(type="stale", level=0.5)
        assert cfg.type == "stale"

    def test_adversarial_corruption(self):
        cfg = CorruptionConfig(type="adversarial", level=0.25)
        assert cfg.type == "adversarial"

    def test_invalid_corruption_type(self):
        with pytest.raises(ValueError, match="type"):
            CorruptionConfig(type="random_noise", level=0.5)

    def test_level_must_be_in_range(self):
        with pytest.raises(ValueError, match="level"):
            CorruptionConfig(type="distractor", level=1.5)

    def test_level_negative_rejected(self):
        with pytest.raises(ValueError, match="level"):
            CorruptionConfig(type="distractor", level=-0.1)

    def test_type_none_level_must_be_zero(self):
        with pytest.raises(ValueError, match="level"):
            CorruptionConfig(type=None, level=0.5)


# ════════════════════════════════════════════════════════════════════
# EstimationConfig
# ════════════════════════════════════════════════════════════════════


class TestEstimationConfig:

    def test_nli_estimation(self):
        cfg = EstimationConfig(method="nli", batch_size=32)
        assert cfg.method == "nli"
        assert cfg.batch_size == 32

    def test_llm_judge_estimation(self):
        cfg = EstimationConfig(method="llm_judge", batch_size=8)
        assert cfg.method == "llm_judge"

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            EstimationConfig(method="random")

    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValueError, match="batch_size"):
            EstimationConfig(method="nli", batch_size=0)


# ════════════════════════════════════════════════════════════════════
# PipelineConfig
# ════════════════════════════════════════════════════════════════════


class TestPipelineConfig:

    def test_default_pipeline(self):
        cfg = PipelineConfig()
        assert cfg.fusion_strategy == "cumulative"
        assert cfg.tau_abstain == 0.7
        assert cfg.tau_conflict == 0.5
        assert cfg.ablation is None

    def test_custom_pipeline(self):
        cfg = PipelineConfig(
            fusion_strategy="averaging",
            default_trust_belief=0.8,
            half_life=336.0,
            conflict_threshold=0.4,
            tau_abstain=0.6,
            tau_conflict=0.4,
        )
        assert cfg.fusion_strategy == "averaging"
        assert cfg.half_life == 336.0

    def test_ablation_no_trust(self):
        cfg = PipelineConfig(ablation="no_trust")
        assert cfg.ablation == "no_trust"

    def test_valid_ablations(self):
        valid = [
            "no_trust", "no_temporal", "no_conflict", "no_fusion",
            "no_deduction", "no_abstention", "scalar_confidence",
        ]
        for abl in valid:
            cfg = PipelineConfig(ablation=abl)
            assert cfg.ablation == abl

    def test_invalid_ablation(self):
        with pytest.raises(ValueError, match="ablation"):
            PipelineConfig(ablation="invalid_ablation")

    def test_invalid_fusion_strategy(self):
        with pytest.raises(ValueError, match="fusion_strategy"):
            PipelineConfig(fusion_strategy="nonexistent")

    def test_tau_abstain_range(self):
        with pytest.raises(ValueError, match="tau_abstain"):
            PipelineConfig(tau_abstain=1.5)

    def test_tau_conflict_range(self):
        with pytest.raises(ValueError, match="tau_conflict"):
            PipelineConfig(tau_conflict=-0.1)


# ════════════════════════════════════════════════════════════════════
# BaselinesConfig
# ════════════════════════════════════════════════════════════════════


class TestBaselinesConfig:

    def test_default_all_enabled(self):
        cfg = BaselinesConfig()
        assert cfg.run_softmax is True
        assert cfg.run_retrieval_conf is True
        assert cfg.run_semantic_entropy is False  # expensive, off by default
        assert cfg.run_p_true is True
        assert cfg.run_combined_heuristic is True
        assert cfg.run_conformal is True

    def test_semantic_entropy_samples(self):
        cfg = BaselinesConfig(
            run_semantic_entropy=True,
            semantic_entropy_n_samples=5,
        )
        assert cfg.semantic_entropy_n_samples == 5

    def test_semantic_entropy_samples_must_be_positive(self):
        with pytest.raises(ValueError, match="semantic_entropy_n_samples"):
            BaselinesConfig(
                run_semantic_entropy=True,
                semantic_entropy_n_samples=0,
            )


# ════════════════════════════════════════════════════════════════════
# GenerationConfig
# ════════════════════════════════════════════════════════════════════


class TestGenerationConfig:

    def test_basic_config(self):
        cfg = GenerationConfig(model_path="/models/Llama-3.1-8B-Instruct")
        assert cfg.model_path == "/models/Llama-3.1-8B-Instruct"
        assert cfg.load_in_4bit is False
        assert cfg.max_new_tokens == 128
        assert cfg.device == "cuda"

    def test_4bit_quantization(self):
        cfg = GenerationConfig(
            model_path="/models/Llama-3.1-8B-Instruct",
            load_in_4bit=True,
        )
        assert cfg.load_in_4bit is True

    def test_max_new_tokens_must_be_positive(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            GenerationConfig(model_path="/m", max_new_tokens=0)


# ════════════════════════════════════════════════════════════════════
# OutputConfig
# ════════════════════════════════════════════════════════════════════


class TestOutputConfig:

    def test_default_output(self):
        cfg = OutputConfig(results_dir="experiments/results")
        assert cfg.results_dir == "experiments/results"
        assert cfg.save_predictions is True

    def test_no_predictions(self):
        cfg = OutputConfig(results_dir="results", save_predictions=False)
        assert cfg.save_predictions is False


# ════════════════════════════════════════════════════════════════════
# ExperimentConfig (top-level)
# ════════════════════════════════════════════════════════════════════


class TestExperimentConfig:

    def _make_minimal_config(self, **overrides) -> ExperimentConfig:
        defaults = dict(
            name="test_run",
            seed=42,
            dataset=DatasetConfig(name="hotpotqa", split="validation", max_samples=10),
            retrieval=RetrievalConfig(method="gold", top_k=10),
            corruption=CorruptionConfig(type=None, level=0.0),
            estimation=EstimationConfig(method="nli", batch_size=32),
            pipeline=PipelineConfig(),
            baselines=BaselinesConfig(),
            generation=GenerationConfig(model_path="/models/test"),
            output=OutputConfig(results_dir="experiments/results"),
        )
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    def test_minimal_valid_config(self):
        cfg = self._make_minimal_config()
        assert cfg.name == "test_run"
        assert cfg.seed == 42

    def test_config_has_all_sections(self):
        cfg = self._make_minimal_config()
        assert isinstance(cfg.dataset, DatasetConfig)
        assert isinstance(cfg.retrieval, RetrievalConfig)
        assert isinstance(cfg.corruption, CorruptionConfig)
        assert isinstance(cfg.estimation, EstimationConfig)
        assert isinstance(cfg.pipeline, PipelineConfig)
        assert isinstance(cfg.baselines, BaselinesConfig)
        assert isinstance(cfg.generation, GenerationConfig)
        assert isinstance(cfg.output, OutputConfig)

    def test_name_must_not_be_empty(self):
        with pytest.raises(ValueError, match="name"):
            self._make_minimal_config(name="")


# ════════════════════════════════════════════════════════════════════
# YAML serialization round-trip
# ════════════════════════════════════════════════════════════════════


class TestConfigSerialization:

    def _make_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name="hotpotqa_llama8b_clean",
            seed=42,
            dataset=DatasetConfig(name="hotpotqa", split="validation", max_samples=100),
            retrieval=RetrievalConfig(method="gold", top_k=10),
            corruption=CorruptionConfig(type=None, level=0.0),
            estimation=EstimationConfig(method="nli", batch_size=32),
            pipeline=PipelineConfig(
                fusion_strategy="cumulative",
                tau_abstain=0.7,
                tau_conflict=0.5,
            ),
            baselines=BaselinesConfig(
                run_softmax=True,
                run_semantic_entropy=False,
            ),
            generation=GenerationConfig(
                model_path="/models/Llama-3.1-8B-Instruct",
                load_in_4bit=True,
            ),
            output=OutputConfig(results_dir="experiments/results"),
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        original = self._make_config()
        filepath = tmp_path / "test_config.yaml"

        save_config(original, str(filepath))
        loaded = load_config(str(filepath))

        assert loaded.name == original.name
        assert loaded.seed == original.seed
        assert loaded.dataset.name == original.dataset.name
        assert loaded.dataset.max_samples == original.dataset.max_samples
        assert loaded.retrieval.method == original.retrieval.method
        assert loaded.retrieval.top_k == original.retrieval.top_k
        assert loaded.corruption.type == original.corruption.type
        assert loaded.estimation.method == original.estimation.method
        assert loaded.pipeline.fusion_strategy == original.pipeline.fusion_strategy
        assert loaded.pipeline.tau_abstain == original.pipeline.tau_abstain
        assert loaded.baselines.run_softmax == original.baselines.run_softmax
        assert loaded.baselines.run_semantic_entropy == original.baselines.run_semantic_entropy
        assert loaded.generation.model_path == original.generation.model_path
        assert loaded.generation.load_in_4bit == original.generation.load_in_4bit
        assert loaded.output.results_dir == original.output.results_dir

    def test_saved_yaml_is_readable(self, tmp_path):
        cfg = self._make_config()
        filepath = tmp_path / "test_config.yaml"
        save_config(cfg, str(filepath))

        with open(filepath) as f:
            raw = yaml.safe_load(f)
        assert isinstance(raw, dict)
        assert raw["name"] == "hotpotqa_llama8b_clean"
        assert raw["dataset"]["name"] == "hotpotqa"

    def test_load_from_dict(self):
        raw = {
            "name": "test_from_dict",
            "seed": 99,
            "dataset": {"name": "musique", "split": "validation", "max_samples": 50},
            "retrieval": {"method": "gold", "top_k": 5},
            "corruption": {"type": None, "level": 0.0},
            "estimation": {"method": "nli", "batch_size": 16},
            "pipeline": {"fusion_strategy": "averaging"},
            "baselines": {"run_softmax": True},
            "generation": {"model_path": "/models/test"},
            "output": {"results_dir": "results"},
        }
        cfg = ExperimentConfig.from_dict(raw)
        assert cfg.name == "test_from_dict"
        assert cfg.dataset.name == "musique"
        assert cfg.pipeline.fusion_strategy == "averaging"

    def test_to_dict(self):
        cfg = self._make_config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "hotpotqa_llama8b_clean"
        assert d["dataset"]["name"] == "hotpotqa"
        assert d["generation"]["load_in_4bit"] is True
