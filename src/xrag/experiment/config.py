"""Experiment configuration schema with validation.

Defines a hierarchy of dataclasses that fully specify an experiment run.
Supports YAML serialization for reproducibility and packaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DATASETS = {"hotpotqa", "musique", "nq", "popqa"}
VALID_RETRIEVAL_METHODS = {"gold", "precomputed"}
VALID_CORRUPTION_TYPES = {
    "distractor", "contradiction", "missing", "stale", "adversarial",
}
VALID_ESTIMATION_METHODS = {"nli", "llm_judge"}
VALID_FUSION_STRATEGIES = {
    "cumulative", "averaging", "robust",
    "byzantine_most_conflicting", "byzantine_least_trusted", "byzantine_combined",
}
VALID_ABLATIONS = {
    "no_trust", "no_temporal", "no_conflict", "no_fusion",
    "no_deduction", "no_abstention", "scalar_confidence",
}


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Which benchmark dataset to use.

    Attributes:
        name: One of 'hotpotqa', 'musique', 'nq', 'popqa'.
        split: Dataset split (e.g. 'validation', 'test').
        max_samples: Maximum examples to load, or None for all.
    """

    name: str
    split: str
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if self.name not in VALID_DATASETS:
            raise ValueError(
                f"Invalid dataset name '{self.name}'. "
                f"Must be one of: {sorted(VALID_DATASETS)}"
            )
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(
                f"max_samples must be positive, got {self.max_samples}"
            )


@dataclass
class RetrievalConfig:
    """How to retrieve passages.

    Attributes:
        method: 'gold' (from dataset) or 'precomputed' (from file).
        top_k: Maximum passages per query.
        precomputed_path: Path to JSONL file (required if method='precomputed').
    """

    method: str
    top_k: int = 10
    precomputed_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.method not in VALID_RETRIEVAL_METHODS:
            raise ValueError(
                f"Invalid retrieval method '{self.method}'. "
                f"Must be one of: {sorted(VALID_RETRIEVAL_METHODS)}"
            )
        if self.method == "precomputed" and not self.precomputed_path:
            raise ValueError(
                "precomputed_path is required when method='precomputed'"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")


@dataclass
class CorruptionConfig:
    """Retrieval corruption settings for robustness testing.

    Attributes:
        type: Corruption type, or None for clean evaluation.
        level: Fraction of passages to corrupt (0.0 to 1.0).
        seed: Random seed for reproducible corruption.
    """

    type: Optional[str] = None
    level: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.type is not None and self.type not in VALID_CORRUPTION_TYPES:
            raise ValueError(
                f"Invalid corruption type '{self.type}'. "
                f"Must be one of: {sorted(VALID_CORRUPTION_TYPES)} or None"
            )
        if not 0.0 <= self.level <= 1.0:
            raise ValueError(
                f"Corruption level must be in [0.0, 1.0], got {self.level}"
            )
        if self.type is None and self.level != 0.0:
            raise ValueError(
                f"Corruption level must be 0.0 when type is None, got {self.level}"
            )


@dataclass
class EstimationConfig:
    """Opinion estimation settings.

    Attributes:
        method: 'nli' (DeBERTa) or 'llm_judge' (LLM self-assessment).
        batch_size: Batch size for NLI inference.
    """

    method: str = "nli"
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.method not in VALID_ESTIMATION_METHODS:
            raise ValueError(
                f"Invalid estimation method '{self.method}'. "
                f"Must be one of: {sorted(VALID_ESTIMATION_METHODS)}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )


@dataclass
class PipelineConfig:
    """SL-RAG pipeline layer settings.

    Attributes:
        fusion_strategy: Fusion operator to use.
        default_trust_belief: Trust belief for all sources (Wikipedia default).
        half_life: Temporal decay half-life in hours.
        conflict_threshold: Pairwise conflict threshold for flagging.
        tau_abstain: Uncertainty threshold for abstention.
        tau_conflict: Aggregate conflict score threshold.
        ablation: Which component to disable, or None for full pipeline.
    """

    fusion_strategy: str = "cumulative"
    default_trust_belief: float = 0.9
    half_life: float = 168.0
    conflict_threshold: float = 0.3
    tau_abstain: float = 0.7
    tau_conflict: float = 0.5
    ablation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.fusion_strategy not in VALID_FUSION_STRATEGIES:
            raise ValueError(
                f"Invalid fusion_strategy '{self.fusion_strategy}'. "
                f"Must be one of: {sorted(VALID_FUSION_STRATEGIES)}"
            )
        if self.ablation is not None and self.ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Invalid ablation '{self.ablation}'. "
                f"Must be one of: {sorted(VALID_ABLATIONS)} or None"
            )
        if not 0.0 <= self.tau_abstain <= 1.0:
            raise ValueError(
                f"tau_abstain must be in [0.0, 1.0], got {self.tau_abstain}"
            )
        if not 0.0 <= self.tau_conflict <= 1.0:
            raise ValueError(
                f"tau_conflict must be in [0.0, 1.0], got {self.tau_conflict}"
            )


@dataclass
class BaselinesConfig:
    """Which UQ baselines to run alongside SL-RAG.

    Attributes:
        run_softmax: Softmax confidence (free, post-hoc from logits).
        run_retrieval_conf: Retrieval score confidence (free).
        run_semantic_entropy: Semantic entropy (expensive, N extra generations).
        run_p_true: P(True) self-evaluation (1 extra LLM call).
        run_combined_heuristic: Combined retrieval × generation confidence.
        run_conformal: Conformal prediction (free after calibration).
        semantic_entropy_n_samples: Number of generations for semantic entropy.
    """

    run_softmax: bool = True
    run_retrieval_conf: bool = True
    run_semantic_entropy: bool = False  # expensive, opt-in
    run_p_true: bool = True
    run_combined_heuristic: bool = True
    run_conformal: bool = True
    semantic_entropy_n_samples: int = 5

    def __post_init__(self) -> None:
        if self.semantic_entropy_n_samples <= 0:
            raise ValueError(
                f"semantic_entropy_n_samples must be positive, "
                f"got {self.semantic_entropy_n_samples}"
            )


@dataclass
class GenerationConfig:
    """LLM generation settings.

    Attributes:
        model_path: Path to local model directory.
        load_in_4bit: Whether to use 4-bit quantization.
        max_new_tokens: Maximum tokens to generate per answer.
        device: Device for inference ('cuda', 'cpu', 'auto').
    """

    model_path: str
    load_in_4bit: bool = False
    max_new_tokens: int = 128
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {self.max_new_tokens}"
            )


@dataclass
class OutputConfig:
    """Where and how to save results.

    Attributes:
        results_dir: Directory for result JSON files.
        save_predictions: Whether to save per-example predictions.
    """

    results_dir: str = "experiments/results"
    save_predictions: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Complete specification for a single experiment run.

    One config file = one reproducible experiment. The config is embedded
    in the result JSON so every output is self-describing.

    Attributes:
        name: Unique identifier for this experiment run.
        seed: Global random seed for reproducibility.
        dataset: Which benchmark to use.
        retrieval: How to retrieve passages.
        corruption: Optional retrieval corruption.
        estimation: Opinion estimation settings.
        pipeline: SL-RAG pipeline settings.
        baselines: Which UQ baselines to run.
        generation: LLM generation settings.
        output: Output directory and format settings.
    """

    name: str
    seed: int
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    corruption: CorruptionConfig
    estimation: EstimationConfig
    pipeline: PipelineConfig
    baselines: BaselinesConfig
    generation: GenerationConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Experiment name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for YAML serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Construct from a plain dict (e.g. loaded from YAML).

        Sub-configs use their defaults for any missing keys.
        """
        return cls(
            name=d["name"],
            seed=d.get("seed", 42),
            dataset=DatasetConfig(**d["dataset"]),
            retrieval=RetrievalConfig(**d["retrieval"]),
            corruption=CorruptionConfig(**d.get("corruption", {})),
            estimation=EstimationConfig(**d.get("estimation", {})),
            pipeline=PipelineConfig(**d.get("pipeline", {})),
            baselines=BaselinesConfig(**d.get("baselines", {})),
            generation=GenerationConfig(**d["generation"]),
            output=OutputConfig(**d.get("output", {})),
        )


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def save_config(config: ExperimentConfig, filepath: str) -> None:
    """Save an experiment config to a YAML file.

    Args:
        config: The config to save.
        filepath: Path to the output YAML file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(filepath: str) -> ExperimentConfig:
    """Load an experiment config from a YAML file.

    Args:
        filepath: Path to the YAML file.

    Returns:
        Validated ExperimentConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the config is invalid.
    """
    with open(filepath) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.from_dict(raw)
