"""Generate experiment configs for ablations and corruption sweeps.

Usage:
    python experiments/scripts/generate_configs.py

Produces YAML configs in experiments/configs/ for:
    - 7 ablations × 2 benchmarks (HotpotQA, MuSiQue) on Llama 8B
    - 4 corruption types × 4 levels × 2 benchmarks on Llama 8B
    - 4 corruption types × 4 levels × 2 benchmarks on Llama 70B (A100)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path so we can import xrag
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

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
    save_config,
)


# ---------------------------------------------------------------------------
# Base configs
# ---------------------------------------------------------------------------

DATASETS = {
    "hotpotqa": {"split": "validation", "max_samples": 500, "top_k": 10},
    "musique": {"split": "validation", "max_samples": 500, "top_k": 20},
}

MODELS_4090 = {
    "llama8b": r"E:\data\code\claudecode\models\Llama-3.1-8B-Instruct",
    "qwen7b": r"E:\data\code\claudecode\models\Qwen2.5-7B-Instruct",
}

MODELS_A100 = {
    "llama70b": "./models/Llama-3.1-70B-Instruct",
    "qwen72b": "./models/Qwen2.5-72B-Instruct",
}

ABLATIONS = [
    "no_trust",
    "no_temporal",
    "no_conflict",
    "no_fusion",
    "no_deduction",
    "no_abstention",
    "scalar_confidence",
]

CORRUPTION_TYPES = ["distractor", "contradiction", "missing", "adversarial"]
CORRUPTION_LEVELS = [0.25, 0.50, 0.75]


def _base_config(
    name: str,
    dataset_name: str,
    model_path: str,
    model_tag: str,
    load_in_4bit: bool = True,
    device: str = "cuda",
    run_semantic_entropy: bool = False,
    ablation: str | None = None,
    corruption_type: str | None = None,
    corruption_level: float = 0.0,
) -> ExperimentConfig:
    ds = DATASETS[dataset_name]
    return ExperimentConfig(
        name=name,
        seed=42,
        dataset=DatasetConfig(
            name=dataset_name,
            split=ds["split"],
            max_samples=ds["max_samples"],
        ),
        retrieval=RetrievalConfig(method="gold", top_k=ds["top_k"]),
        corruption=CorruptionConfig(
            type=corruption_type,
            level=corruption_level,
        ),
        estimation=EstimationConfig(method="nli", batch_size=32),
        pipeline=PipelineConfig(
            fusion_strategy="cumulative",
            ablation=ablation,
        ),
        baselines=BaselinesConfig(
            run_softmax=True,
            run_retrieval_conf=True,
            run_semantic_entropy=run_semantic_entropy,
            run_p_true=True,
            run_combined_heuristic=True,
            run_conformal=True,
        ),
        generation=GenerationConfig(
            model_path=model_path,
            load_in_4bit=load_in_4bit,
            device=device,
        ),
        output=OutputConfig(
            results_dir="experiments/results",
            save_predictions=True,
        ),
    )


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_ablation_configs(out_dir: Path) -> int:
    """Generate ablation configs for Llama 8B on both benchmarks."""
    abl_dir = out_dir / "ablations"
    abl_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for ds_name in DATASETS:
        for ablation in ABLATIONS:
            name = f"{ds_name}_llama8b_abl_{ablation}"
            cfg = _base_config(
                name=name,
                dataset_name=ds_name,
                model_path=MODELS_4090["llama8b"],
                model_tag="llama8b",
                ablation=ablation,
            )
            save_config(cfg, str(abl_dir / f"{name}.yaml"))
            count += 1

    return count


def generate_corruption_configs_4090(out_dir: Path) -> int:
    """Generate corruption sweep configs for 4090 (Llama 8B)."""
    corr_dir = out_dir / "corruption"
    corr_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for ds_name in DATASETS:
        for ctype in CORRUPTION_TYPES:
            for level in CORRUPTION_LEVELS:
                level_tag = f"{int(level * 100)}"
                name = f"{ds_name}_llama8b_{ctype}_{level_tag}"
                cfg = _base_config(
                    name=name,
                    dataset_name=ds_name,
                    model_path=MODELS_4090["llama8b"],
                    model_tag="llama8b",
                    corruption_type=ctype,
                    corruption_level=level,
                )
                save_config(cfg, str(corr_dir / f"{name}.yaml"))
                count += 1

    return count


def generate_corruption_configs_a100(out_dir: Path) -> int:
    """Generate corruption sweep configs for A100 (Llama 70B)."""
    corr_dir = out_dir / "a100" / "corruption"
    corr_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for ds_name in DATASETS:
        for ctype in CORRUPTION_TYPES:
            for level in CORRUPTION_LEVELS:
                level_tag = f"{int(level * 100)}"
                name = f"{ds_name}_llama70b_{ctype}_{level_tag}"
                cfg = _base_config(
                    name=name,
                    dataset_name=ds_name,
                    model_path=MODELS_A100["llama70b"],
                    model_tag="llama70b",
                    load_in_4bit=False,
                    device="auto",
                    run_semantic_entropy=True,
                    corruption_type=ctype,
                    corruption_level=level,
                )
                save_config(cfg, str(corr_dir / f"{name}.yaml"))
                count += 1

    return count


def generate_a100_qwen_configs(out_dir: Path) -> int:
    """Generate Qwen 72B clean configs for A100 (if time permits)."""
    a100_dir = out_dir / "a100"
    a100_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for ds_name in DATASETS:
        ds = DATASETS[ds_name]
        name = f"{ds_name}_qwen72b_clean"
        cfg = _base_config(
            name=name,
            dataset_name=ds_name,
            model_path=MODELS_A100["qwen72b"],
            model_tag="qwen72b",
            load_in_4bit=False,
            device="auto",
            run_semantic_entropy=True,
        )
        save_config(cfg, str(a100_dir / f"{name}.yaml"))
        count += 1

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_abl = generate_ablation_configs(out_dir)
    print(f"Generated {n_abl} ablation configs")

    n_corr = generate_corruption_configs_4090(out_dir)
    print(f"Generated {n_corr} corruption configs (4090)")

    n_corr_a100 = generate_corruption_configs_a100(out_dir)
    print(f"Generated {n_corr_a100} corruption configs (A100)")

    n_qwen = generate_a100_qwen_configs(out_dir)
    print(f"Generated {n_qwen} Qwen 72B configs (A100)")

    total = n_abl + n_corr + n_corr_a100 + n_qwen
    print(f"\nTotal: {total} configs generated in {out_dir}")


if __name__ == "__main__":
    main()
