"""Experiment runner — orchestrates a single experiment from config to results.

Usage:
    # As a library
    from xrag.experiment.config import load_config
    from xrag.experiment.runner import run_experiment, save_result
    config = load_config("experiments/configs/hotpotqa_main.yaml")
    result = run_experiment(config)
    save_result(result, "experiments/results/hotpotqa_main.json")

    # As CLI
    python -m xrag.experiment.runner --config experiments/configs/hotpotqa_main.yaml
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from xrag.benchmarks.data_loader import (
    DatasetLoader,
    HotpotQALoader,
    MuSiQueLoader,
    NQLoader,
    PopQALoader,
    QAExample,
)
from xrag.benchmarks.retriever import (
    GoldRetriever,
    PrecomputedRetriever,
    RetrievalResult,
    RetrievedPassage,
    Retriever,
)
from xrag.benchmarks.corruption import (
    CorruptionConfig as CorrCfg,
    CorruptedResult,
    inject_distractors,
    inject_contradictions,
    remove_evidence,
    backdate_timestamps,
    inject_adversarial,
)
from xrag.evaluation import exact_match, token_f1, batch_em, batch_f1
from xrag.evaluation import (
    expected_calibration_error,
    brier_score,
    auroc_selective,
    auprc_selective,
    e_aurc,
)
from xrag.experiment.config import ExperimentConfig
from xrag.generation.generator import GenerationResult, HuggingFaceGenerator
from xrag.opinion_estimation.base import BaseOpinionEstimator
from xrag.pipeline.sl_rag_pipeline import PipelineResult, SLRAGPipeline


# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Complete output of a single experiment run.

    Attributes:
        config: The config that produced this result (for reproducibility).
        metrics: Aggregate metrics per method. Keys are method names
            ('sl_rag', 'softmax', etc.), values are dicts of metric_name → value.
        predictions: Per-example predictions if save_predictions was True.
        timing: Wall-clock seconds per stage.
    """

    config: ExperimentConfig
    metrics: dict[str, dict[str, float]]
    predictions: Optional[list[dict[str, Any]]]
    timing: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "predictions": self.predictions,
            "timing": self.timing,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentResult:
        """Deserialize from a plain dict."""
        return cls(
            config=ExperimentConfig.from_dict(d["config"]),
            metrics=d["metrics"],
            predictions=d.get("predictions"),
            timing=d["timing"],
        )


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def save_result(result: ExperimentResult, filepath: str) -> None:
    """Save experiment result to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=_json_default)


def load_result(filepath: str) -> ExperimentResult:
    """Load experiment result from a JSON file."""
    with open(filepath) as f:
        raw = json.load(f)
    return ExperimentResult.from_dict(raw)


def _json_default(obj: Any) -> Any:
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Factory functions (mockable in tests)
# ---------------------------------------------------------------------------


def _build_loader(config: ExperimentConfig) -> DatasetLoader:
    """Create the dataset loader from config."""
    loaders = {
        "hotpotqa": HotpotQALoader,
        "musique": MuSiQueLoader,
        "nq": NQLoader,
        "popqa": PopQALoader,
    }
    return loaders[config.dataset.name]()


def _build_estimator(config: ExperimentConfig) -> BaseOpinionEstimator:
    """Create the opinion estimator from config."""
    if config.estimation.method == "nli":
        from xrag.opinion_estimation.nli_estimator import NLIRelevanceEstimator
        return NLIRelevanceEstimator(batch_size=config.estimation.batch_size)
    elif config.estimation.method == "llm_judge":
        from xrag.opinion_estimation.llm_judge_estimator import LLMJudgeEstimator
        return LLMJudgeEstimator()
    else:
        raise ValueError(f"Unknown estimation method: {config.estimation.method}")


def _build_generator(config: ExperimentConfig) -> HuggingFaceGenerator:
    """Create the LLM generator from config."""
    return HuggingFaceGenerator(
        model_path=config.generation.model_path,
        device=config.generation.device,
        load_in_4bit=config.generation.load_in_4bit,
        max_new_tokens=config.generation.max_new_tokens,
    )


def _build_pipeline(config: ExperimentConfig) -> SLRAGPipeline:
    """Create the SL-RAG pipeline from config, respecting ablations."""
    from jsonld_ex.confidence_algebra import Opinion
    from xrag.pipeline.conflict_layer import NoOpConflictLayer, SLConflictLayer
    from xrag.pipeline.decision_layer import NoOpDecisionLayer, SLDecisionLayer
    from xrag.pipeline.deduction_layer import NoOpDeductionLayer
    from xrag.pipeline.fusion_layer import NoOpFusionLayer, SLFusionLayer
    from xrag.pipeline.temporal_layer import NoOpTemporalLayer, SLTemporalLayer
    from xrag.pipeline.trust_layer import NoOpTrustLayer, SLTrustLayer

    ablation = config.pipeline.ablation

    # Trust layer
    if ablation == "no_trust":
        trust = NoOpTrustLayer()
    else:
        default_trust = Opinion(
            belief=config.pipeline.default_trust_belief,
            disbelief=0.0,
            uncertainty=1.0 - config.pipeline.default_trust_belief,
            base_rate=0.5,
        )
        trust = SLTrustLayer(default_trust=default_trust)

    # Temporal layer
    if ablation == "no_temporal":
        temporal = NoOpTemporalLayer()
    else:
        temporal = SLTemporalLayer(half_life=config.pipeline.half_life)

    # Conflict layer
    if ablation == "no_conflict":
        conflict = NoOpConflictLayer()
    else:
        conflict = SLConflictLayer(threshold=config.pipeline.conflict_threshold)

    # Fusion layer
    if ablation == "no_fusion":
        fusion = NoOpFusionLayer()
    else:
        fusion = SLFusionLayer(strategy=config.pipeline.fusion_strategy)

    # Deduction layer (always no-op for now — multi-hop is future work)
    deduction = NoOpDeductionLayer()

    # Decision layer
    if ablation == "no_abstention":
        decision = NoOpDecisionLayer()
    else:
        decision = SLDecisionLayer(
            tau_abstain=config.pipeline.tau_abstain,
            tau_conflict=config.pipeline.tau_conflict,
        )

    return SLRAGPipeline(
        trust_layer=trust,
        temporal_layer=temporal,
        conflict_layer=conflict,
        fusion_layer=fusion,
        deduction_layer=deduction,
        decision_layer=decision,
    )


def _build_retriever(
    config: ExperimentConfig, examples: list[QAExample],
) -> Retriever:
    """Create the retriever from config."""
    if config.retrieval.method == "gold":
        return GoldRetriever(examples)
    elif config.retrieval.method == "precomputed":
        return PrecomputedRetriever(config.retrieval.precomputed_path)
    else:
        raise ValueError(f"Unknown retrieval method: {config.retrieval.method}")


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _opinion_to_confidence(opinion) -> float:
    """Project an SL opinion to a scalar confidence in [0, 1].

    Uses the standard projection: P = b + a * u
    where a is the base rate.
    """
    return opinion.belief + opinion.base_rate * opinion.uncertainty


def _build_distractor_pool(
    examples: list[QAExample], max_pool: int = 200,
) -> list[RetrievedPassage]:
    """Build a distractor pool from non-supporting paragraphs across examples."""
    pool: list[RetrievedPassage] = []
    for ex in examples:
        if ex.paragraphs is None:
            continue
        for i, para in enumerate(ex.paragraphs):
            if not para.is_supporting:
                pool.append(RetrievedPassage(
                    id=f"{ex.id}_dist_{i}",
                    title=para.title,
                    text=para.text,
                    score=0.5,
                    has_answer=False,
                ))
            if len(pool) >= max_pool:
                return pool
    return pool


def _simple_contradiction_fn(passage: RetrievedPassage) -> RetrievedPassage:
    """Create a counterfactual version of a passage by negating it."""
    return RetrievedPassage(
        id=f"{passage.id}_contra",
        title=passage.title,
        text=f"Actually, this is false: {passage.text}",
        score=passage.score,
        has_answer=False,
    )


def _apply_corruption(
    retrieval: RetrievalResult,
    config: 'ExperimentConfig',
    distractor_pool: list[RetrievedPassage],
) -> tuple[RetrievalResult, list | None]:
    """Apply corruption to a retrieval result.

    Returns:
        Tuple of (corrupted_retrieval_result, corruption_log_or_None).
    """
    if config.corruption.type is None:
        return retrieval, None

    corr_cfg = CorrCfg(
        corruption_type=_map_corruption_type(config.corruption.type),
        fraction=config.corruption.level,
        target_policy="random",
        seed=config.corruption.seed,
    )

    ctype = config.corruption.type
    if ctype == "distractor":
        result = inject_distractors(retrieval, distractor_pool, corr_cfg)
    elif ctype == "contradiction":
        result = inject_contradictions(retrieval, _simple_contradiction_fn, corr_cfg)
    elif ctype == "missing":
        result = remove_evidence(retrieval, corr_cfg)
    elif ctype == "adversarial":
        result = inject_adversarial(retrieval, distractor_pool, corr_cfg)
    elif ctype == "stale":
        result = backdate_timestamps(retrieval, max_age_hours=720.0, config=corr_cfg)
    else:
        raise ValueError(f"Unknown corruption type: {ctype}")

    return result.retrieval_result, [
        {"index": e.index, "type": e.corruption_type}
        for e in result.corruption_log
    ]


def _map_corruption_type(runner_type: str) -> str:
    """Map runner config corruption types to corruption module types."""
    mapping = {
        "distractor": "distractor",
        "contradiction": "contradiction",
        "missing": "evidence_removal",
        "adversarial": "adversarial",
        "stale": "timestamp_backdate",
    }
    return mapping[runner_type]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single experiment from config to results.

    Args:
        config: Complete experiment specification.

    Returns:
        ExperimentResult with metrics, predictions, and timing.
    """
    timings: dict[str, float] = {}
    t_total_start = time.time()

    # 0. Set seed
    _set_seed(config.seed)

    # 1. Load dataset
    t0 = time.time()
    loader = _build_loader(config)
    examples = loader.load(
        split=config.dataset.split,
        max_samples=config.dataset.max_samples,
    )
    timings["load_data"] = time.time() - t0

    # 2. Build retriever
    retriever = _build_retriever(config, examples)

    # 3. Build estimator and generator
    t0 = time.time()
    estimator = _build_estimator(config)
    timings["load_estimator"] = time.time() - t0

    t0 = time.time()
    generator = _build_generator(config)
    timings["load_generator"] = time.time() - t0

    # 4. Build pipeline
    pipeline = _build_pipeline(config)

    # 4b. Build distractor pool for corruption (if needed)
    distractor_pool: list[RetrievedPassage] = []
    if config.corruption.type in ("distractor", "adversarial"):
        distractor_pool = _build_distractor_pool(examples)

    # 5. Process each example
    t_estimate = 0.0
    t_pipeline = 0.0
    t_generate = 0.0

    per_example: list[dict[str, Any]] = []

    for example in examples:
        # 5a. Retrieve
        retrieval = retriever.retrieve(example.question, top_k=config.retrieval.top_k)
        passages = retrieval.passages

        if not passages:
            per_example.append(_empty_prediction(example))
            continue

        # 5b. Apply corruption (if configured)
        retrieval, corruption_log = _apply_corruption(
            retrieval, config, distractor_pool,
        )
        passages = retrieval.passages

        if not passages:
            per_example.append(_empty_prediction(example))
            continue

        # 5c. Estimate opinions
        t0 = time.time()
        doc_opinions = []
        for passage in passages:
            est = estimator.estimate(example.question, passage.text)
            doc_opinions.append(est.opinion)
        t_estimate += time.time() - t0

        # 5d. Run SL-RAG pipeline
        t0 = time.time()
        pipeline_result = pipeline.run(
            query=example.question,
            doc_opinions=doc_opinions,
        )
        t_pipeline += time.time() - t0

        # 5e. Generate answer
        t0 = time.time()
        should_abstain = pipeline_result.decision != "generate"
        gen_result = generator.generate(
            example.question, passages,
            return_logprobs=True,
            abstain=should_abstain,
        )
        t_generate += time.time() - t0

        # 5f. Compute per-example metrics
        sl_confidence = _opinion_to_confidence(pipeline_result.fused_opinion)
        em = exact_match(gen_result.answer, example.answers)
        f1 = token_f1(gen_result.answer, example.answers)

        # 5g. Compute baseline scores
        baseline_scores = _compute_baselines(
            config=config,
            passages=passages,
            gen_result=gen_result,
        )

        pred = {
            "id": example.id,
            "question": example.question,
            "gold_answers": example.answers,
            "predicted_answer": gen_result.answer,
            "sl_rag_confidence": sl_confidence,
            "sl_rag_decision": pipeline_result.decision,
            "sl_rag_belief": pipeline_result.fused_opinion.belief,
            "sl_rag_disbelief": pipeline_result.fused_opinion.disbelief,
            "sl_rag_uncertainty": pipeline_result.fused_opinion.uncertainty,
            "em": em,
            "f1": f1,
            **{f"{name}_confidence": score
               for name, score in baseline_scores.items()},
        }
        per_example.append(pred)

    timings["estimate_opinions"] = t_estimate
    timings["run_pipeline"] = t_pipeline
    timings["generate"] = t_generate

    # 6. Aggregate metrics
    t0 = time.time()
    metrics = _aggregate_metrics(config, per_example)
    timings["evaluate"] = time.time() - t0
    timings["total"] = time.time() - t_total_start

    # 7. Build result
    return ExperimentResult(
        config=config,
        metrics=metrics,
        predictions=per_example if config.output.save_predictions else None,
        timing=timings,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_prediction(example: QAExample) -> dict[str, Any]:
    """Create a prediction dict for an example with no passages."""
    return {
        "id": example.id,
        "question": example.question,
        "gold_answers": example.answers,
        "predicted_answer": "",
        "sl_rag_confidence": 0.0,
        "sl_rag_decision": "abstain",
        "sl_rag_belief": 0.0,
        "sl_rag_disbelief": 0.0,
        "sl_rag_uncertainty": 1.0,
        "em": 0.0,
        "f1": 0.0,
    }


def _compute_baselines(
    config: ExperimentConfig,
    passages: list[RetrievedPassage],
    gen_result: GenerationResult,
) -> dict[str, float]:
    """Compute enabled baseline confidence scores."""
    scores: dict[str, float] = {}

    if config.baselines.run_softmax and gen_result.token_logprobs:
        from xrag.baselines.softmax_confidence import SoftmaxConfidenceScorer
        scorer = SoftmaxConfidenceScorer(method="normalized_seq_prob")
        uq = scorer.score(token_logprobs=gen_result.token_logprobs)
        scores["softmax"] = uq.confidence

    if config.baselines.run_retrieval_conf:
        from xrag.baselines.retrieval_confidence import RetrievalConfidenceScorer
        scorer = RetrievalConfidenceScorer(method="mean_retrieval_score")
        retrieval_scores = [p.score for p in passages]
        uq = scorer.score(retrieval_scores=retrieval_scores)
        scores["retrieval_conf"] = uq.confidence

    if config.baselines.run_combined_heuristic and gen_result.token_logprobs:
        from xrag.baselines.combined_heuristic import CombinedHeuristicScorer
        from xrag.baselines.softmax_confidence import SoftmaxConfidenceScorer as _SC
        from xrag.baselines.retrieval_confidence import RetrievalConfidenceScorer as _RC
        gen_conf = _SC(method="normalized_seq_prob").score(
            token_logprobs=gen_result.token_logprobs,
        ).confidence
        ret_conf = _RC(method="max_retrieval_score").score(
            retrieval_scores=[p.score for p in passages],
        ).confidence
        scorer = CombinedHeuristicScorer(strategy="product")
        uq = scorer.score(
            retrieval_confidence=ret_conf,
            generation_confidence=gen_conf,
        )
        scores["combined_heuristic"] = uq.confidence

    # Note: semantic_entropy, p_true, conformal require special handling
    # (multiple generations, extra LLM calls, calibration set)
    # and are wired separately in the full experiment harness.

    return scores


def _aggregate_metrics(
    config: ExperimentConfig,
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute aggregate metrics for SL-RAG and each baseline."""
    metrics: dict[str, dict[str, float]] = {}

    if not predictions:
        return metrics

    # Ground truth
    em_scores = [p["em"] for p in predictions]
    f1_scores = [p["f1"] for p in predictions]
    correctness = np.array(em_scores, dtype=float)

    # SL-RAG metrics
    sl_confidences = np.array([p["sl_rag_confidence"] for p in predictions])
    metrics["sl_rag"] = _compute_method_metrics(
        correctness, sl_confidences, em_scores, f1_scores,
    )

    # Baseline metrics
    baseline_keys = []
    if config.baselines.run_softmax:
        baseline_keys.append("softmax")
    if config.baselines.run_retrieval_conf:
        baseline_keys.append("retrieval_conf")
    if config.baselines.run_combined_heuristic:
        baseline_keys.append("combined_heuristic")

    for key in baseline_keys:
        conf_key = f"{key}_confidence"
        if conf_key not in predictions[0]:
            continue
        confs = np.array([p.get(conf_key, 0.0) for p in predictions])
        metrics[key] = _compute_method_metrics(
            correctness, confs, em_scores, f1_scores,
        )

    return metrics


def _compute_method_metrics(
    correctness: np.ndarray,
    confidences: np.ndarray,
    em_scores: list[float],
    f1_scores: list[float],
) -> dict[str, float]:
    """Compute the standard metric suite for one method."""
    n = len(correctness)
    result: dict[str, float] = {
        "em": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "n_examples": n,
    }

    # Calibration metrics (need sufficient examples)
    if n >= 10:
        try:
            result["ece"] = float(expected_calibration_error(
                confidences, correctness, n_bins=10,
            ))
        except Exception:
            result["ece"] = float("nan")

        try:
            result["brier"] = float(brier_score(confidences, correctness))
        except Exception:
            result["brier"] = float("nan")

    # Selective prediction metrics
    if n >= 5 and len(np.unique(correctness)) > 1:
        try:
            result["auroc"] = float(auroc_selective(confidences, correctness))
        except Exception:
            result["auroc"] = float("nan")

        try:
            result["auprc"] = float(auprc_selective(confidences, correctness))
        except Exception:
            result["auprc"] = float("nan")

        try:
            result["e_aurc"] = float(e_aurc(confidences, correctness))
        except Exception:
            result["e_aurc"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: python -m xrag.experiment.runner --config path.yaml"""
    import argparse

    parser = argparse.ArgumentParser(description="Run an SL-RAG experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default=None, help="Override output filepath")
    args = parser.parse_args()

    from xrag.experiment.config import load_config

    config = load_config(args.config)
    print(f"Running experiment: {config.name}")
    print(f"  Dataset: {config.dataset.name} ({config.dataset.split})")
    print(f"  Model: {config.generation.model_path}")
    print(f"  Seed: {config.seed}")

    result = run_experiment(config)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        out_dir = Path(config.output.results_dir)
        out_path = str(out_dir / f"{config.name}.json")

    save_result(result, out_path)
    print(f"\nResults saved to {out_path}")
    print(f"  Total time: {result.timing['total']:.1f}s")
    for method, m in result.metrics.items():
        print(f"  {method}: EM={m.get('em', 0):.3f} F1={m.get('f1', 0):.3f}")


if __name__ == "__main__":
    main()
