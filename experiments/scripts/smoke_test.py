"""Smoke test — run 3 examples end-to-end to verify the full pipeline."""

from xrag.experiment.config import (
    ExperimentConfig, DatasetConfig, RetrievalConfig, CorruptionConfig,
    EstimationConfig, PipelineConfig, BaselinesConfig, GenerationConfig,
    OutputConfig,
)
from xrag.experiment.runner import run_experiment, save_result

config = ExperimentConfig(
    name="smoke_test",
    seed=42,
    dataset=DatasetConfig(name="hotpotqa", split="validation", max_samples=20),
    retrieval=RetrievalConfig(method="gold", top_k=10),
    corruption=CorruptionConfig(type=None, level=0.0),
    estimation=EstimationConfig(method="nli", batch_size=32),
    pipeline=PipelineConfig(),
    baselines=BaselinesConfig(
        run_softmax=True,
        run_retrieval_conf=True,
        run_semantic_entropy=False,
        run_p_true=False,
        run_combined_heuristic=True,
        run_conformal=False,
    ),
    generation=GenerationConfig(
        model_path=r"E:\data\code\claudecode\models\Qwen2.5-7B-Instruct",
        load_in_4bit=True,
        max_new_tokens=64,
    ),
    output=OutputConfig(results_dir="experiments/results", save_predictions=True),
)

print("Running smoke test (3 examples, Qwen 7B 4-bit)...")
result = run_experiment(config)
save_result(result, "experiments/results/smoke_test.json")

print(f"Total time: {result.timing['total']:.1f}s")
for method, m in result.metrics.items():
    print(f"  {method}: EM={m.get('em', 0):.3f} F1={m.get('f1', 0):.3f}")

print(f"Predictions: {len(result.predictions)} examples")
for p in result.predictions:
    print(f"  Q: {p['question'][:60]}...")
    print(f"  A: {p['predicted_answer'][:60]}")
    print(f"  Gold: {p['gold_answers']}")
    print(f"  EM={p['em']} F1={p['f1']:.2f} SL_conf={p['sl_rag_confidence']:.3f} decision={p['sl_rag_decision']}")
    print()
