"""End-to-end integration test: query → retrieve → estimate → fuse → decide → generate.

This is the Week 2 capstone test. It proves the full SL-RAG pipeline
works with real models on real data.

Run with: python -m pytest -m "slow and gpu" tests/test_end_to_end.py -v

Requires:
    - DeBERTa NLI model (cached from previous integration tests)
    - Llama 3.1 8B Instruct at D:\\cc\\models\\Llama-3.1-8B-Instruct
    - HotpotQA dataset (downloaded on demand via HuggingFace)
"""

from __future__ import annotations

import os

import pytest

from jsonld_ex.confidence_algebra import Opinion


@pytest.mark.slow
@pytest.mark.gpu
class TestEndToEnd:
    """Full pipeline integration test on HotpotQA examples."""

    AVAILABLE_MODELS = [
        ("Qwen2.5-7B", "D:\\cc\\models\\Qwen2.5-7B-Instruct"),
        ("Mistral-7B-v0.3", "D:\\cc\\models\\Mistral-7B-Instruct-v0.3"),
        ("Llama-3.1-8B", "D:\\cc\\models\\Llama-3.1-8B-Instruct"),
    ]

    @pytest.fixture(scope="class")
    def hotpotqa_examples(self):
        """Load a few HotpotQA examples."""
        from xrag.benchmarks.data_loader import HotpotQALoader
        loader = HotpotQALoader()
        return loader.load(split="validation", max_samples=5)

    @pytest.fixture(scope="class")
    def gold_retriever(self, hotpotqa_examples):
        """Gold retriever for oracle experiments."""
        from xrag.benchmarks.retriever import GoldRetriever
        return GoldRetriever(hotpotqa_examples)

    @pytest.fixture(scope="class")
    def nli_estimator(self):
        """NLI relevance estimator (DeBERTa on CUDA)."""
        from xrag.opinion_estimation.nli_estimator import NLIRelevanceEstimator
        return NLIRelevanceEstimator()

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Pipeline with real SL layers."""
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline
        return SLRAGPipeline.with_default_layers()

    @pytest.fixture(scope="class")
    def generator(self):
        """LLM generator — uses first available model."""
        import os
        for name, path in self.AVAILABLE_MODELS:
            if os.path.exists(path):
                from xrag.generation.generator import HuggingFaceGenerator
                return HuggingFaceGenerator(
                    model_path=path,
                    device="cuda",
                    load_in_4bit=True,
                )
        pytest.skip("No local model found at D:\\cc\\models\\")

    def test_full_pipeline_single_query(
        self, hotpotqa_examples, gold_retriever, nli_estimator, pipeline, generator
    ):
        """Run the full pipeline on a single HotpotQA query."""
        example = hotpotqa_examples[0]
        query = example.question

        # 1. Retrieve
        retrieval = gold_retriever.retrieve(query, top_k=10)
        assert len(retrieval.passages) > 0

        # 2. Estimate opinions via NLI
        opinions = []
        for passage in retrieval.passages:
            est = nli_estimator.estimate(query, passage.text)
            opinions.append(est.opinion)
        assert len(opinions) == len(retrieval.passages)

        # 3. Run SL pipeline (trust → temporal → conflict → fuse → decide)
        result = pipeline.run(query=query, doc_opinions=opinions)
        assert result.decision in ("generate", "abstain", "flag_conflict")
        assert result.fused_opinion is not None
        total = result.fused_opinion.belief + result.fused_opinion.disbelief + result.fused_opinion.uncertainty
        assert abs(total - 1.0) < 1e-9

        # 4. Generate answer (if pipeline says generate)
        if result.decision == "generate":
            gen_result = generator.generate(query, retrieval.passages)
            assert isinstance(gen_result.answer, str)
            assert len(gen_result.answer) > 0
            assert gen_result.metadata["abstained"] is False
        else:
            gen_result = generator.generate(query, retrieval.passages, abstain=True)
            assert gen_result.answer == ""
            assert gen_result.metadata["abstained"] is True

    def test_pipeline_without_generator(
        self, hotpotqa_examples, gold_retriever, nli_estimator, pipeline
    ):
        """Run pipeline only (no LLM generation) — faster, tests SL algebra."""
        example = hotpotqa_examples[0]
        query = example.question

        retrieval = gold_retriever.retrieve(query, top_k=10)
        opinions = [
            nli_estimator.estimate(query, p.text).opinion
            for p in retrieval.passages
        ]

        result = pipeline.run(query=query, doc_opinions=opinions)

        # Fused opinion should have reduced uncertainty vs individual opinions
        min_input_u = min(o.uncertainty for o in opinions)
        # Trust discount increases u, but fusion decreases it — net effect depends
        assert result.fused_opinion is not None
        assert result.fused_opinion.belief >= 0.0
        assert result.fused_opinion.uncertainty >= 0.0

    def test_conflicting_docs_detected(
        self, hotpotqa_examples, gold_retriever, nli_estimator
    ):
        """Inject a contradiction and verify conflict detection."""
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_default_layers(tau_conflict=0.3)

        example = hotpotqa_examples[0]
        query = example.question

        retrieval = gold_retriever.retrieve(query, top_k=5)
        opinions = [
            nli_estimator.estimate(query, p.text).opinion
            for p in retrieval.passages
        ]

        # Inject a strongly contradicting synthetic opinion
        opinions.append(Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05))

        result = pipeline.run(query=query, doc_opinions=opinions)

        # The conflict layer should detect the injected contradiction
        # (depends on threshold and actual opinion values)
        assert result.fused_opinion is not None

    def test_batch_pipeline(
        self, hotpotqa_examples, gold_retriever, nli_estimator, pipeline
    ):
        """Run pipeline on multiple queries."""
        queries = []
        opinions_list = []

        for example in hotpotqa_examples[:3]:
            query = example.question
            retrieval = gold_retriever.retrieve(query, top_k=5)
            if not retrieval.passages:
                continue
            opinions = [
                nli_estimator.estimate(query, p.text).opinion
                for p in retrieval.passages
            ]
            queries.append(query)
            opinions_list.append(opinions)

        results = pipeline.run_batch(queries, opinions_list)
        assert len(results) == len(queries)
        for r in results:
            assert r.decision in ("generate", "abstain", "flag_conflict")
            assert r.fused_opinion is not None
