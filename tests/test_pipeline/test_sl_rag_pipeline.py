"""TDD tests for the SL-RAG pipeline skeleton.

Tests the pipeline orchestrator and all layer interfaces:
- PipelineResult / PipelineContext dataclasses
- Layer ABCs + no-op implementations for each of the 6 pipeline stages
- SLRAGPipeline wiring and end-to-end with no-ops

Red phase: all tests should FAIL until implementation is written.
"""

from __future__ import annotations

import abc
from dataclasses import fields
from typing import Optional
from unittest.mock import MagicMock

import pytest
from jsonld_ex.confidence_algebra import Opinion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_opinions(n: int = 3) -> list[Opinion]:
    """Create n sample opinions for testing."""
    return [
        Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5),
        Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3, base_rate=0.5),
    ][:n]


# ---------------------------------------------------------------------------
# Section 1: PipelineResult dataclass
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """Tests for PipelineResult — the final output of the pipeline."""

    def test_import(self):
        from xrag.pipeline.sl_rag_pipeline import PipelineResult

        assert PipelineResult is not None

    def test_field_names(self):
        from xrag.pipeline.sl_rag_pipeline import PipelineResult

        names = {f.name for f in fields(PipelineResult)}
        expected = {
            "query", "answer", "decision",
            "fused_opinion", "doc_opinions",
            "conflict_detected", "metadata",
        }
        assert names == expected

    def test_construction_generate(self):
        from xrag.pipeline.sl_rag_pipeline import PipelineResult

        fused = Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5)
        result = PipelineResult(
            query="What is X?",
            answer="X is Y.",
            decision="generate",
            fused_opinion=fused,
            doc_opinions=_sample_opinions(2),
            conflict_detected=False,
            metadata={},
        )
        assert result.answer == "X is Y."
        assert result.decision == "generate"

    def test_construction_abstain(self):
        from xrag.pipeline.sl_rag_pipeline import PipelineResult

        result = PipelineResult(
            query="What is Z?",
            answer=None,
            decision="abstain",
            fused_opinion=Opinion(belief=0.1, disbelief=0.1, uncertainty=0.8, base_rate=0.5),
            doc_opinions=[],
            conflict_detected=False,
            metadata={},
        )
        assert result.answer is None
        assert result.decision == "abstain"

    def test_construction_flag_conflict(self):
        from xrag.pipeline.sl_rag_pipeline import PipelineResult

        result = PipelineResult(
            query="Q",
            answer=None,
            decision="flag_conflict",
            fused_opinion=None,
            doc_opinions=_sample_opinions(2),
            conflict_detected=True,
            metadata={},
        )
        assert result.conflict_detected is True
        assert result.decision == "flag_conflict"


# ---------------------------------------------------------------------------
# Section 2: Trust Layer
# ---------------------------------------------------------------------------


class TestTrustLayer:
    def test_import_abc(self):
        from xrag.pipeline.trust_layer import TrustLayer

        assert TrustLayer is not None

    def test_is_abstract(self):
        from xrag.pipeline.trust_layer import TrustLayer

        assert abc.ABC in TrustLayer.__mro__

    def test_cannot_instantiate(self):
        from xrag.pipeline.trust_layer import TrustLayer

        with pytest.raises(TypeError):
            TrustLayer()

    def test_has_apply_method(self):
        from xrag.pipeline.trust_layer import TrustLayer

        assert hasattr(TrustLayer, "apply")
        assert getattr(TrustLayer.apply, "__isabstractmethod__", False)


class TestNoOpTrustLayer:
    def test_import(self):
        from xrag.pipeline.trust_layer import NoOpTrustLayer

        assert NoOpTrustLayer is not None

    def test_is_trust_layer(self):
        from xrag.pipeline.trust_layer import NoOpTrustLayer, TrustLayer

        assert issubclass(NoOpTrustLayer, TrustLayer)

    def test_passthrough(self):
        from xrag.pipeline.trust_layer import NoOpTrustLayer

        layer = NoOpTrustLayer()
        opinions = _sample_opinions(3)
        result = layer.apply(opinions)
        assert len(result) == 3
        # No-op: opinions should be identical
        for orig, out in zip(opinions, result):
            assert orig.belief == out.belief
            assert orig.disbelief == out.disbelief
            assert orig.uncertainty == out.uncertainty

    def test_empty_input(self):
        from xrag.pipeline.trust_layer import NoOpTrustLayer

        layer = NoOpTrustLayer()
        assert layer.apply([]) == []


# ---------------------------------------------------------------------------
# Section 3: Temporal Layer
# ---------------------------------------------------------------------------


class TestTemporalLayer:
    def test_import_abc(self):
        from xrag.pipeline.temporal_layer import TemporalLayer

        assert TemporalLayer is not None

    def test_is_abstract(self):
        from xrag.pipeline.temporal_layer import TemporalLayer

        assert abc.ABC in TemporalLayer.__mro__

    def test_has_apply_method(self):
        from xrag.pipeline.temporal_layer import TemporalLayer

        assert getattr(TemporalLayer.apply, "__isabstractmethod__", False)


class TestNoOpTemporalLayer:
    def test_import(self):
        from xrag.pipeline.temporal_layer import NoOpTemporalLayer

        assert NoOpTemporalLayer is not None

    def test_is_temporal_layer(self):
        from xrag.pipeline.temporal_layer import NoOpTemporalLayer, TemporalLayer

        assert issubclass(NoOpTemporalLayer, TemporalLayer)

    def test_passthrough(self):
        from xrag.pipeline.temporal_layer import NoOpTemporalLayer

        layer = NoOpTemporalLayer()
        opinions = _sample_opinions(2)
        result = layer.apply(opinions)
        assert len(result) == 2
        for orig, out in zip(opinions, result):
            assert orig.belief == out.belief


# ---------------------------------------------------------------------------
# Section 4: Conflict Layer
# ---------------------------------------------------------------------------


class TestConflictResult:
    def test_import(self):
        from xrag.pipeline.conflict_layer import ConflictResult

        assert ConflictResult is not None

    def test_field_names(self):
        from xrag.pipeline.conflict_layer import ConflictResult

        names = {f.name for f in fields(ConflictResult)}
        assert names == {"conflict_detected", "conflict_pairs", "max_conflict_score"}

    def test_no_conflict(self):
        from xrag.pipeline.conflict_layer import ConflictResult

        result = ConflictResult(
            conflict_detected=False,
            conflict_pairs=[],
            max_conflict_score=0.0,
        )
        assert result.conflict_detected is False


class TestConflictLayer:
    def test_import_abc(self):
        from xrag.pipeline.conflict_layer import ConflictLayer

        assert ConflictLayer is not None

    def test_is_abstract(self):
        from xrag.pipeline.conflict_layer import ConflictLayer

        assert abc.ABC in ConflictLayer.__mro__

    def test_has_detect_method(self):
        from xrag.pipeline.conflict_layer import ConflictLayer

        assert getattr(ConflictLayer.detect, "__isabstractmethod__", False)


class TestNoOpConflictLayer:
    def test_import(self):
        from xrag.pipeline.conflict_layer import NoOpConflictLayer

        assert NoOpConflictLayer is not None

    def test_is_conflict_layer(self):
        from xrag.pipeline.conflict_layer import ConflictLayer, NoOpConflictLayer

        assert issubclass(NoOpConflictLayer, ConflictLayer)

    def test_returns_no_conflict(self):
        from xrag.pipeline.conflict_layer import NoOpConflictLayer

        layer = NoOpConflictLayer()
        result = layer.detect(_sample_opinions(3))
        assert result.conflict_detected is False
        assert result.conflict_pairs == []
        assert result.max_conflict_score == 0.0


# ---------------------------------------------------------------------------
# Section 5: Fusion Layer
# ---------------------------------------------------------------------------


class TestFusionLayer:
    def test_import_abc(self):
        from xrag.pipeline.fusion_layer import FusionLayer

        assert FusionLayer is not None

    def test_is_abstract(self):
        from xrag.pipeline.fusion_layer import FusionLayer

        assert abc.ABC in FusionLayer.__mro__

    def test_has_fuse_method(self):
        from xrag.pipeline.fusion_layer import FusionLayer

        assert getattr(FusionLayer.fuse, "__isabstractmethod__", False)


class TestNoOpFusionLayer:
    def test_import(self):
        from xrag.pipeline.fusion_layer import NoOpFusionLayer

        assert NoOpFusionLayer is not None

    def test_is_fusion_layer(self):
        from xrag.pipeline.fusion_layer import FusionLayer, NoOpFusionLayer

        assert issubclass(NoOpFusionLayer, FusionLayer)

    def test_returns_vacuous_opinion(self):
        """No-op fusion returns a vacuous (max uncertainty) opinion."""
        from xrag.pipeline.fusion_layer import NoOpFusionLayer

        layer = NoOpFusionLayer()
        result = layer.fuse(_sample_opinions(3))
        assert isinstance(result, Opinion)
        assert result.uncertainty == pytest.approx(1.0)
        assert result.belief == pytest.approx(0.0)
        assert result.disbelief == pytest.approx(0.0)

    def test_empty_input_returns_vacuous(self):
        from xrag.pipeline.fusion_layer import NoOpFusionLayer

        layer = NoOpFusionLayer()
        result = layer.fuse([])
        assert result.uncertainty == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Section 6: Deduction Layer
# ---------------------------------------------------------------------------


class TestDeductionLayer:
    def test_import_abc(self):
        from xrag.pipeline.deduction_layer import DeductionLayer

        assert DeductionLayer is not None

    def test_is_abstract(self):
        from xrag.pipeline.deduction_layer import DeductionLayer

        assert abc.ABC in DeductionLayer.__mro__

    def test_has_deduce_method(self):
        from xrag.pipeline.deduction_layer import DeductionLayer

        assert getattr(DeductionLayer.deduce, "__isabstractmethod__", False)


class TestNoOpDeductionLayer:
    def test_import(self):
        from xrag.pipeline.deduction_layer import NoOpDeductionLayer

        assert NoOpDeductionLayer is not None

    def test_is_deduction_layer(self):
        from xrag.pipeline.deduction_layer import DeductionLayer, NoOpDeductionLayer

        assert issubclass(NoOpDeductionLayer, DeductionLayer)

    def test_passthrough(self):
        """No-op deduction returns the input opinion unchanged."""
        from xrag.pipeline.deduction_layer import NoOpDeductionLayer

        layer = NoOpDeductionLayer()
        opinion = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2, base_rate=0.5)
        result = layer.deduce(opinion, sub_questions=None)
        assert result.belief == opinion.belief
        assert result.disbelief == opinion.disbelief
        assert result.uncertainty == opinion.uncertainty


# ---------------------------------------------------------------------------
# Section 7: Decision Layer
# ---------------------------------------------------------------------------


class TestDecisionResult:
    def test_import(self):
        from xrag.pipeline.decision_layer import DecisionResult

        assert DecisionResult is not None

    def test_field_names(self):
        from xrag.pipeline.decision_layer import DecisionResult

        names = {f.name for f in fields(DecisionResult)}
        assert names == {"decision", "reason"}

    def test_generate_decision(self):
        from xrag.pipeline.decision_layer import DecisionResult

        result = DecisionResult(decision="generate", reason="sufficient evidence")
        assert result.decision == "generate"


class TestDecisionLayer:
    def test_import_abc(self):
        from xrag.pipeline.decision_layer import DecisionLayer

        assert DecisionLayer is not None

    def test_is_abstract(self):
        from xrag.pipeline.decision_layer import DecisionLayer

        assert abc.ABC in DecisionLayer.__mro__

    def test_has_decide_method(self):
        from xrag.pipeline.decision_layer import DecisionLayer

        assert getattr(DecisionLayer.decide, "__isabstractmethod__", False)


class TestNoOpDecisionLayer:
    def test_import(self):
        from xrag.pipeline.decision_layer import NoOpDecisionLayer

        assert NoOpDecisionLayer is not None

    def test_is_decision_layer(self):
        from xrag.pipeline.decision_layer import DecisionLayer, NoOpDecisionLayer

        assert issubclass(NoOpDecisionLayer, DecisionLayer)

    def test_always_generates(self):
        """No-op decision layer always decides to generate."""
        from xrag.pipeline.conflict_layer import ConflictResult
        from xrag.pipeline.decision_layer import NoOpDecisionLayer

        layer = NoOpDecisionLayer()
        opinion = Opinion(belief=0.1, disbelief=0.1, uncertainty=0.8, base_rate=0.5)
        conflict = ConflictResult(
            conflict_detected=False, conflict_pairs=[], max_conflict_score=0.0
        )
        result = layer.decide(opinion, conflict)
        assert result.decision == "generate"


# ---------------------------------------------------------------------------
# Section 8: SLRAGPipeline orchestrator
# ---------------------------------------------------------------------------


class TestSLRAGPipelineBasics:
    """Tests for the SLRAGPipeline class."""

    def test_import(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        assert SLRAGPipeline is not None

    def test_instantiate_with_all_noop(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        assert pipeline is not None

    def test_has_run_method(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        assert hasattr(SLRAGPipeline, "run")

    def test_has_run_batch_method(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        assert hasattr(SLRAGPipeline, "run_batch")


class TestSLRAGPipelineConstruction:
    """Tests for constructing SLRAGPipeline with explicit layers."""

    def test_accepts_all_layer_args(self):
        from xrag.pipeline.conflict_layer import NoOpConflictLayer
        from xrag.pipeline.decision_layer import NoOpDecisionLayer
        from xrag.pipeline.deduction_layer import NoOpDeductionLayer
        from xrag.pipeline.fusion_layer import NoOpFusionLayer
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline
        from xrag.pipeline.temporal_layer import NoOpTemporalLayer
        from xrag.pipeline.trust_layer import NoOpTrustLayer

        pipeline = SLRAGPipeline(
            trust_layer=NoOpTrustLayer(),
            temporal_layer=NoOpTemporalLayer(),
            conflict_layer=NoOpConflictLayer(),
            fusion_layer=NoOpFusionLayer(),
            deduction_layer=NoOpDeductionLayer(),
            decision_layer=NoOpDecisionLayer(),
        )
        assert pipeline is not None


# ---------------------------------------------------------------------------
# Section 9: End-to-end pipeline run with no-ops
# ---------------------------------------------------------------------------


class TestSLRAGPipelineRun:
    """Tests for running the pipeline end-to-end with no-op layers."""

    def test_run_returns_pipeline_result(self):
        from xrag.pipeline.sl_rag_pipeline import PipelineResult, SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        doc_opinions = _sample_opinions(3)
        result = pipeline.run(query="What is X?", doc_opinions=doc_opinions)
        assert isinstance(result, PipelineResult)

    def test_run_query_preserved(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        result = pipeline.run(query="What is X?", doc_opinions=_sample_opinions(2))
        assert result.query == "What is X?"

    def test_run_noop_always_generates(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        result = pipeline.run(query="Q?", doc_opinions=_sample_opinions(2))
        assert result.decision == "generate"

    def test_run_noop_no_conflict(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        result = pipeline.run(query="Q?", doc_opinions=_sample_opinions(2))
        assert result.conflict_detected is False

    def test_run_doc_opinions_preserved(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        opinions = _sample_opinions(3)
        result = pipeline.run(query="Q?", doc_opinions=opinions)
        # After no-op trust + temporal, doc opinions should be unchanged
        assert len(result.doc_opinions) == 3

    def test_run_fused_opinion_is_vacuous(self):
        """No-op fusion returns vacuous opinion."""
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        result = pipeline.run(query="Q?", doc_opinions=_sample_opinions(2))
        assert result.fused_opinion is not None
        assert result.fused_opinion.uncertainty == pytest.approx(1.0)

    def test_run_answer_is_none_for_noop(self):
        """No-op pipeline has no generator — answer should be None."""
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        result = pipeline.run(query="Q?", doc_opinions=_sample_opinions(2))
        assert result.answer is None

    def test_run_empty_opinions(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        result = pipeline.run(query="Q?", doc_opinions=[])
        assert result.decision == "generate"
        assert result.fused_opinion.uncertainty == pytest.approx(1.0)

    def test_run_batch(self):
        from xrag.pipeline.sl_rag_pipeline import PipelineResult, SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        queries = ["Q1?", "Q2?"]
        opinions_list = [_sample_opinions(2), _sample_opinions(3)]
        results = pipeline.run_batch(queries=queries, doc_opinions_list=opinions_list)
        assert len(results) == 2
        assert all(isinstance(r, PipelineResult) for r in results)
        assert results[0].query == "Q1?"
        assert results[1].query == "Q2?"

    def test_run_batch_mismatched_raises(self):
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        with pytest.raises(ValueError):
            pipeline.run_batch(
                queries=["Q1?"],
                doc_opinions_list=[_sample_opinions(2), _sample_opinions(3)],
            )

    def test_run_with_sub_questions(self):
        """Pipeline should accept optional sub_questions for multi-hop."""
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        result = pipeline.run(
            query="Multi-hop Q?",
            doc_opinions=_sample_opinions(2),
            sub_questions=["Sub Q1?", "Sub Q2?"],
        )
        assert isinstance(result.fused_opinion, Opinion)

    def test_metadata_contains_layer_outputs(self):
        """Metadata should record intermediate layer outputs for debugging."""
        from xrag.pipeline.sl_rag_pipeline import SLRAGPipeline

        pipeline = SLRAGPipeline.with_noop_layers()
        result = pipeline.run(query="Q?", doc_opinions=_sample_opinions(2))
        assert isinstance(result.metadata, dict)
        # Should have keys for each layer's output
        assert "trust_opinions" in result.metadata
        assert "temporal_opinions" in result.metadata
        assert "conflict_result" in result.metadata
        assert "decision_result" in result.metadata
