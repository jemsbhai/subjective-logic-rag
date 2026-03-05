"""Tests for NLI-based opinion estimators (TDD — written before implementation).

Architecture under test:
    BaseOpinionEstimator (ABC)
        └── NLIEstimator (shared model loading, inference, doc-doc conflict)
                ├── NLIRelevanceEstimator    (Stage 1: no disbelief)
                └── NLIFaithfulnessEstimator (Stage 2: full 3-class)

Stage 1 (Relevance): NLI(document, query)
    b = entailment × (1 − base_u)
    d = 0
    u = base_u + (neutral + contradiction) × (1 − base_u)

Stage 2 (Faithfulness): NLI(document, generated_answer)
    b = entailment × (1 − base_u)
    d = contradiction × (1 − base_u)
    u = base_u + neutral × (1 − base_u)

Doc-doc conflict: NLI(doc_a, doc_b) — uses faithfulness mapping
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jsonld_ex.confidence_algebra import Opinion

from xrag.opinion_estimation.base import BaseOpinionEstimator, EstimationResult
from xrag.opinion_estimation.nli_estimator import (
    NLIEstimator,
    NLIFaithfulnessEstimator,
    NLIRelevanceEstimator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_opinion(op: Opinion, tol: float = 1e-9) -> bool:
    """Check that b + d + u == 1 and all components are non-negative."""
    return (
        op.belief >= -tol
        and op.disbelief >= -tol
        and op.uncertainty >= -tol
        and abs(op.belief + op.disbelief + op.uncertainty - 1.0) < tol
    )


def _make_estimator(cls, evidence_weight: float = 10.0):
    """Create an estimator instance with mocked model (no download)."""
    est = cls.__new__(cls)
    est.evidence_weight = evidence_weight
    est.model_name = "mock-model"
    est.device = "cpu"
    est.batch_size = 32
    est._model = MagicMock()
    est._tokenizer = MagicMock()
    return est


# ===========================================================================
# Class hierarchy and interface tests
# ===========================================================================

class TestClassHierarchy:
    """Verify the class hierarchy is correctly structured."""

    def test_nli_estimator_is_subclass_of_base(self):
        assert issubclass(NLIEstimator, BaseOpinionEstimator)

    def test_relevance_is_subclass_of_nli(self):
        assert issubclass(NLIRelevanceEstimator, NLIEstimator)

    def test_faithfulness_is_subclass_of_nli(self):
        assert issubclass(NLIFaithfulnessEstimator, NLIEstimator)

    def test_relevance_is_subclass_of_base(self):
        assert issubclass(NLIRelevanceEstimator, BaseOpinionEstimator)

    def test_faithfulness_is_subclass_of_base(self):
        assert issubclass(NLIFaithfulnessEstimator, BaseOpinionEstimator)

    def test_nli_estimator_not_directly_instantiable(self):
        """NLIEstimator should not implement estimate/estimate_batch."""
        with pytest.raises(TypeError):
            NLIEstimator(model_name="mock", _lazy_load=True)

    def test_relevance_has_estimate(self):
        assert callable(getattr(NLIRelevanceEstimator, "estimate", None))

    def test_relevance_has_estimate_batch(self):
        assert callable(getattr(NLIRelevanceEstimator, "estimate_batch", None))

    def test_faithfulness_has_estimate(self):
        assert callable(getattr(NLIFaithfulnessEstimator, "estimate", None))

    def test_faithfulness_has_estimate_batch(self):
        assert callable(getattr(NLIFaithfulnessEstimator, "estimate_batch", None))


# ===========================================================================
# Constructor validation (shared by both subclasses)
# ===========================================================================

class TestConstructorValidation:
    """Test constructor validation for both estimator types."""

    def test_relevance_rejects_zero_evidence_weight(self):
        with pytest.raises(ValueError, match="evidence_weight"):
            NLIRelevanceEstimator(model_name="mock", evidence_weight=0.0, _lazy_load=True)

    def test_relevance_rejects_negative_evidence_weight(self):
        with pytest.raises(ValueError, match="evidence_weight"):
            NLIRelevanceEstimator(model_name="mock", evidence_weight=-1.0, _lazy_load=True)

    def test_faithfulness_rejects_zero_evidence_weight(self):
        with pytest.raises(ValueError, match="evidence_weight"):
            NLIFaithfulnessEstimator(model_name="mock", evidence_weight=0.0, _lazy_load=True)

    def test_relevance_accepts_valid_weight(self):
        est = NLIRelevanceEstimator(model_name="mock", evidence_weight=10.0, _lazy_load=True)
        assert est.evidence_weight == 10.0

    def test_faithfulness_accepts_valid_weight(self):
        est = NLIFaithfulnessEstimator(model_name="mock", evidence_weight=10.0, _lazy_load=True)
        assert est.evidence_weight == 10.0

    def test_default_evidence_weight(self):
        est = NLIRelevanceEstimator(model_name="mock", _lazy_load=True)
        assert est.evidence_weight > 0


# ===========================================================================
# Relevance mapping formula (Stage 1: no disbelief)
# ===========================================================================

class TestRelevanceMapping:
    """Test the relevance NLI → SL opinion mapping.

    Formula:
        base_u = 1 / (W + 1)
        b = entailment * (1 - base_u)
        d = 0
        u = base_u + (neutral + contradiction) * (1 - base_u)
    """

    def test_pure_entailment(self):
        """NLI = (1, 0, 0) → high belief, d=0, u=base_u."""
        est = _make_estimator(NLIRelevanceEstimator)
        base_u = 1.0 / 11.0

        op = est._nli_scores_to_opinion(1.0, 0.0, 0.0)

        assert abs(op.belief - (1.0 * (1.0 - base_u))) < 1e-9
        assert abs(op.disbelief - 0.0) < 1e-9
        assert abs(op.uncertainty - base_u) < 1e-9
        assert _valid_opinion(op)

    def test_pure_contradiction(self):
        """NLI = (0, 0, 1) → d=0, all goes to uncertainty."""
        est = _make_estimator(NLIRelevanceEstimator)
        base_u = 1.0 / 11.0

        op = est._nli_scores_to_opinion(0.0, 0.0, 1.0)

        assert abs(op.belief - 0.0) < 1e-9
        assert abs(op.disbelief - 0.0) < 1e-9
        expected_u = base_u + 1.0 * (1.0 - base_u)
        assert abs(op.uncertainty - expected_u) < 1e-9
        assert _valid_opinion(op)

    def test_pure_neutral(self):
        """NLI = (0, 1, 0) → d=0, high uncertainty."""
        est = _make_estimator(NLIRelevanceEstimator)
        base_u = 1.0 / 11.0

        op = est._nli_scores_to_opinion(0.0, 1.0, 0.0)

        assert abs(op.belief - 0.0) < 1e-9
        assert abs(op.disbelief - 0.0) < 1e-9
        expected_u = base_u + 1.0 * (1.0 - base_u)
        assert abs(op.uncertainty - expected_u) < 1e-9
        assert _valid_opinion(op)

    def test_mixed_scores(self):
        """NLI = (0.6, 0.3, 0.1) → belief from entailment only, d=0."""
        est = _make_estimator(NLIRelevanceEstimator)
        base_u = 1.0 / 11.0

        op = est._nli_scores_to_opinion(0.6, 0.3, 0.1)

        assert abs(op.belief - (0.6 * (1.0 - base_u))) < 1e-9
        assert abs(op.disbelief - 0.0) < 1e-9
        expected_u = base_u + (0.3 + 0.1) * (1.0 - base_u)
        assert abs(op.uncertainty - expected_u) < 1e-9
        assert _valid_opinion(op)

    def test_disbelief_always_zero(self):
        """Relevance mapping must never produce disbelief."""
        est = _make_estimator(NLIRelevanceEstimator)

        test_cases = [
            (0.8, 0.15, 0.05),
            (0.1, 0.1, 0.8),
            (0.33, 0.34, 0.33),
            (0.0, 0.0, 1.0),  # pure contradiction → still d=0
        ]
        for entail, neutral, contradict in test_cases:
            op = est._nli_scores_to_opinion(entail, neutral, contradict)
            assert abs(op.disbelief - 0.0) < 1e-9, (
                f"Relevance disbelief must be 0, got d={op.disbelief} "
                f"for NLI=({entail}, {neutral}, {contradict})"
            )
            assert _valid_opinion(op)

    def test_simplex_preserved(self):
        """b + d + u must equal 1 for any valid NLI scores."""
        est = _make_estimator(NLIRelevanceEstimator, evidence_weight=5.0)

        test_cases = [
            (0.8, 0.15, 0.05),
            (0.1, 0.1, 0.8),
            (0.33, 0.34, 0.33),
            (0.0, 0.5, 0.5),
            (0.5, 0.5, 0.0),
        ]
        for entail, neutral, contradict in test_cases:
            op = est._nli_scores_to_opinion(entail, neutral, contradict)
            assert _valid_opinion(op), (
                f"Invalid opinion for NLI=({entail}, {neutral}, {contradict}): "
                f"b={op.belief}, d={op.disbelief}, u={op.uncertainty}"
            )

    def test_uncertainty_floor(self):
        """Uncertainty must always be >= base_u."""
        for w in [1.0, 5.0, 10.0, 50.0, 100.0]:
            est = _make_estimator(NLIRelevanceEstimator, evidence_weight=w)
            base_u = 1.0 / (w + 1.0)

            op = est._nli_scores_to_opinion(1.0, 0.0, 0.0)
            assert op.uncertainty >= base_u - 1e-9

    def test_evidence_weight_scales_uncertainty(self):
        """Higher evidence weight → lower base uncertainty."""
        ops = []
        for w in [1.0, 10.0, 100.0]:
            est = _make_estimator(NLIRelevanceEstimator, evidence_weight=w)
            op = est._nli_scores_to_opinion(0.8, 0.1, 0.1)
            ops.append(op)

        assert ops[0].uncertainty > ops[1].uncertainty > ops[2].uncertainty

    def test_base_rate_default(self):
        """Opinion base rate should default to 0.5."""
        est = _make_estimator(NLIRelevanceEstimator)
        op = est._nli_scores_to_opinion(0.7, 0.2, 0.1)
        assert abs(op.base_rate - 0.5) < 1e-9


# ===========================================================================
# Faithfulness mapping formula (Stage 2: full 3-class)
# ===========================================================================

class TestFaithfulnessMapping:
    """Test the faithfulness NLI → SL opinion mapping.

    Formula:
        base_u = 1 / (W + 1)
        b = entailment * (1 - base_u)
        d = contradiction * (1 - base_u)
        u = base_u + neutral * (1 - base_u)
    """

    def test_pure_entailment(self):
        """NLI = (1, 0, 0) → high belief, d=0, u=base_u."""
        est = _make_estimator(NLIFaithfulnessEstimator)
        base_u = 1.0 / 11.0

        op = est._nli_scores_to_opinion(1.0, 0.0, 0.0)

        assert abs(op.belief - (1.0 * (1.0 - base_u))) < 1e-9
        assert abs(op.disbelief - 0.0) < 1e-9
        assert abs(op.uncertainty - base_u) < 1e-9
        assert _valid_opinion(op)

    def test_pure_contradiction(self):
        """NLI = (0, 0, 1) → high disbelief."""
        est = _make_estimator(NLIFaithfulnessEstimator)
        base_u = 1.0 / 11.0

        op = est._nli_scores_to_opinion(0.0, 0.0, 1.0)

        assert abs(op.belief - 0.0) < 1e-9
        assert abs(op.disbelief - (1.0 * (1.0 - base_u))) < 1e-9
        assert abs(op.uncertainty - base_u) < 1e-9
        assert _valid_opinion(op)

    def test_pure_neutral(self):
        """NLI = (0, 1, 0) → high uncertainty."""
        est = _make_estimator(NLIFaithfulnessEstimator)
        base_u = 1.0 / 11.0

        op = est._nli_scores_to_opinion(0.0, 1.0, 0.0)

        assert abs(op.belief - 0.0) < 1e-9
        assert abs(op.disbelief - 0.0) < 1e-9
        expected_u = base_u + 1.0 * (1.0 - base_u)
        assert abs(op.uncertainty - expected_u) < 1e-9
        assert _valid_opinion(op)

    def test_mixed_scores(self):
        """NLI = (0.6, 0.3, 0.1) → proportional mapping."""
        est = _make_estimator(NLIFaithfulnessEstimator)
        base_u = 1.0 / 11.0

        op = est._nli_scores_to_opinion(0.6, 0.3, 0.1)

        assert abs(op.belief - (0.6 * (1.0 - base_u))) < 1e-9
        assert abs(op.disbelief - (0.1 * (1.0 - base_u))) < 1e-9
        expected_u = base_u + 0.3 * (1.0 - base_u)
        assert abs(op.uncertainty - expected_u) < 1e-9
        assert _valid_opinion(op)

    def test_contradiction_produces_disbelief(self):
        """Unlike relevance, faithfulness must produce nonzero disbelief."""
        est = _make_estimator(NLIFaithfulnessEstimator)

        op = est._nli_scores_to_opinion(0.1, 0.1, 0.8)
        assert op.disbelief > 0.5, "High contradiction should produce high disbelief"

    def test_simplex_preserved(self):
        """b + d + u must equal 1 for any valid NLI scores."""
        est = _make_estimator(NLIFaithfulnessEstimator, evidence_weight=5.0)

        test_cases = [
            (0.8, 0.15, 0.05),
            (0.1, 0.1, 0.8),
            (0.33, 0.34, 0.33),
            (0.0, 0.5, 0.5),
            (0.5, 0.5, 0.0),
        ]
        for entail, neutral, contradict in test_cases:
            op = est._nli_scores_to_opinion(entail, neutral, contradict)
            assert _valid_opinion(op)

    def test_base_rate_default(self):
        """Opinion base rate should default to 0.5."""
        est = _make_estimator(NLIFaithfulnessEstimator)
        op = est._nli_scores_to_opinion(0.7, 0.2, 0.1)
        assert abs(op.base_rate - 0.5) < 1e-9


# ===========================================================================
# Mapping comparison: relevance vs. faithfulness
# ===========================================================================

class TestMappingComparison:
    """Verify the two mappings differ only in how contradiction is handled."""

    def test_same_belief_for_entailment(self):
        """Both mappings should produce identical belief for same entailment score."""
        rel = _make_estimator(NLIRelevanceEstimator)
        faith = _make_estimator(NLIFaithfulnessEstimator)

        rel_op = rel._nli_scores_to_opinion(0.7, 0.2, 0.1)
        faith_op = faith._nli_scores_to_opinion(0.7, 0.2, 0.1)

        assert abs(rel_op.belief - faith_op.belief) < 1e-9

    def test_relevance_more_uncertain_than_faithfulness(self):
        """Relevance routes contradiction → uncertainty, so u_rel ≥ u_faith."""
        rel = _make_estimator(NLIRelevanceEstimator)
        faith = _make_estimator(NLIFaithfulnessEstimator)

        for entail, neutral, contradict in [(0.6, 0.2, 0.2), (0.3, 0.3, 0.4), (0.1, 0.1, 0.8)]:
            rel_op = rel._nli_scores_to_opinion(entail, neutral, contradict)
            faith_op = faith._nli_scores_to_opinion(entail, neutral, contradict)

            assert rel_op.uncertainty >= faith_op.uncertainty - 1e-9, (
                f"Relevance should be >= faithfulness uncertainty "
                f"for NLI=({entail}, {neutral}, {contradict})"
            )

    def test_same_when_no_contradiction(self):
        """When contradiction=0, both mappings should produce identical opinions."""
        rel = _make_estimator(NLIRelevanceEstimator)
        faith = _make_estimator(NLIFaithfulnessEstimator)

        rel_op = rel._nli_scores_to_opinion(0.7, 0.3, 0.0)
        faith_op = faith._nli_scores_to_opinion(0.7, 0.3, 0.0)

        assert abs(rel_op.belief - faith_op.belief) < 1e-9
        assert abs(rel_op.disbelief - faith_op.disbelief) < 1e-9
        assert abs(rel_op.uncertainty - faith_op.uncertainty) < 1e-9


# ===========================================================================
# estimate() — single pair (mocked NLI model)
# ===========================================================================

class TestRelevanceEstimateSingle:
    """Test NLIRelevanceEstimator.estimate() with mocked model."""

    @pytest.fixture
    def estimator(self):
        return _make_estimator(NLIRelevanceEstimator)

    def test_returns_estimation_result(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.7, 0.2, 0.1)])
        result = estimator.estimate("What is Python?", "Python is a programming language.")
        assert isinstance(result, EstimationResult)

    def test_result_has_valid_opinion(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.7, 0.2, 0.1)])
        result = estimator.estimate("query", "document")
        assert _valid_opinion(result.opinion)

    def test_result_has_nli_scores(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.7, 0.2, 0.1)])
        result = estimator.estimate("query", "document")
        assert result.nli_scores is not None
        assert len(result.nli_scores) == 3
        assert abs(sum(result.nli_scores) - 1.0) < 1e-6

    def test_entailing_doc_high_belief(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.95, 0.03, 0.02)])
        result = estimator.estimate("query", "entailing doc")
        assert result.opinion.belief > 0.5

    def test_contradicting_doc_no_disbelief(self, estimator):
        """In relevance mode, contradiction → uncertainty, NOT disbelief."""
        estimator._predict_nli = MagicMock(return_value=[(0.02, 0.03, 0.95)])
        result = estimator.estimate("query", "contradicting doc")
        assert abs(result.opinion.disbelief - 0.0) < 1e-9
        assert result.opinion.uncertainty > 0.5

    def test_irrelevant_doc_high_uncertainty(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.05, 0.90, 0.05)])
        result = estimator.estimate("query", "irrelevant doc")
        assert result.opinion.uncertainty > 0.5


class TestFaithfulnessEstimateSingle:
    """Test NLIFaithfulnessEstimator.estimate() with mocked model."""

    @pytest.fixture
    def estimator(self):
        return _make_estimator(NLIFaithfulnessEstimator)

    def test_returns_estimation_result(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.7, 0.2, 0.1)])
        result = estimator.estimate("The capital is Paris.", "Paris is the capital of France.")
        assert isinstance(result, EstimationResult)

    def test_entailing_doc_high_belief(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.95, 0.03, 0.02)])
        result = estimator.estimate("generated answer", "supporting document")
        assert result.opinion.belief > 0.5
        assert result.opinion.belief > result.opinion.disbelief

    def test_contradicting_doc_high_disbelief(self, estimator):
        """In faithfulness mode, contradiction → disbelief."""
        estimator._predict_nli = MagicMock(return_value=[(0.02, 0.03, 0.95)])
        result = estimator.estimate("generated answer", "contradicting document")
        assert result.opinion.disbelief > 0.5
        assert result.opinion.disbelief > result.opinion.belief

    def test_irrelevant_doc_high_uncertainty(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.05, 0.90, 0.05)])
        result = estimator.estimate("generated answer", "unrelated document")
        assert result.opinion.uncertainty > 0.5


# ===========================================================================
# estimate_batch() — multiple pairs (mocked NLI model)
# ===========================================================================

class TestBatchEstimation:
    """Test estimate_batch() for both estimator types."""

    @pytest.fixture(params=["relevance", "faithfulness"])
    def estimator(self, request):
        cls = NLIRelevanceEstimator if request.param == "relevance" else NLIFaithfulnessEstimator
        return _make_estimator(cls)

    def test_returns_list_of_results(self, estimator):
        estimator._predict_nli = MagicMock(
            return_value=[(0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)]
        )
        results = estimator.estimate_batch(
            queries=["q1", "q2", "q3"],
            documents=["d1", "d2", "d3"],
        )
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, EstimationResult) for r in results)

    def test_all_opinions_valid(self, estimator):
        estimator._predict_nli = MagicMock(
            return_value=[(0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)]
        )
        results = estimator.estimate_batch(
            queries=["q1", "q2", "q3"],
            documents=["d1", "d2", "d3"],
        )
        for i, r in enumerate(results):
            assert _valid_opinion(r.opinion), f"Invalid opinion at index {i}"

    def test_length_mismatch_raises(self, estimator):
        with pytest.raises(ValueError, match="same length"):
            estimator.estimate_batch(queries=["q1", "q2"], documents=["d1"])

    def test_empty_batch(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[])
        results = estimator.estimate_batch(queries=[], documents=[])
        assert results == []

    def test_batch_preserves_order(self, estimator):
        estimator._predict_nli = MagicMock(
            return_value=[
                (0.9, 0.05, 0.05),
                (0.05, 0.05, 0.9),
                (0.05, 0.9, 0.05),
            ]
        )
        results = estimator.estimate_batch(
            queries=["q1", "q2", "q3"],
            documents=["d_entail", "d_contradict", "d_neutral"],
        )
        # First should have highest belief
        assert results[0].opinion.belief > results[1].opinion.belief
        assert results[0].opinion.belief > results[2].opinion.belief
        # Third should have highest uncertainty
        assert results[2].opinion.uncertainty > results[0].opinion.uncertainty


# ===========================================================================
# Document-document conflict estimation (on NLIEstimator base)
# ===========================================================================

class TestDocumentPairEstimation:
    """Test doc-doc NLI conflict estimation.

    Doc-doc uses the faithfulness mapping since both inputs are declarative.
    This method lives on NLIEstimator but is accessible through subclasses.
    """

    @pytest.fixture
    def estimator(self):
        """Use faithfulness estimator — has access to doc-doc methods."""
        return _make_estimator(NLIFaithfulnessEstimator)

    def test_returns_estimation_result(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.8, 0.1, 0.1)])
        result = estimator.estimate_document_pair(
            doc_a="Paris is the capital of France.",
            doc_b="Paris is the capital of France.",
        )
        assert isinstance(result, EstimationResult)

    def test_agreeing_docs_high_belief(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.9, 0.05, 0.05)])
        result = estimator.estimate_document_pair(
            doc_a="The Earth orbits the Sun.",
            doc_b="Our planet revolves around the Sun.",
        )
        assert result.opinion.belief > 0.5

    def test_contradicting_docs_high_disbelief(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.05, 0.05, 0.9)])
        result = estimator.estimate_document_pair(
            doc_a="Paris is the capital of France.",
            doc_b="Lyon is the capital of France, not Paris.",
        )
        assert result.opinion.disbelief > 0.5

    def test_unrelated_docs_high_uncertainty(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.05, 0.9, 0.05)])
        result = estimator.estimate_document_pair(
            doc_a="Paris is the capital of France.",
            doc_b="Photosynthesis converts sunlight into energy.",
        )
        assert result.opinion.uncertainty > 0.5

    def test_valid_opinion(self, estimator):
        estimator._predict_nli = MagicMock(return_value=[(0.6, 0.2, 0.2)])
        result = estimator.estimate_document_pair(doc_a="A", doc_b="B")
        assert _valid_opinion(result.opinion)

    def test_batch_doc_pairs(self, estimator):
        """Test batched document pair estimation."""
        estimator._predict_nli = MagicMock(
            return_value=[(0.9, 0.05, 0.05), (0.05, 0.05, 0.9)]
        )
        results = estimator.estimate_document_pairs_batch(
            docs_a=["Earth orbits Sun.", "Paris is capital."],
            docs_b=["Planet revolves around Sun.", "Paris is not capital."],
        )
        assert len(results) == 2
        assert all(isinstance(r, EstimationResult) for r in results)
        assert all(_valid_opinion(r.opinion) for r in results)

    def test_batch_doc_pairs_length_mismatch(self, estimator):
        with pytest.raises(ValueError, match="same length"):
            estimator.estimate_document_pairs_batch(
                docs_a=["A", "B"],
                docs_b=["C"],
            )

    def test_doc_pair_accessible_from_relevance_estimator(self):
        """Doc-doc methods should also be accessible from relevance estimator."""
        est = _make_estimator(NLIRelevanceEstimator)
        est._predict_nli = MagicMock(return_value=[(0.8, 0.1, 0.1)])
        result = est.estimate_document_pair(doc_a="A", doc_b="B")
        assert isinstance(result, EstimationResult)
        # Doc-doc should always use faithfulness mapping (both declarative)
        # So even on a relevance estimator, disbelief should be possible
        est._predict_nli = MagicMock(return_value=[(0.05, 0.05, 0.9)])
        result = est.estimate_document_pair(doc_a="A contradicts B.", doc_b="B contradicts A.")
        assert result.opinion.disbelief > 0.5


# ===========================================================================
# Integration tests (real model, slow)
# ===========================================================================

@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationRelevance:
    """Integration tests for NLIRelevanceEstimator with real model."""

    @pytest.fixture(scope="class")
    def estimator(self):
        return NLIRelevanceEstimator(
            model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            evidence_weight=10.0,
            device="cuda",
        )

    def test_relevant_doc_high_belief(self, estimator):
        result = estimator.estimate(
            query="What is the capital of France?",
            document="Paris is the capital and largest city of France.",
        )
        assert _valid_opinion(result.opinion)
        assert result.opinion.belief > result.opinion.uncertainty

    def test_irrelevant_doc_high_uncertainty(self, estimator):
        result = estimator.estimate(
            query="What is the capital of France?",
            document="Photosynthesis converts sunlight into chemical energy in plants.",
        )
        assert _valid_opinion(result.opinion)
        assert result.opinion.uncertainty > result.opinion.belief

    def test_contradicting_doc_still_no_disbelief(self, estimator):
        """Relevance mode must produce d=0 even when NLI detects contradiction."""
        result = estimator.estimate(
            query="What is the capital of France?",
            document="Paris is not the capital of France. The capital is Lyon.",
        )
        assert _valid_opinion(result.opinion)
        assert abs(result.opinion.disbelief - 0.0) < 1e-9


@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationFaithfulness:
    """Integration tests for NLIFaithfulnessEstimator with real model."""

    @pytest.fixture(scope="class")
    def estimator(self):
        return NLIFaithfulnessEstimator(
            model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            evidence_weight=10.0,
            device="cuda",
        )

    def test_supporting_doc_high_belief(self, estimator):
        result = estimator.estimate(
            query="Paris is the capital of France.",
            document="Paris is the capital and largest city of France.",
        )
        assert _valid_opinion(result.opinion)
        assert result.opinion.belief > result.opinion.disbelief

    def test_contradicting_doc_high_disbelief(self, estimator):
        result = estimator.estimate(
            query="Paris is the capital of France.",
            document="Paris is not the capital of France. The capital is Lyon.",
        )
        assert _valid_opinion(result.opinion)
        assert result.opinion.disbelief > result.opinion.belief

    def test_irrelevant_doc_high_uncertainty(self, estimator):
        result = estimator.estimate(
            query="Paris is the capital of France.",
            document="Photosynthesis converts sunlight into chemical energy in plants.",
        )
        assert _valid_opinion(result.opinion)
        assert result.opinion.uncertainty > result.opinion.belief


@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationDocPair:
    """Integration tests for document-document conflict estimation."""

    @pytest.fixture(scope="class")
    def estimator(self):
        return NLIFaithfulnessEstimator(
            model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            evidence_weight=10.0,
            device="cuda",
        )

    def test_agreeing_docs(self, estimator):
        result = estimator.estimate_document_pair(
            doc_a="Paris is the capital of France.",
            doc_b="The capital city of France is Paris.",
        )
        assert _valid_opinion(result.opinion)
        assert result.opinion.belief > result.opinion.disbelief

    def test_contradicting_docs(self, estimator):
        result = estimator.estimate_document_pair(
            doc_a="Paris is the capital of France.",
            doc_b="Paris is not the capital of France. The capital is Lyon.",
        )
        assert _valid_opinion(result.opinion)
        assert result.opinion.disbelief > result.opinion.belief

    def test_batch_matches_singles(self, estimator):
        docs_a = ["Paris is the capital of France.", "Python is a programming language."]
        docs_b = ["The capital of France is Paris.", "Python was created by Guido van Rossum."]

        batch_results = estimator.estimate_document_pairs_batch(docs_a, docs_b)
        single_results = [
            estimator.estimate_document_pair(a, b) for a, b in zip(docs_a, docs_b)
        ]

        for br, sr in zip(batch_results, single_results):
            assert abs(br.opinion.belief - sr.opinion.belief) < 0.01
            assert abs(br.opinion.disbelief - sr.opinion.disbelief) < 0.01
            assert abs(br.opinion.uncertainty - sr.opinion.uncertainty) < 0.01
