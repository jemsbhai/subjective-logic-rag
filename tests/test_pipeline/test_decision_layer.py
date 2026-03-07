"""Tests for the real SL decision layer implementation.

Covers threshold-based abstention, conflict flagging, priority ordering,
DecisionResult metadata, and edge cases.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion

from xrag.pipeline.conflict_layer import ConflictResult
from xrag.pipeline.decision_layer import (
    DecisionLayer,
    NoOpDecisionLayer,
    SLDecisionLayer,
    DecisionResult,
    SLDecisionResult,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def no_conflict():
    """ConflictResult with no conflict detected."""
    return ConflictResult(
        conflict_detected=False,
        conflict_pairs=[],
        max_conflict_score=0.05,
    )


@pytest.fixture
def high_conflict():
    """ConflictResult with high conflict."""
    return ConflictResult(
        conflict_detected=True,
        conflict_pairs=[(0, 1)],
        max_conflict_score=0.8,
    )


@pytest.fixture
def confident_opinion():
    """High belief, low uncertainty — should generate."""
    return Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5)


@pytest.fixture
def uncertain_opinion():
    """High uncertainty — should abstain."""
    return Opinion(belief=0.1, disbelief=0.05, uncertainty=0.85, base_rate=0.5)


@pytest.fixture
def vacuous_opinion():
    """Maximum uncertainty."""
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


# ════════════════════════════════════════════════════════════════════
# SLDecisionLayer — construction
# ════════════════════════════════════════════════════════════════════


class TestSLDecisionLayerConstruction:

    def test_is_subclass(self):
        layer = SLDecisionLayer()
        assert isinstance(layer, DecisionLayer)

    def test_stores_thresholds(self):
        layer = SLDecisionLayer(tau_abstain=0.6, tau_conflict=0.4)
        assert layer.tau_abstain == 0.6
        assert layer.tau_conflict == 0.4

    def test_default_thresholds(self):
        layer = SLDecisionLayer()
        # Sensible defaults — not too aggressive, not too permissive
        assert 0.0 < layer.tau_abstain < 1.0
        assert 0.0 < layer.tau_conflict < 1.0

    def test_last_result_initially_none(self):
        layer = SLDecisionLayer()
        assert layer.last_result is None


# ════════════════════════════════════════════════════════════════════
# GENERATE decisions
# ════════════════════════════════════════════════════════════════════


class TestGenerateDecision:

    def test_confident_no_conflict_generates(self, confident_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(confident_opinion, no_conflict)
        assert result.decision == "generate"

    def test_reason_is_nonempty(self, confident_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(confident_opinion, no_conflict)
        assert len(result.reason) > 0

    def test_moderate_uncertainty_below_threshold(self, no_conflict):
        """Uncertainty at 0.5 with threshold 0.7 → generate."""
        op = Opinion(belief=0.3, disbelief=0.2, uncertainty=0.5)
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(op, no_conflict)
        assert result.decision == "generate"


# ════════════════════════════════════════════════════════════════════
# ABSTAIN decisions
# ════════════════════════════════════════════════════════════════════


class TestAbstainDecision:

    def test_high_uncertainty_abstains(self, uncertain_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(uncertain_opinion, no_conflict)
        assert result.decision == "abstain"

    def test_vacuous_abstains(self, vacuous_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(vacuous_opinion, no_conflict)
        assert result.decision == "abstain"

    def test_reason_mentions_uncertainty(self, uncertain_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(uncertain_opinion, no_conflict)
        assert "uncert" in result.reason.lower() or "abstain" in result.reason.lower()

    def test_uncertainty_exactly_at_threshold_generates(self, no_conflict):
        """Uncertainty == tau_abstain should NOT abstain (strictly greater)."""
        op = Opinion(belief=0.15, disbelief=0.15, uncertainty=0.7)
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(op, no_conflict)
        assert result.decision == "generate"

    def test_uncertainty_just_above_threshold_abstains(self, no_conflict):
        op = Opinion(belief=0.1, disbelief=0.19, uncertainty=0.71)
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(op, no_conflict)
        assert result.decision == "abstain"


# ════════════════════════════════════════════════════════════════════
# FLAG_CONFLICT decisions
# ════════════════════════════════════════════════════════════════════


class TestFlagConflictDecision:

    def test_high_conflict_flags(self, confident_opinion, high_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(confident_opinion, high_conflict)
        assert result.decision == "flag_conflict"

    def test_reason_mentions_conflict(self, confident_opinion, high_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(confident_opinion, high_conflict)
        assert "conflict" in result.reason.lower()

    def test_conflict_exactly_at_threshold_generates(self, confident_opinion):
        """Conflict == tau_conflict should NOT flag (strictly greater)."""
        conflict = ConflictResult(
            conflict_detected=False,
            conflict_pairs=[],
            max_conflict_score=0.5,
        )
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(confident_opinion, conflict)
        assert result.decision == "generate"


# ════════════════════════════════════════════════════════════════════
# Priority ordering: conflict > abstain > generate
# ════════════════════════════════════════════════════════════════════


class TestDecisionPriority:

    def test_conflict_takes_priority_over_abstain(self, uncertain_opinion, high_conflict):
        """When BOTH uncertainty and conflict exceed thresholds,
        conflict should take priority."""
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(uncertain_opinion, high_conflict)
        assert result.decision == "flag_conflict"

    def test_conflict_priority_with_vacuous(self, vacuous_opinion, high_conflict):
        """Even maximum uncertainty yields flag_conflict when conflict is high."""
        layer = SLDecisionLayer(tau_abstain=0.5, tau_conflict=0.3)
        result = layer.decide(vacuous_opinion, high_conflict)
        assert result.decision == "flag_conflict"

    def test_abstain_when_no_conflict(self, uncertain_opinion, no_conflict):
        """High uncertainty without conflict → abstain, not flag."""
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(uncertain_opinion, no_conflict)
        assert result.decision == "abstain"


# ════════════════════════════════════════════════════════════════════
# SLDecisionResult — enriched metadata
# ════════════════════════════════════════════════════════════════════


class TestSLDecisionResult:

    def test_records_fused_uncertainty(self, confident_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        layer.decide(confident_opinion, no_conflict)
        lr = layer.last_result
        assert lr.fused_uncertainty == pytest.approx(confident_opinion.uncertainty)

    def test_records_max_conflict_score(self, confident_opinion, high_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        layer.decide(confident_opinion, high_conflict)
        lr = layer.last_result
        assert lr.max_conflict_score == pytest.approx(high_conflict.max_conflict_score)

    def test_records_thresholds(self, confident_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.6, tau_conflict=0.4)
        layer.decide(confident_opinion, no_conflict)
        lr = layer.last_result
        assert lr.tau_abstain == 0.6
        assert lr.tau_conflict == 0.4

    def test_records_decision_and_reason(self, uncertain_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        layer.decide(uncertain_opinion, no_conflict)
        lr = layer.last_result
        assert lr.decision == "abstain"
        assert len(lr.reason) > 0

    def test_records_projected_probability(self, confident_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        layer.decide(confident_opinion, no_conflict)
        lr = layer.last_result
        expected_pp = confident_opinion.projected_probability()
        assert lr.projected_probability == pytest.approx(expected_pp)

    def test_last_result_updated_each_call(self, confident_opinion, uncertain_opinion, no_conflict):
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        layer.decide(confident_opinion, no_conflict)
        assert layer.last_result.decision == "generate"
        layer.decide(uncertain_opinion, no_conflict)
        assert layer.last_result.decision == "abstain"


# ════════════════════════════════════════════════════════════════════
# Threshold configuration
# ════════════════════════════════════════════════════════════════════


class TestThresholdConfiguration:

    def test_very_low_abstain_threshold_always_abstains(self, confident_opinion, no_conflict):
        """tau_abstain=0.0 means ANY uncertainty triggers abstain."""
        layer = SLDecisionLayer(tau_abstain=0.0, tau_conflict=0.99)
        # confident_opinion has u=0.15 > 0.0
        result = layer.decide(confident_opinion, no_conflict)
        assert result.decision == "abstain"

    def test_tau_abstain_one_never_abstains(self, vacuous_opinion, no_conflict):
        """tau_abstain=1.0 means nothing triggers abstain (u can't exceed 1)."""
        layer = SLDecisionLayer(tau_abstain=1.0, tau_conflict=0.99)
        result = layer.decide(vacuous_opinion, no_conflict)
        assert result.decision == "generate"

    def test_very_low_conflict_threshold(self, confident_opinion):
        """tau_conflict=0.0 means any nonzero conflict triggers flag."""
        conflict = ConflictResult(
            conflict_detected=True,
            conflict_pairs=[(0, 1)],
            max_conflict_score=0.01,
        )
        layer = SLDecisionLayer(tau_abstain=0.99, tau_conflict=0.0)
        result = layer.decide(confident_opinion, conflict)
        assert result.decision == "flag_conflict"


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════


class TestDecisionEdgeCases:

    def test_dogmatic_belief_generates(self, no_conflict):
        """Dogmatic opinion (u=0) should always generate."""
        dog = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        layer = SLDecisionLayer(tau_abstain=0.5, tau_conflict=0.5)
        result = layer.decide(dog, no_conflict)
        assert result.decision == "generate"

    def test_dogmatic_disbelief_generates(self, no_conflict):
        """High disbelief but low uncertainty still generates
        (decision is about whether we KNOW, not about what we know)."""
        dog = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        layer = SLDecisionLayer(tau_abstain=0.5, tau_conflict=0.5)
        result = layer.decide(dog, no_conflict)
        assert result.decision == "generate"

    def test_zero_conflict_score(self, confident_opinion):
        conflict = ConflictResult(
            conflict_detected=False,
            conflict_pairs=[],
            max_conflict_score=0.0,
        )
        layer = SLDecisionLayer(tau_abstain=0.7, tau_conflict=0.5)
        result = layer.decide(confident_opinion, conflict)
        assert result.decision == "generate"


# ════════════════════════════════════════════════════════════════════
# NoOpDecisionLayer still works
# ════════════════════════════════════════════════════════════════════


class TestNoOpDecisionLayerStillWorks:

    def test_always_generates(self, uncertain_opinion, high_conflict):
        layer = NoOpDecisionLayer()
        result = layer.decide(uncertain_opinion, high_conflict)
        assert result.decision == "generate"

    def test_is_decision_layer_subclass(self):
        assert isinstance(NoOpDecisionLayer(), DecisionLayer)
