"""Tests for the real SL trust discount layer implementation.

Covers uniform and per-doc trust discounting, TrustResult metadata
preservation, mathematical properties of trust_discount(), and edge cases.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion, trust_discount

from xrag.pipeline.trust_layer import (
    TrustLayer,
    NoOpTrustLayer,
    SLTrustLayer,
    TrustResult,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def high_trust():
    """High trust opinion — e.g., Wikipedia source."""
    return Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1, base_rate=0.5)


@pytest.fixture
def low_trust():
    """Low trust opinion — e.g., random forum post."""
    return Opinion(belief=0.3, disbelief=0.2, uncertainty=0.5, base_rate=0.5)


@pytest.fixture
def full_trust():
    """Full trust — discounting should be identity."""
    return Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.5)


@pytest.fixture
def zero_trust():
    """Zero trust (vacuous) — discounting should yield vacuous opinion."""
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


@pytest.fixture
def three_doc_opinions():
    """Three document opinions for batch testing."""
    return [
        Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5),
    ]


# ════════════════════════════════════════════════════════════════════
# TrustResult dataclass
# ════════════════════════════════════════════════════════════════════


class TestTrustResult:
    """Tests for the TrustResult dataclass."""

    def test_basic_construction(self, three_doc_opinions, high_trust):
        discounted = [trust_discount(high_trust, o) for o in three_doc_opinions]
        trusts_used = [high_trust] * 3
        result = TrustResult(
            opinions_before=three_doc_opinions,
            opinions_after=discounted,
            trust_opinions_used=trusts_used,
        )
        assert result.opinions_before is three_doc_opinions
        assert len(result.opinions_after) == 3
        assert len(result.trust_opinions_used) == 3

    def test_preserves_all_three_lists(self, three_doc_opinions, high_trust):
        discounted = [trust_discount(high_trust, o) for o in three_doc_opinions]
        trusts_used = [high_trust] * 3
        result = TrustResult(
            opinions_before=three_doc_opinions,
            opinions_after=discounted,
            trust_opinions_used=trusts_used,
        )
        for i in range(3):
            assert result.opinions_before[i] is three_doc_opinions[i]
            assert result.trust_opinions_used[i] is high_trust


# ════════════════════════════════════════════════════════════════════
# SLTrustLayer — construction
# ════════════════════════════════════════════════════════════════════


class TestSLTrustLayerConstruction:
    """Tests for SLTrustLayer instantiation and configuration."""

    def test_is_subclass_of_trust_layer(self, high_trust):
        layer = SLTrustLayer(default_trust=high_trust)
        assert isinstance(layer, TrustLayer)

    def test_stores_default_trust(self, high_trust):
        layer = SLTrustLayer(default_trust=high_trust)
        assert layer.default_trust is high_trust

    def test_last_result_initially_none(self, high_trust):
        layer = SLTrustLayer(default_trust=high_trust)
        assert layer.last_result is None


# ════════════════════════════════════════════════════════════════════
# SLTrustLayer — uniform trust (default_trust applied to all)
# ════════════════════════════════════════════════════════════════════


class TestUniformTrust:
    """Tests for uniform trust discounting using default_trust."""

    def test_matches_jsonld_ex_directly(self, high_trust, three_doc_opinions):
        """Verify output matches direct trust_discount calls."""
        layer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply(three_doc_opinions)
        for i, op in enumerate(three_doc_opinions):
            expected = trust_discount(high_trust, op)
            assert result[i].belief == pytest.approx(expected.belief)
            assert result[i].disbelief == pytest.approx(expected.disbelief)
            assert result[i].uncertainty == pytest.approx(expected.uncertainty)

    def test_output_length_matches_input(self, high_trust, three_doc_opinions):
        layer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply(three_doc_opinions)
        assert len(result) == len(three_doc_opinions)

    def test_additivity_invariant(self, high_trust, three_doc_opinions):
        """b + d + u = 1 must hold for every discounted opinion."""
        layer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply(three_doc_opinions)
        for op in result:
            assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0)

    def test_last_result_populated(self, high_trust, three_doc_opinions):
        layer = SLTrustLayer(default_trust=high_trust)
        layer.apply(three_doc_opinions)
        lr = layer.last_result
        assert lr is not None
        assert lr.opinions_before is three_doc_opinions
        assert len(lr.opinions_after) == 3
        assert all(t is high_trust for t in lr.trust_opinions_used)

    def test_high_trust_preserves_most_evidence(self, high_trust, three_doc_opinions):
        """High trust (b=0.9) should retain most of the original belief."""
        layer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply(three_doc_opinions)
        for orig, disc in zip(three_doc_opinions, result):
            # With b_trust=0.9: b_discounted = 0.9 * b_orig
            assert disc.belief == pytest.approx(high_trust.belief * orig.belief)

    def test_low_trust_dilutes_toward_uncertainty(self, low_trust, three_doc_opinions):
        """Low trust should increase uncertainty relative to original."""
        layer = SLTrustLayer(default_trust=low_trust)
        result = layer.apply(three_doc_opinions)
        for orig, disc in zip(three_doc_opinions, result):
            assert disc.uncertainty >= orig.uncertainty - 1e-9


# ════════════════════════════════════════════════════════════════════
# SLTrustLayer — per-document trust overrides
# ════════════════════════════════════════════════════════════════════


class TestPerDocTrust:
    """Tests for per-document trust opinion overrides."""

    def test_per_doc_trusts_applied_individually(
        self, high_trust, low_trust, three_doc_opinions
    ):
        """Each doc gets its own trust opinion."""
        per_doc = [high_trust, low_trust, high_trust]
        layer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply(three_doc_opinions, trust_opinions=per_doc)
        # Doc 0: high trust
        expected_0 = trust_discount(high_trust, three_doc_opinions[0])
        assert result[0].belief == pytest.approx(expected_0.belief)
        # Doc 1: low trust
        expected_1 = trust_discount(low_trust, three_doc_opinions[1])
        assert result[1].belief == pytest.approx(expected_1.belief)
        # Doc 2: high trust
        expected_2 = trust_discount(high_trust, three_doc_opinions[2])
        assert result[2].belief == pytest.approx(expected_2.belief)

    def test_last_result_records_per_doc_trusts(
        self, high_trust, low_trust, three_doc_opinions
    ):
        per_doc = [high_trust, low_trust, high_trust]
        layer = SLTrustLayer(default_trust=high_trust)
        layer.apply(three_doc_opinions, trust_opinions=per_doc)
        lr = layer.last_result
        assert lr.trust_opinions_used[0] is high_trust
        assert lr.trust_opinions_used[1] is low_trust
        assert lr.trust_opinions_used[2] is high_trust

    def test_per_doc_length_mismatch_raises(self, high_trust, three_doc_opinions):
        per_doc = [high_trust, high_trust]  # only 2 for 3 docs
        layer = SLTrustLayer(default_trust=high_trust)
        with pytest.raises(ValueError, match="[Ll]ength|[Mm]atch|trust"):
            layer.apply(three_doc_opinions, trust_opinions=per_doc)

    def test_mixed_trust_different_beliefs(self, three_doc_opinions):
        """Documents with different trusts should have different discounted beliefs."""
        very_high = Opinion(belief=0.95, disbelief=0.0, uncertainty=0.05, base_rate=0.5)
        very_low = Opinion(belief=0.1, disbelief=0.3, uncertainty=0.6, base_rate=0.5)
        per_doc = [very_high, very_low, very_high]
        layer = SLTrustLayer(default_trust=very_high)
        result = layer.apply(three_doc_opinions, trust_opinions=per_doc)
        # Doc 0 (high trust) should have much higher belief than doc 1 (low trust),
        # assuming similar original beliefs
        # Doc 0 orig b=0.7, disc b=0.95*0.7=0.665
        # Doc 1 orig b=0.5, disc b=0.1*0.5=0.05
        assert result[0].belief > result[1].belief


# ════════════════════════════════════════════════════════════════════
# Mathematical properties of trust_discount
# ════════════════════════════════════════════════════════════════════


class TestTrustDiscountMathProperties:
    """Verify key mathematical properties through the layer."""

    def test_full_trust_is_identity(self, full_trust, three_doc_opinions):
        """trust_discount with b_trust=1.0 should return the original opinion."""
        layer = SLTrustLayer(default_trust=full_trust)
        result = layer.apply(three_doc_opinions)
        for orig, disc in zip(three_doc_opinions, result):
            assert disc.belief == pytest.approx(orig.belief)
            assert disc.disbelief == pytest.approx(orig.disbelief)
            assert disc.uncertainty == pytest.approx(orig.uncertainty)

    def test_vacuous_trust_yields_vacuous(self, zero_trust, three_doc_opinions):
        """trust_discount with vacuous trust (b=0, u=1) should yield vacuous opinions.

        Formula: b_Ax = b_AB * b_Bx = 0 * b = 0
                 d_Ax = b_AB * d_Bx = 0 * d = 0
                 u_Ax = d_AB + u_AB + b_AB * u_Bx = 0 + 1 + 0 = 1
        """
        layer = SLTrustLayer(default_trust=zero_trust)
        result = layer.apply(three_doc_opinions)
        for op in result:
            assert op.belief == pytest.approx(0.0)
            assert op.disbelief == pytest.approx(0.0)
            assert op.uncertainty == pytest.approx(1.0)

    def test_belief_scales_by_trust_belief(self, three_doc_opinions):
        """b_discounted = b_trust * b_original (the core formula)."""
        trust = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        layer = SLTrustLayer(default_trust=trust)
        result = layer.apply(three_doc_opinions)
        for orig, disc in zip(three_doc_opinions, result):
            assert disc.belief == pytest.approx(trust.belief * orig.belief)

    def test_disbelief_scales_by_trust_belief(self, three_doc_opinions):
        """d_discounted = b_trust * d_original."""
        trust = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        layer = SLTrustLayer(default_trust=trust)
        result = layer.apply(three_doc_opinions)
        for orig, disc in zip(three_doc_opinions, result):
            assert disc.disbelief == pytest.approx(trust.belief * orig.disbelief)

    def test_uncertainty_formula(self, three_doc_opinions):
        """u_discounted = d_trust + u_trust + b_trust * u_original."""
        trust = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        layer = SLTrustLayer(default_trust=trust)
        result = layer.apply(three_doc_opinions)
        for orig, disc in zip(three_doc_opinions, result):
            expected_u = trust.disbelief + trust.uncertainty + trust.belief * orig.uncertainty
            assert disc.uncertainty == pytest.approx(expected_u)

    def test_partial_trust_increases_uncertainty(self, three_doc_opinions):
        """Any trust with b < 1 must increase uncertainty (or leave unchanged if u=1)."""
        trust = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        layer = SLTrustLayer(default_trust=trust)
        result = layer.apply(three_doc_opinions)
        for orig, disc in zip(three_doc_opinions, result):
            # u_disc = d_trust + u_trust + b_trust * u_orig
            # = 0.1 + 0.2 + 0.7 * u_orig = 0.3 + 0.7*u_orig
            # For u_orig < 1: 0.3 + 0.7*u_orig > u_orig iff 0.3 > 0.3*u_orig iff u_orig < 1
            if orig.uncertainty < 1.0:
                assert disc.uncertainty > orig.uncertainty

    def test_base_rate_preserved_from_doc(self, three_doc_opinions):
        """trust_discount preserves the document's base rate, not the trust's."""
        trust = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.9)
        layer = SLTrustLayer(default_trust=trust)
        result = layer.apply(three_doc_opinions)
        for orig, disc in zip(three_doc_opinions, result):
            assert disc.base_rate == pytest.approx(orig.base_rate)


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════


class TestTrustEdgeCases:
    """Edge cases for SLTrustLayer."""

    def test_empty_list_returns_empty(self, high_trust):
        layer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply([])
        assert result == []

    def test_empty_list_last_result(self, high_trust):
        layer = SLTrustLayer(default_trust=high_trust)
        layer.apply([])
        lr = layer.last_result
        assert lr.opinions_before == []
        assert lr.opinions_after == []
        assert lr.trust_opinions_used == []

    def test_single_opinion(self, high_trust):
        op = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        layer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply([op])
        assert len(result) == 1
        expected = trust_discount(high_trust, op)
        assert result[0].belief == pytest.approx(expected.belief)

    def test_last_result_updated_each_call(self, high_trust):
        layer = SLTrustLayer(default_trust=high_trust)
        op1 = [Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)]
        op2 = [
            Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2),
            Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
        ]
        layer.apply(op1)
        assert len(layer.last_result.opinions_before) == 1
        layer.apply(op2)
        assert len(layer.last_result.opinions_before) == 2

    def test_dogmatic_doc_with_partial_trust(self, high_trust):
        """A dogmatic document (u=0) discounted by partial trust gains uncertainty."""
        doc = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        layer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply([doc])
        # u_disc = d_trust + u_trust + b_trust * 0 = 0.0 + 0.1 + 0 = 0.1
        assert result[0].uncertainty == pytest.approx(
            high_trust.disbelief + high_trust.uncertainty
        )


# ════════════════════════════════════════════════════════════════════
# ABC interface compatibility
# ════════════════════════════════════════════════════════════════════


class TestABCCompatibility:
    """Verify SLTrustLayer works through the TrustLayer ABC interface."""

    def test_apply_via_abc_reference(self, high_trust, three_doc_opinions):
        """Pipeline calls layer.apply(opinions) — must work without trust_opinions kwarg."""
        layer: TrustLayer = SLTrustLayer(default_trust=high_trust)
        result = layer.apply(three_doc_opinions)
        assert len(result) == 3
        for op in result:
            assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════════
# NoOpTrustLayer still works
# ════════════════════════════════════════════════════════════════════


class TestNoOpTrustLayerStillWorks:
    """Ensure the existing NoOpTrustLayer is not broken."""

    def test_returns_same_opinions(self, three_doc_opinions):
        layer = NoOpTrustLayer()
        result = layer.apply(three_doc_opinions)
        for orig, out in zip(three_doc_opinions, result):
            assert out.belief == orig.belief
            assert out.disbelief == orig.disbelief
            assert out.uncertainty == orig.uncertainty

    def test_is_trust_layer_subclass(self):
        assert isinstance(NoOpTrustLayer(), TrustLayer)
