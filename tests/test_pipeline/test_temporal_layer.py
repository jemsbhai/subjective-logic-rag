"""Tests for the real SL temporal decay layer implementation.

Covers all 3 decay functions, TemporalResult metadata preservation,
per-doc elapsed times, skip-decay semantics, mathematical properties
(b+d+u=1, ratio preservation, monotonicity), and edge cases.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import (
    decay_opinion,
    exponential_decay,
    linear_decay,
    step_decay,
)

from xrag.pipeline.temporal_layer import (
    TemporalLayer,
    NoOpTemporalLayer,
    SLTemporalLayer,
    TemporalResult,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def three_doc_opinions():
    return [
        Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5),
    ]


@pytest.fixture
def half_life():
    """Half-life in hours (arbitrary unit)."""
    return 24.0


# ════════════════════════════════════════════════════════════════════
# TemporalResult dataclass
# ════════════════════════════════════════════════════════════════════


class TestTemporalResult:
    """Tests for the TemporalResult dataclass."""

    def test_basic_construction(self, three_doc_opinions):
        result = TemporalResult(
            opinions_before=three_doc_opinions,
            opinions_after=three_doc_opinions,
            elapsed_times=[1.0, 2.0, 3.0],
            decay_factors=[0.9, 0.8, 0.7],
        )
        assert result.opinions_before is three_doc_opinions
        assert len(result.elapsed_times) == 3
        assert len(result.decay_factors) == 3

    def test_none_entries_for_skipped_docs(self, three_doc_opinions):
        result = TemporalResult(
            opinions_before=three_doc_opinions,
            opinions_after=three_doc_opinions,
            elapsed_times=[1.0, None, 3.0],
            decay_factors=[0.9, None, 0.7],
        )
        assert result.elapsed_times[1] is None
        assert result.decay_factors[1] is None


# ════════════════════════════════════════════════════════════════════
# SLTemporalLayer — construction
# ════════════════════════════════════════════════════════════════════


class TestSLTemporalLayerConstruction:

    def test_is_subclass(self, half_life):
        layer = SLTemporalLayer(half_life=half_life)
        assert isinstance(layer, TemporalLayer)

    def test_stores_half_life(self, half_life):
        layer = SLTemporalLayer(half_life=half_life)
        assert layer.half_life == half_life

    def test_default_decay_fn_is_exponential(self, half_life):
        layer = SLTemporalLayer(half_life=half_life)
        assert layer.decay_fn is exponential_decay

    def test_custom_decay_fn(self, half_life):
        layer = SLTemporalLayer(half_life=half_life, decay_fn=linear_decay)
        assert layer.decay_fn is linear_decay

    def test_last_result_initially_none(self, half_life):
        layer = SLTemporalLayer(half_life=half_life)
        assert layer.last_result is None

    def test_rejects_non_positive_half_life(self):
        with pytest.raises(ValueError, match="half_life|positive"):
            SLTemporalLayer(half_life=0.0)
        with pytest.raises(ValueError, match="half_life|positive"):
            SLTemporalLayer(half_life=-1.0)


# ════════════════════════════════════════════════════════════════════
# SLTemporalLayer — no elapsed times (passthrough)
# ════════════════════════════════════════════════════════════════════


class TestNoElapsedPassthrough:
    """When elapsed_times is None, all opinions pass through unchanged."""

    def test_passthrough_returns_same_opinions(self, half_life, three_doc_opinions):
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions)
        for orig, out in zip(three_doc_opinions, result):
            assert out.belief == pytest.approx(orig.belief)
            assert out.disbelief == pytest.approx(orig.disbelief)
            assert out.uncertainty == pytest.approx(orig.uncertainty)

    def test_passthrough_last_result(self, half_life, three_doc_opinions):
        layer = SLTemporalLayer(half_life=half_life)
        layer.apply(three_doc_opinions)
        lr = layer.last_result
        assert all(t is None for t in lr.elapsed_times)
        assert all(f is None for f in lr.decay_factors)


# ════════════════════════════════════════════════════════════════════
# SLTemporalLayer — uniform elapsed times
# ════════════════════════════════════════════════════════════════════


class TestUniformElapsed:
    """All docs have the same elapsed time."""

    def test_matches_jsonld_ex_directly(self, half_life, three_doc_opinions):
        elapsed = [12.0, 12.0, 12.0]
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions, elapsed_times=elapsed)
        for i, op in enumerate(three_doc_opinions):
            expected = decay_opinion(op, elapsed=12.0, half_life=half_life)
            assert result[i].belief == pytest.approx(expected.belief)
            assert result[i].disbelief == pytest.approx(expected.disbelief)
            assert result[i].uncertainty == pytest.approx(expected.uncertainty)

    def test_additivity_invariant(self, half_life, three_doc_opinions):
        elapsed = [12.0, 12.0, 12.0]
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions, elapsed_times=elapsed)
        for op in result:
            assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0)

    def test_output_length(self, half_life, three_doc_opinions):
        elapsed = [12.0, 12.0, 12.0]
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions, elapsed_times=elapsed)
        assert len(result) == 3

    def test_last_result_records_elapsed(self, half_life, three_doc_opinions):
        elapsed = [12.0, 24.0, 48.0]
        layer = SLTemporalLayer(half_life=half_life)
        layer.apply(three_doc_opinions, elapsed_times=elapsed)
        lr = layer.last_result
        assert lr.elapsed_times == [12.0, 24.0, 48.0]

    def test_last_result_records_decay_factors(self, half_life, three_doc_opinions):
        elapsed = [0.0, 24.0, 48.0]  # 0 hours, 1 half-life, 2 half-lives
        layer = SLTemporalLayer(half_life=half_life)
        layer.apply(three_doc_opinions, elapsed_times=elapsed)
        lr = layer.last_result
        assert lr.decay_factors[0] == pytest.approx(1.0)      # no decay
        assert lr.decay_factors[1] == pytest.approx(0.5)      # one half-life
        assert lr.decay_factors[2] == pytest.approx(0.25)     # two half-lives


# ════════════════════════════════════════════════════════════════════
# SLTemporalLayer — per-doc elapsed with None entries
# ════════════════════════════════════════════════════════════════════


class TestMixedElapsed:
    """Some docs have timestamps, some don't."""

    def test_none_entry_skips_decay(self, half_life, three_doc_opinions):
        elapsed = [12.0, None, 48.0]
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions, elapsed_times=elapsed)
        # Doc 1 (None elapsed) should pass through unchanged
        assert result[1].belief == pytest.approx(three_doc_opinions[1].belief)
        assert result[1].disbelief == pytest.approx(three_doc_opinions[1].disbelief)
        assert result[1].uncertainty == pytest.approx(three_doc_opinions[1].uncertainty)

    def test_non_none_entries_are_decayed(self, half_life, three_doc_opinions):
        elapsed = [12.0, None, 48.0]
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions, elapsed_times=elapsed)
        # Doc 0 and 2 should be decayed
        expected_0 = decay_opinion(three_doc_opinions[0], elapsed=12.0, half_life=half_life)
        assert result[0].belief == pytest.approx(expected_0.belief)
        expected_2 = decay_opinion(three_doc_opinions[2], elapsed=48.0, half_life=half_life)
        assert result[2].belief == pytest.approx(expected_2.belief)

    def test_last_result_none_entries(self, half_life, three_doc_opinions):
        elapsed = [12.0, None, 48.0]
        layer = SLTemporalLayer(half_life=half_life)
        layer.apply(three_doc_opinions, elapsed_times=elapsed)
        lr = layer.last_result
        assert lr.elapsed_times[1] is None
        assert lr.decay_factors[1] is None
        assert lr.decay_factors[0] is not None
        assert lr.decay_factors[2] is not None

    def test_length_mismatch_raises(self, half_life, three_doc_opinions):
        elapsed = [1.0, 2.0]  # only 2 for 3 docs
        layer = SLTemporalLayer(half_life=half_life)
        with pytest.raises(ValueError, match="[Ll]ength|[Mm]atch"):
            layer.apply(three_doc_opinions, elapsed_times=elapsed)


# ════════════════════════════════════════════════════════════════════
# Decay function variants
# ════════════════════════════════════════════════════════════════════


class TestDecayFunctions:
    """Test all 3 decay function variants through the layer."""

    def test_exponential_at_half_life(self, three_doc_opinions):
        """At elapsed == half_life, exponential decay factor is 0.5."""
        hl = 10.0
        layer = SLTemporalLayer(half_life=hl, decay_fn=exponential_decay)
        layer.apply(three_doc_opinions, elapsed_times=[hl, hl, hl])
        for f in layer.last_result.decay_factors:
            assert f == pytest.approx(0.5)

    def test_linear_at_half_life(self, three_doc_opinions):
        """At elapsed == half_life, linear decay factor is 0.5."""
        hl = 10.0
        layer = SLTemporalLayer(half_life=hl, decay_fn=linear_decay)
        layer.apply(three_doc_opinions, elapsed_times=[hl, hl, hl])
        for f in layer.last_result.decay_factors:
            assert f == pytest.approx(0.5)

    def test_linear_at_double_half_life_is_zero(self, three_doc_opinions):
        """At elapsed == 2*half_life, linear decay reaches zero."""
        hl = 10.0
        layer = SLTemporalLayer(half_life=hl, decay_fn=linear_decay)
        layer.apply(three_doc_opinions, elapsed_times=[20.0, 20.0, 20.0])
        for f in layer.last_result.decay_factors:
            assert f == pytest.approx(0.0)

    def test_step_below_threshold_no_decay(self, three_doc_opinions):
        """Step decay: below half_life → factor = 1.0 (no decay)."""
        hl = 10.0
        layer = SLTemporalLayer(half_life=hl, decay_fn=step_decay)
        layer.apply(three_doc_opinions, elapsed_times=[5.0, 5.0, 5.0])
        for f in layer.last_result.decay_factors:
            assert f == pytest.approx(1.0)

    def test_step_at_threshold_full_decay(self, three_doc_opinions):
        """Step decay: at half_life → factor = 0.0 (fully stale)."""
        hl = 10.0
        layer = SLTemporalLayer(half_life=hl, decay_fn=step_decay)
        layer.apply(three_doc_opinions, elapsed_times=[10.0, 10.0, 10.0])
        for f in layer.last_result.decay_factors:
            assert f == pytest.approx(0.0)

    def test_step_decay_binary_output(self, three_doc_opinions):
        """Step decay produces fully fresh or fully stale opinions."""
        hl = 10.0
        layer = SLTemporalLayer(half_life=hl, decay_fn=step_decay)
        result = layer.apply(three_doc_opinions, elapsed_times=[5.0, 15.0, 5.0])
        # Doc 0, 2: fresh (unchanged)
        assert result[0].belief == pytest.approx(three_doc_opinions[0].belief)
        assert result[2].belief == pytest.approx(three_doc_opinions[2].belief)
        # Doc 1: fully stale (vacuous)
        assert result[1].belief == pytest.approx(0.0)
        assert result[1].disbelief == pytest.approx(0.0)
        assert result[1].uncertainty == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════════
# Mathematical properties
# ════════════════════════════════════════════════════════════════════


class TestDecayMathProperties:
    """Verify key mathematical properties through the layer."""

    def test_zero_elapsed_is_identity(self, half_life, three_doc_opinions):
        """No time elapsed → no decay."""
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions, elapsed_times=[0.0, 0.0, 0.0])
        for orig, out in zip(three_doc_opinions, result):
            assert out.belief == pytest.approx(orig.belief)
            assert out.disbelief == pytest.approx(orig.disbelief)
            assert out.uncertainty == pytest.approx(orig.uncertainty)

    def test_belief_disbelief_ratio_preserved(self, half_life, three_doc_opinions):
        """Decay preserves b/d ratio — we forget how much, not which direction."""
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions, elapsed_times=[12.0, 12.0, 12.0])
        for orig, out in zip(three_doc_opinions, result):
            if orig.disbelief > 1e-10 and out.disbelief > 1e-10:
                orig_ratio = orig.belief / orig.disbelief
                out_ratio = out.belief / out.disbelief
                assert out_ratio == pytest.approx(orig_ratio)

    def test_uncertainty_monotonically_increases(self, half_life, three_doc_opinions):
        """More elapsed time → more uncertainty (for exponential)."""
        layer = SLTemporalLayer(half_life=half_life)
        times_short = [6.0, 6.0, 6.0]
        times_long = [48.0, 48.0, 48.0]
        result_short = layer.apply(three_doc_opinions, elapsed_times=times_short)
        result_long = layer.apply(three_doc_opinions, elapsed_times=times_long)
        for short, long in zip(result_short, result_long):
            assert long.uncertainty >= short.uncertainty - 1e-9

    def test_belief_monotonically_decreases(self, half_life, three_doc_opinions):
        """More elapsed time → less belief."""
        layer = SLTemporalLayer(half_life=half_life)
        result_short = layer.apply(three_doc_opinions, elapsed_times=[6.0, 6.0, 6.0])
        result_long = layer.apply(three_doc_opinions, elapsed_times=[48.0, 48.0, 48.0])
        for short, long in zip(result_short, result_long):
            assert long.belief <= short.belief + 1e-9

    def test_base_rate_preserved(self, half_life):
        """Decay does not alter the base rate."""
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.8)
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply([op], elapsed_times=[12.0])
        assert result[0].base_rate == pytest.approx(0.8)

    def test_very_large_elapsed_approaches_vacuous(self, half_life):
        """With exponential decay, very large elapsed → nearly vacuous."""
        op = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply([op], elapsed_times=[half_life * 100])
        assert result[0].uncertainty > 0.99


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════


class TestTemporalEdgeCases:

    def test_empty_list(self, half_life):
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply([])
        assert result == []

    def test_empty_list_with_empty_elapsed(self, half_life):
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply([], elapsed_times=[])
        assert result == []

    def test_single_opinion(self, half_life):
        op = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply([op], elapsed_times=[12.0])
        expected = decay_opinion(op, elapsed=12.0, half_life=half_life)
        assert result[0].belief == pytest.approx(expected.belief)

    def test_last_result_updated_each_call(self, half_life):
        layer = SLTemporalLayer(half_life=half_life)
        op1 = [Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)]
        op2 = [
            Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2),
            Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
        ]
        layer.apply(op1, elapsed_times=[1.0])
        assert len(layer.last_result.opinions_before) == 1
        layer.apply(op2, elapsed_times=[1.0, 2.0])
        assert len(layer.last_result.opinions_before) == 2

    def test_vacuous_opinion_stays_vacuous(self, half_life):
        """Decaying a vacuous opinion should remain vacuous."""
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        layer = SLTemporalLayer(half_life=half_life)
        result = layer.apply([vac], elapsed_times=[100.0])
        assert result[0].belief == pytest.approx(0.0)
        assert result[0].disbelief == pytest.approx(0.0)
        assert result[0].uncertainty == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════════
# ABC compatibility
# ════════════════════════════════════════════════════════════════════


class TestABCCompatibility:

    def test_apply_via_abc_reference(self, half_life, three_doc_opinions):
        """Pipeline calls layer.apply(opinions) — must work without elapsed_times."""
        layer: TemporalLayer = SLTemporalLayer(half_life=half_life)
        result = layer.apply(three_doc_opinions)
        assert len(result) == 3
        # Without elapsed_times, passthrough
        for orig, out in zip(three_doc_opinions, result):
            assert out.belief == pytest.approx(orig.belief)


# ════════════════════════════════════════════════════════════════════
# NoOpTemporalLayer still works
# ════════════════════════════════════════════════════════════════════


class TestNoOpTemporalLayerStillWorks:

    def test_returns_same_opinions(self, three_doc_opinions):
        layer = NoOpTemporalLayer()
        result = layer.apply(three_doc_opinions)
        for orig, out in zip(three_doc_opinions, result):
            assert out.belief == orig.belief

    def test_is_temporal_layer_subclass(self):
        assert isinstance(NoOpTemporalLayer(), TemporalLayer)
