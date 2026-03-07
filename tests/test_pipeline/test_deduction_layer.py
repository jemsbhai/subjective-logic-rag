"""Tests for the real SL deduction layer implementation.

Covers linear chains, DAG deduction, counterfactual strategies,
DeductionResult metadata, mathematical properties of deduce(),
and edge cases.

This is the most scientifically critical layer — it enables the paper's
core claim that SL propagates uncertainty across reasoning hops in a
principled, algebraic way.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    deduce,
)

from xrag.pipeline.deduction_layer import (
    DeductionLayer,
    NoOpDeductionLayer,
    SLDeductionLayer,
    DeductionResult,
    DeductionStep,
    HopInput,
    vacuous_counterfactual,
    adversarial_counterfactual,
    prior_counterfactual,
    COUNTERFACTUAL_STRATEGIES,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def hop1_confident():
    """Hop 1: strong evidence."""
    return Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5)


@pytest.fixture
def hop2_moderate():
    """Hop 2: moderate evidence."""
    return Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)


@pytest.fixture
def hop3_weak():
    """Hop 3: weak evidence."""
    return Opinion(belief=0.4, disbelief=0.15, uncertainty=0.45, base_rate=0.5)


@pytest.fixture
def hop4_uncertain():
    """Hop 4: high uncertainty."""
    return Opinion(belief=0.2, disbelief=0.1, uncertainty=0.7, base_rate=0.5)


@pytest.fixture
def two_hop_opinions(hop1_confident, hop2_moderate):
    return [hop1_confident, hop2_moderate]


@pytest.fixture
def three_hop_opinions(hop1_confident, hop2_moderate, hop3_weak):
    return [hop1_confident, hop2_moderate, hop3_weak]


@pytest.fixture
def four_hop_opinions(hop1_confident, hop2_moderate, hop3_weak, hop4_uncertain):
    return [hop1_confident, hop2_moderate, hop3_weak, hop4_uncertain]


# ════════════════════════════════════════════════════════════════════
# HopInput dataclass
# ════════════════════════════════════════════════════════════════════


class TestHopInput:

    def test_basic_construction(self, hop1_confident):
        hop = HopInput(
            hop_id=0,
            opinion=hop1_confident,
            sub_question="Who directed Inception?",
            parent_hop_ids=(),
        )
        assert hop.hop_id == 0
        assert hop.opinion is hop1_confident
        assert hop.sub_question == "Who directed Inception?"
        assert hop.parent_hop_ids == ()

    def test_default_parent_ids_empty(self, hop1_confident):
        hop = HopInput(hop_id=0, opinion=hop1_confident)
        assert hop.parent_hop_ids == ()

    def test_with_parents(self, hop2_moderate):
        hop = HopInput(
            hop_id=2,
            opinion=hop2_moderate,
            parent_hop_ids=(0, 1),
        )
        assert hop.parent_hop_ids == (0, 1)


# ════════════════════════════════════════════════════════════════════
# DeductionStep dataclass
# ════════════════════════════════════════════════════════════════════


class TestDeductionStep:

    def test_basic_construction(self, hop1_confident, hop2_moderate):
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
        result_op = deduce(hop1_confident, hop2_moderate, vac)
        step = DeductionStep(
            hop_id=1,
            antecedent=hop1_confident,
            conditional=hop2_moderate,
            counterfactual=vac,
            result=result_op,
        )
        assert step.hop_id == 1
        assert step.antecedent is hop1_confident
        assert step.conditional is hop2_moderate
        assert step.result is result_op


# ════════════════════════════════════════════════════════════════════
# Counterfactual strategies
# ════════════════════════════════════════════════════════════════════


class TestCounterfactualStrategies:

    def test_vacuous_returns_vacuous(self, hop2_moderate):
        cf = vacuous_counterfactual(hop2_moderate)
        assert cf.belief == pytest.approx(0.0)
        assert cf.disbelief == pytest.approx(0.0)
        assert cf.uncertainty == pytest.approx(1.0)

    def test_vacuous_preserves_base_rate(self, hop2_moderate):
        cf = vacuous_counterfactual(hop2_moderate)
        assert cf.base_rate == pytest.approx(hop2_moderate.base_rate)

    def test_adversarial_has_disbelief(self, hop2_moderate):
        cf = adversarial_counterfactual(hop2_moderate)
        assert cf.disbelief > 0.0
        assert cf.belief == pytest.approx(0.0)
        assert cf.belief + cf.disbelief + cf.uncertainty == pytest.approx(1.0)

    def test_adversarial_preserves_base_rate(self, hop2_moderate):
        cf = adversarial_counterfactual(hop2_moderate)
        assert cf.base_rate == pytest.approx(hop2_moderate.base_rate)

    def test_prior_uses_base_rate(self):
        op = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.7)
        cf = prior_counterfactual(op)
        # Prior counterfactual should have belief proportional to base rate
        assert cf.belief + cf.disbelief + cf.uncertainty == pytest.approx(1.0)
        assert cf.base_rate == pytest.approx(0.7)
        # Should have substantial uncertainty (prior, not dogmatic)
        assert cf.uncertainty > 0.3

    def test_all_strategies_registered(self):
        assert "vacuous" in COUNTERFACTUAL_STRATEGIES
        assert "adversarial" in COUNTERFACTUAL_STRATEGIES
        assert "prior" in COUNTERFACTUAL_STRATEGIES
        assert len(COUNTERFACTUAL_STRATEGIES) >= 3

    def test_all_strategies_produce_valid_opinions(self, hop2_moderate):
        for name, fn in COUNTERFACTUAL_STRATEGIES.items():
            cf = fn(hop2_moderate)
            assert cf.belief + cf.disbelief + cf.uncertainty == pytest.approx(1.0), (
                f"Strategy {name} produced invalid opinion"
            )
            assert cf.belief >= 0.0
            assert cf.disbelief >= 0.0
            assert cf.uncertainty >= 0.0

    def test_different_strategies_different_results(
        self, hop1_confident, hop2_moderate
    ):
        """Different counterfactuals should produce different deduced opinions."""
        vac = vacuous_counterfactual(hop2_moderate)
        adv = adversarial_counterfactual(hop2_moderate)
        result_vac = deduce(hop1_confident, hop2_moderate, vac)
        result_adv = deduce(hop1_confident, hop2_moderate, adv)
        # They should differ (adversarial has d > 0, which shifts the result)
        assert result_vac.belief != pytest.approx(result_adv.belief, abs=1e-6) or \
               result_vac.disbelief != pytest.approx(result_adv.disbelief, abs=1e-6)


# ════════════════════════════════════════════════════════════════════
# SLDeductionLayer — construction
# ════════════════════════════════════════════════════════════════════


class TestSLDeductionLayerConstruction:

    def test_is_subclass(self):
        layer = SLDeductionLayer()
        assert isinstance(layer, DeductionLayer)

    def test_default_counterfactual_is_vacuous(self):
        layer = SLDeductionLayer()
        assert layer.counterfactual_fn is vacuous_counterfactual

    def test_custom_counterfactual(self):
        layer = SLDeductionLayer(counterfactual_fn=adversarial_counterfactual)
        assert layer.counterfactual_fn is adversarial_counterfactual

    def test_strategy_name_stored(self):
        layer = SLDeductionLayer(counterfactual_strategy_name="adversarial")
        assert layer.counterfactual_strategy_name == "adversarial"

    def test_last_result_initially_none(self):
        layer = SLDeductionLayer()
        assert layer.last_result is None


# ════════════════════════════════════════════════════════════════════
# Single-hop passthrough
# ════════════════════════════════════════════════════════════════════


class TestSingleHopPassthrough:

    def test_no_hop_opinions_passes_through(self, hop1_confident):
        """When hop_opinions is None, fused_opinion passes through."""
        layer = SLDeductionLayer()
        result = layer.deduce(hop1_confident)
        assert result.belief == pytest.approx(hop1_confident.belief)
        assert result.disbelief == pytest.approx(hop1_confident.disbelief)
        assert result.uncertainty == pytest.approx(hop1_confident.uncertainty)

    def test_single_hop_opinion_passes_through(self, hop1_confident):
        """A single-element hop_opinions is also a passthrough."""
        layer = SLDeductionLayer()
        result = layer.deduce(hop1_confident, hop_opinions=[hop1_confident])
        assert result.belief == pytest.approx(hop1_confident.belief)

    def test_single_hop_last_result(self, hop1_confident):
        layer = SLDeductionLayer()
        layer.deduce(hop1_confident)
        lr = layer.last_result
        assert lr.is_single_hop is True
        assert lr.chain_length <= 1
        assert lr.steps == []


# ════════════════════════════════════════════════════════════════════
# 2-hop linear chain (HotpotQA)
# ════════════════════════════════════════════════════════════════════


class TestTwoHopLinearChain:

    def test_matches_manual_deduce(self, two_hop_opinions):
        """Verify 2-hop chain matches a direct deduce() call."""
        layer = SLDeductionLayer()
        result = layer.deduce(
            two_hop_opinions[0],  # fused_opinion ignored when hop_opinions given
            hop_opinions=two_hop_opinions,
        )
        h1, h2 = two_hop_opinions
        cf = vacuous_counterfactual(h2)
        expected = deduce(h1, h2, cf)
        assert result.belief == pytest.approx(expected.belief)
        assert result.disbelief == pytest.approx(expected.disbelief)
        assert result.uncertainty == pytest.approx(expected.uncertainty)

    def test_additivity_invariant(self, two_hop_opinions):
        layer = SLDeductionLayer()
        result = layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_deduction_result_metadata(self, two_hop_opinions):
        layer = SLDeductionLayer()
        layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        lr = layer.last_result
        assert lr.is_single_hop is False
        assert lr.chain_length == 2
        assert len(lr.steps) == 1  # one deduce() call for 2 hops
        assert lr.counterfactual_strategy == "vacuous"

    def test_step_records_all_inputs(self, two_hop_opinions):
        layer = SLDeductionLayer()
        layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        step = layer.last_result.steps[0]
        assert step.hop_id == 1
        assert step.antecedent is two_hop_opinions[0]
        assert step.conditional is two_hop_opinions[1]
        assert step.counterfactual.uncertainty == pytest.approx(1.0)  # vacuous

    def test_intermediate_opinions_recorded(self, two_hop_opinions):
        layer = SLDeductionLayer()
        layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        lr = layer.last_result
        # For 2-hop: intermediate_opinions should have [hop1, deduced_1_2]
        assert len(lr.intermediate_opinions) == 2
        # First intermediate is hop 1 opinion
        assert lr.intermediate_opinions[0] is two_hop_opinions[0]
        # Second intermediate is the deduced result
        assert lr.intermediate_opinions[1].belief == pytest.approx(lr.final_opinion.belief)


# ════════════════════════════════════════════════════════════════════
# 3-hop and 4-hop linear chains (MuSiQue)
# ════════════════════════════════════════════════════════════════════


class TestDeepLinearChains:

    def test_three_hop_matches_manual(self, three_hop_opinions):
        """3-hop: deduce(deduce(h1, h2, cf), h3, cf)."""
        layer = SLDeductionLayer()
        result = layer.deduce(three_hop_opinions[0], hop_opinions=three_hop_opinions)
        h1, h2, h3 = three_hop_opinions
        cf2 = vacuous_counterfactual(h2)
        cf3 = vacuous_counterfactual(h3)
        step1 = deduce(h1, h2, cf2)
        expected = deduce(step1, h3, cf3)
        assert result.belief == pytest.approx(expected.belief)
        assert result.disbelief == pytest.approx(expected.disbelief)
        assert result.uncertainty == pytest.approx(expected.uncertainty)

    def test_three_hop_two_steps(self, three_hop_opinions):
        layer = SLDeductionLayer()
        layer.deduce(three_hop_opinions[0], hop_opinions=three_hop_opinions)
        assert len(layer.last_result.steps) == 2
        assert layer.last_result.chain_length == 3

    def test_three_hop_intermediates(self, three_hop_opinions):
        layer = SLDeductionLayer()
        layer.deduce(three_hop_opinions[0], hop_opinions=three_hop_opinions)
        lr = layer.last_result
        assert len(lr.intermediate_opinions) == 3
        # Each intermediate should be valid
        for op in lr.intermediate_opinions:
            assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0)

    def test_four_hop_chain(self, four_hop_opinions):
        layer = SLDeductionLayer()
        result = layer.deduce(four_hop_opinions[0], hop_opinions=four_hop_opinions)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        layer_result = layer.last_result
        assert layer_result.chain_length == 4
        assert len(layer_result.steps) == 3
        assert len(layer_result.intermediate_opinions) == 4

    def test_four_hop_matches_manual(self, four_hop_opinions):
        layer = SLDeductionLayer()
        result = layer.deduce(four_hop_opinions[0], hop_opinions=four_hop_opinions)
        h1, h2, h3, h4 = four_hop_opinions
        acc = h1
        acc = deduce(acc, h2, vacuous_counterfactual(h2))
        acc = deduce(acc, h3, vacuous_counterfactual(h3))
        acc = deduce(acc, h4, vacuous_counterfactual(h4))
        assert result.belief == pytest.approx(acc.belief)
        assert result.disbelief == pytest.approx(acc.disbelief)
        assert result.uncertainty == pytest.approx(acc.uncertainty)


# ════════════════════════════════════════════════════════════════════
# DAG deduction (parallel branches)
# ════════════════════════════════════════════════════════════════════


class TestDAGDeduction:
    """Test tree/DAG deduction with parallel branches.

    Example topology:
        hop 0 (root) ─┐
                       ├──> hop 2 (depends on 0 and 1)
        hop 1 (root) ─┘

    This models questions like:
    "Are the birthplaces of the directors of Inception and Interstellar
     in the same country?"
    - Hop 0: "Who directed Inception?"
    - Hop 1: "Who directed Interstellar?"
    - Hop 2: "Are #0's and #1's birthplaces in the same country?"
    """

    def test_parallel_branches_merge(
        self, hop1_confident, hop2_moderate, hop3_weak
    ):
        """Two root hops merge into a third hop."""
        hops = [
            HopInput(hop_id=0, opinion=hop1_confident, parent_hop_ids=()),
            HopInput(hop_id=1, opinion=hop2_moderate, parent_hop_ids=()),
            HopInput(hop_id=2, opinion=hop3_weak, parent_hop_ids=(0, 1)),
        ]
        layer = SLDeductionLayer()
        result = layer.deduce(hop1_confident, hop_inputs=hops)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_parallel_branches_manual_verification(
        self, hop1_confident, hop2_moderate, hop3_weak
    ):
        """Verify DAG result matches manual: fuse parents, then deduce."""
        hops = [
            HopInput(hop_id=0, opinion=hop1_confident, parent_hop_ids=()),
            HopInput(hop_id=1, opinion=hop2_moderate, parent_hop_ids=()),
            HopInput(hop_id=2, opinion=hop3_weak, parent_hop_ids=(0, 1)),
        ]
        layer = SLDeductionLayer()
        result = layer.deduce(hop1_confident, hop_inputs=hops)

        # Manual: fuse the two root opinions as compound antecedent
        fused_parents = cumulative_fuse(hop1_confident, hop2_moderate)
        cf = vacuous_counterfactual(hop3_weak)
        expected = deduce(fused_parents, hop3_weak, cf)

        assert result.belief == pytest.approx(expected.belief)
        assert result.disbelief == pytest.approx(expected.disbelief)
        assert result.uncertainty == pytest.approx(expected.uncertainty)

    def test_dag_metadata(self, hop1_confident, hop2_moderate, hop3_weak):
        hops = [
            HopInput(hop_id=0, opinion=hop1_confident, parent_hop_ids=()),
            HopInput(hop_id=1, opinion=hop2_moderate, parent_hop_ids=()),
            HopInput(hop_id=2, opinion=hop3_weak, parent_hop_ids=(0, 1)),
        ]
        layer = SLDeductionLayer()
        layer.deduce(hop1_confident, hop_inputs=hops)
        lr = layer.last_result
        assert lr.chain_length == 3  # 3 hops
        assert lr.is_single_hop is False
        # 1 deduce() step (hop 2 merging from hop 0 and 1)
        assert len(lr.steps) == 1

    def test_diamond_dag(self, hop1_confident, hop2_moderate, hop3_weak, hop4_uncertain):
        """Diamond topology: 0 -> 1, 0 -> 2, 1+2 -> 3.

        Models: "Who directed X?" -> "Where was #0 born?" and
                "What language is spoken in #0's country?" -> "Is #1 in #2?"
        """
        hops = [
            HopInput(hop_id=0, opinion=hop1_confident, parent_hop_ids=()),
            HopInput(hop_id=1, opinion=hop2_moderate, parent_hop_ids=(0,)),
            HopInput(hop_id=2, opinion=hop3_weak, parent_hop_ids=(0,)),
            HopInput(hop_id=3, opinion=hop4_uncertain, parent_hop_ids=(1, 2)),
        ]
        layer = SLDeductionLayer()
        result = layer.deduce(hop1_confident, hop_inputs=hops)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        lr = layer.last_result
        assert lr.chain_length == 4
        # Steps: deduce(h0, h1), deduce(h0, h2), fuse(ded_1, ded_2) then deduce into h3
        assert len(lr.steps) >= 2

    def test_hop_inputs_and_hop_opinions_mutual_exclusion(self, two_hop_opinions, hop1_confident):
        """Cannot provide both hop_opinions and hop_inputs."""
        hops = [HopInput(hop_id=0, opinion=hop1_confident)]
        layer = SLDeductionLayer()
        with pytest.raises(ValueError, match="[Bb]oth|[Mm]utual|[Ee]xclusive"):
            layer.deduce(
                hop1_confident,
                hop_opinions=two_hop_opinions,
                hop_inputs=hops,
            )


# ════════════════════════════════════════════════════════════════════
# Counterfactual strategy impact
# ════════════════════════════════════════════════════════════════════


class TestCounterfactualImpact:
    """Test that different counterfactuals produce meaningfully different results."""

    def test_adversarial_more_uncertain_than_vacuous(self, two_hop_opinions):
        """Adversarial counterfactual should yield higher uncertainty
        (or at least different disbelief) than vacuous."""
        layer_vac = SLDeductionLayer(counterfactual_fn=vacuous_counterfactual)
        layer_adv = SLDeductionLayer(counterfactual_fn=adversarial_counterfactual)
        result_vac = layer_vac.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        result_adv = layer_adv.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        # Adversarial injects disbelief into the counterfactual,
        # which flows through the deduction formula
        assert result_adv.disbelief > result_vac.disbelief + 1e-9

    def test_strategy_name_in_result(self, two_hop_opinions):
        layer = SLDeductionLayer(
            counterfactual_fn=adversarial_counterfactual,
            counterfactual_strategy_name="adversarial",
        )
        layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        assert layer.last_result.counterfactual_strategy == "adversarial"


# ════════════════════════════════════════════════════════════════════
# Mathematical properties
# ════════════════════════════════════════════════════════════════════


class TestDeductionMathProperties:

    def test_additivity_at_every_step(self, three_hop_opinions):
        """b + d + u = 1 at every intermediate step."""
        layer = SLDeductionLayer()
        layer.deduce(three_hop_opinions[0], hop_opinions=three_hop_opinions)
        for op in layer.last_result.intermediate_opinions:
            assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0)
        for step in layer.last_result.steps:
            assert step.result.belief + step.result.disbelief + step.result.uncertainty == pytest.approx(1.0)

    def test_component_wise_law_of_total_probability(self, two_hop_opinions):
        """Each component of the deduced opinion satisfies:

            c_y = P(ω_x) * c_y|x + (1 - P(ω_x)) * c_y|¬x

        for c ∈ {b, d, u}.  This is the subjective-logic generalization
        of the law of total probability, applied component-wise.

        Note: the projected probability P(ω_y) = b_y + a_y·u_y does NOT
        satisfy P(y) = P(x)·P(y|x) + (1-P(x))·P(y|¬x) in general,
        because a_y is computed from a_x, P(y|x), P(y|¬x) and differs
        from the conditional base rates.  This is mathematically correct
        per Jøsang (2016, §12.6).
        """
        h1, h2 = two_hop_opinions
        layer = SLDeductionLayer()
        result = layer.deduce(h1, hop_opinions=two_hop_opinions)
        cf = vacuous_counterfactual(h2)
        p_x = h1.projected_probability()
        # Component-wise: c_y = P(x) * c_y|x + (1 - P(x)) * c_y|¬x
        expected_b = p_x * h2.belief + (1 - p_x) * cf.belief
        expected_d = p_x * h2.disbelief + (1 - p_x) * cf.disbelief
        expected_u = p_x * h2.uncertainty + (1 - p_x) * cf.uncertainty
        assert result.belief == pytest.approx(expected_b)
        assert result.disbelief == pytest.approx(expected_d)
        assert result.uncertainty == pytest.approx(expected_u)

    def test_vacuous_antecedent_scales_by_base_rate(self, hop2_moderate):
        """If antecedent is vacuous, b_y = a_x * b_y|x (NOT vacuous result).

        This is a critical mathematical fact: vacuous antecedent doesn't
        yield vacuous conclusion — it scales by the base rate. This is
        algebraically correct per Jøsang's deduction formula.
        """
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
        layer = SLDeductionLayer()
        result = layer.deduce(vac, hop_opinions=[vac, hop2_moderate])
        # b_y = b_y|x * P(vac) = b_y|x * a = 0.6 * 0.5 = 0.3
        expected_b = hop2_moderate.belief * vac.base_rate
        assert result.belief == pytest.approx(expected_b)

    def test_all_dogmatic_chain_is_deterministic(self):
        """If every hop is dogmatic (u=0), deduction is deterministic.

        With vacuous counterfactual, dogmatic h1 (b=1, u=0) and
        dogmatic h2 (b=1, u=0):
        deduce((1,0,0,a), (1,0,0,a), (0,0,1,a))
        = b_y = 1*1 + 0*0 + 0*(a*1 + ā*0) = 1
        """
        h1 = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.5)
        h2 = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.5)
        layer = SLDeductionLayer()
        result = layer.deduce(h1, hop_opinions=[h1, h2])
        assert result.belief == pytest.approx(1.0)
        assert result.uncertainty == pytest.approx(0.0)

    def test_vacuous_hop_in_chain_increases_uncertainty(self, hop1_confident):
        """A vacuous hop in the chain should push result toward uncertainty."""
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
        layer = SLDeductionLayer()
        result = layer.deduce(hop1_confident, hop_opinions=[hop1_confident, vac])
        # With vacuous conditional, all conditional mass is in u_y|x = 1
        # deduce produces high uncertainty
        assert result.uncertainty > hop1_confident.uncertainty

    def test_deeper_chains_generally_increase_uncertainty_with_imperfect_hops(
        self, two_hop_opinions, three_hop_opinions, four_hop_opinions
    ):
        """With imperfect hops and vacuous counterfactual, deeper chains
        should generally increase final uncertainty.

        This isn't a strict mathematical guarantee but a strong tendency
        that the paper relies on for the "uncertainty propagation" story.
        """
        layer = SLDeductionLayer()
        r2 = layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        u2 = r2.uncertainty
        r3 = layer.deduce(three_hop_opinions[0], hop_opinions=three_hop_opinions)
        u3 = r3.uncertainty
        r4 = layer.deduce(four_hop_opinions[0], hop_opinions=four_hop_opinions)
        u4 = r4.uncertainty
        # Monotonic increase in uncertainty with chain depth
        assert u3 >= u2 - 1e-9
        assert u4 >= u3 - 1e-9

    def test_base_rate_propagation(self, two_hop_opinions):
        """Deduced opinion base rate follows the formula:
        a_y = a_x * P(y|x) + ā_x * P(y|¬x)."""
        h1, h2 = two_hop_opinions
        cf = vacuous_counterfactual(h2)
        layer = SLDeductionLayer()
        result = layer.deduce(h1, hop_opinions=two_hop_opinions)
        a_x = h1.base_rate
        p_y_x = h2.projected_probability()
        p_y_nx = cf.projected_probability()
        expected_a = a_x * p_y_x + (1 - a_x) * p_y_nx
        assert result.base_rate == pytest.approx(expected_a)


# ════════════════════════════════════════════════════════════════════
# Sub-question metadata traceability
# ════════════════════════════════════════════════════════════════════


class TestSubQuestionTraceability:

    def test_sub_questions_recorded_in_result(self, two_hop_opinions):
        sqs = ["Who directed Inception?", "Where was #1 born?"]
        layer = SLDeductionLayer()
        layer.deduce(
            two_hop_opinions[0],
            hop_opinions=two_hop_opinions,
            sub_questions=sqs,
        )
        assert layer.last_result.sub_questions == sqs

    def test_sub_questions_none_when_not_provided(self, two_hop_opinions):
        layer = SLDeductionLayer()
        layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        assert layer.last_result.sub_questions is None


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════


class TestDeductionEdgeCases:

    def test_empty_hop_opinions_passthrough(self, hop1_confident):
        """Empty hop_opinions list → passthrough."""
        layer = SLDeductionLayer()
        result = layer.deduce(hop1_confident, hop_opinions=[])
        assert result.belief == pytest.approx(hop1_confident.belief)

    def test_last_result_updated_each_call(
        self, hop1_confident, two_hop_opinions
    ):
        layer = SLDeductionLayer()
        layer.deduce(hop1_confident)
        assert layer.last_result.is_single_hop is True
        layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        assert layer.last_result.is_single_hop is False

    def test_all_vacuous_chain(self):
        """All vacuous hops → high uncertainty result (but not necessarily u=1)."""
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
        layer = SLDeductionLayer()
        result = layer.deduce(vac, hop_opinions=[vac, vac, vac])
        # With vacuous antecedent and vacuous conditional:
        # b_y = 0, d_y = 0, u_y = 1 (all mass in uncertainty)
        assert result.uncertainty == pytest.approx(1.0)
        assert result.belief == pytest.approx(0.0)
        assert result.disbelief == pytest.approx(0.0)

    def test_custom_counterfactual_function(self, two_hop_opinions):
        """User can provide any callable as counterfactual."""
        def custom_cf(conditional: Opinion) -> Opinion:
            # Always return a specific opinion
            return Opinion(belief=0.1, disbelief=0.1, uncertainty=0.8, base_rate=0.5)

        layer = SLDeductionLayer(
            counterfactual_fn=custom_cf,
            counterfactual_strategy_name="custom",
        )
        result = layer.deduce(two_hop_opinions[0], hop_opinions=two_hop_opinions)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        assert layer.last_result.counterfactual_strategy == "custom"


# ════════════════════════════════════════════════════════════════════
# ABC compatibility
# ════════════════════════════════════════════════════════════════════


class TestABCCompatibility:

    def test_deduce_via_abc_reference(self, hop1_confident):
        """Pipeline calls layer.deduce(fused_opinion, sub_questions=...)."""
        layer: DeductionLayer = SLDeductionLayer()
        result = layer.deduce(hop1_confident, sub_questions=["test?"])
        assert result.belief == pytest.approx(hop1_confident.belief)


# ════════════════════════════════════════════════════════════════════
# NoOpDeductionLayer still works
# ════════════════════════════════════════════════════════════════════


class TestNoOpDeductionLayerStillWorks:

    def test_returns_same_opinion(self, hop1_confident):
        layer = NoOpDeductionLayer()
        result = layer.deduce(hop1_confident)
        assert result.belief == pytest.approx(hop1_confident.belief)

    def test_is_deduction_layer_subclass(self):
        assert isinstance(NoOpDeductionLayer(), DeductionLayer)
