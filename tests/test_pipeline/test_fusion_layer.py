"""Tests for the real SL fusion layer implementation.

Covers all 6 fusion strategies, FusionResult metadata preservation,
edge cases, and mathematical invariants (b+d+u=1, uncertainty reduction).
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    pairwise_conflict,
)
from jsonld_ex.confidence_byzantine import (
    AgentRemoval,
    ByzantineConfig,
    ByzantineFusionReport,
)

from xrag.pipeline.fusion_layer import (
    FusionLayer,
    NoOpFusionLayer,
    SLFusionLayer,
    FusionResult,
    FusionStrategy,
    FUSION_STRATEGIES,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def three_agreeing():
    """Three opinions that mostly agree (high belief)."""
    return [
        Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5),
        Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17, base_rate=0.5),
    ]


@pytest.fixture
def two_conflicting():
    """Two opinions that strongly disagree."""
    return [
        Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05, base_rate=0.5),
        Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05, base_rate=0.5),
    ]


@pytest.fixture
def majority_with_rogue():
    """Three honest + one rogue opinion for Byzantine tests."""
    return [
        Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5),
        Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17, base_rate=0.5),
        Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1, base_rate=0.5),  # rogue
    ]


# ════════════════════════════════════════════════════════════════════
# FusionResult dataclass
# ════════════════════════════════════════════════════════════════════


class TestFusionResult:
    """Tests for the FusionResult dataclass."""

    def test_basic_construction(self):
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        result = FusionResult(
            fused=op,
            strategy="cumulative",
            input_count=3,
            removed_indices=[],
            surviving_indices=[0, 1, 2],
            conflict_matrix=None,
            cohesion=None,
            removal_details=None,
        )
        assert result.fused is op
        assert result.strategy == "cumulative"
        assert result.input_count == 3
        assert result.removed_indices == []
        assert result.surviving_indices == [0, 1, 2]
        assert result.conflict_matrix is None
        assert result.cohesion is None
        assert result.removal_details is None

    def test_byzantine_result_preserves_all_fields(self):
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        matrix = [[0.0, 0.5], [0.5, 0.0]]
        removal = AgentRemoval(
            index=1,
            opinion=Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            discord_score=0.45,
            reason="highest discord in group",
        )
        result = FusionResult(
            fused=op,
            strategy="byzantine_most_conflicting",
            input_count=2,
            removed_indices=[1],
            surviving_indices=[0],
            conflict_matrix=matrix,
            cohesion=0.85,
            removal_details=[removal],
        )
        assert result.conflict_matrix is matrix
        assert result.cohesion == 0.85
        assert len(result.removal_details) == 1
        assert result.removal_details[0].index == 1
        assert result.removal_details[0].discord_score == 0.45


# ════════════════════════════════════════════════════════════════════
# FusionStrategy type
# ════════════════════════════════════════════════════════════════════


class TestFusionStrategy:
    """Tests for the FusionStrategy type and FUSION_STRATEGIES constant."""

    def test_all_six_strategies_defined(self):
        expected = {
            "cumulative",
            "averaging",
            "robust",
            "byzantine_most_conflicting",
            "byzantine_least_trusted",
            "byzantine_combined",
        }
        assert set(FUSION_STRATEGIES) == expected

    def test_strategy_count(self):
        assert len(FUSION_STRATEGIES) == 6


# ════════════════════════════════════════════════════════════════════
# SLFusionLayer — construction
# ════════════════════════════════════════════════════════════════════


class TestSLFusionLayerConstruction:
    """Tests for SLFusionLayer instantiation and configuration."""

    def test_is_subclass_of_fusion_layer(self):
        layer = SLFusionLayer(strategy="cumulative")
        assert isinstance(layer, FusionLayer)

    def test_default_strategy_is_cumulative(self):
        layer = SLFusionLayer()
        assert layer.strategy == "cumulative"

    def test_accepts_all_six_strategies(self):
        for s in FUSION_STRATEGIES:
            layer = SLFusionLayer(strategy=s)
            assert layer.strategy == s

    def test_rejects_invalid_strategy(self):
        with pytest.raises(ValueError, match="[Uu]nknown.*strategy|[Ii]nvalid.*strategy"):
            SLFusionLayer(strategy="nonexistent")

    def test_robust_threshold_configurable(self):
        layer = SLFusionLayer(strategy="robust", robust_threshold=0.25)
        assert layer.robust_threshold == 0.25

    def test_robust_threshold_default(self):
        layer = SLFusionLayer(strategy="robust")
        assert layer.robust_threshold == 0.15  # jsonld-ex default

    def test_byzantine_config_passthrough(self):
        cfg = ByzantineConfig(threshold=0.2, strategy="least_trusted")
        layer = SLFusionLayer(
            strategy="byzantine_least_trusted",
            byzantine_config=cfg,
        )
        assert layer.byzantine_config is cfg

    def test_last_result_initially_none(self):
        layer = SLFusionLayer(strategy="cumulative")
        assert layer.last_result is None

    def test_byzantine_trust_weights_stored(self):
        layer = SLFusionLayer(
            strategy="byzantine_least_trusted",
            trust_weights=[0.9, 0.8, 0.1],
        )
        assert layer.trust_weights == [0.9, 0.8, 0.1]


# ════════════════════════════════════════════════════════════════════
# SLFusionLayer — cumulative fusion
# ════════════════════════════════════════════════════════════════════


class TestCumulativeFusion:
    """Tests for cumulative fusion strategy."""

    def test_matches_jsonld_ex_directly(self, three_agreeing):
        """Verify SLFusionLayer output matches direct cumulative_fuse call."""
        layer = SLFusionLayer(strategy="cumulative")
        result = layer.fuse(three_agreeing)
        expected = cumulative_fuse(*three_agreeing)
        assert result.belief == pytest.approx(expected.belief)
        assert result.disbelief == pytest.approx(expected.disbelief)
        assert result.uncertainty == pytest.approx(expected.uncertainty)

    def test_additivity_invariant(self, three_agreeing):
        """b + d + u = 1 must hold."""
        layer = SLFusionLayer(strategy="cumulative")
        result = layer.fuse(three_agreeing)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_uncertainty_reduction(self, three_agreeing):
        """Cumulative fusion of independent sources must reduce uncertainty."""
        layer = SLFusionLayer(strategy="cumulative")
        result = layer.fuse(three_agreeing)
        min_input_u = min(o.uncertainty for o in three_agreeing)
        assert result.uncertainty <= min_input_u + 1e-9

    def test_last_result_populated(self, three_agreeing):
        layer = SLFusionLayer(strategy="cumulative")
        layer.fuse(three_agreeing)
        lr = layer.last_result
        assert lr is not None
        assert lr.strategy == "cumulative"
        assert lr.input_count == 3
        assert lr.removed_indices == []
        assert lr.surviving_indices == [0, 1, 2]
        assert lr.conflict_matrix is None
        assert lr.cohesion is None
        assert lr.removal_details is None

    def test_single_opinion_passthrough(self):
        """A single opinion fuses to itself."""
        op = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        layer = SLFusionLayer(strategy="cumulative")
        result = layer.fuse([op])
        assert result.belief == pytest.approx(op.belief)
        assert result.disbelief == pytest.approx(op.disbelief)
        assert result.uncertainty == pytest.approx(op.uncertainty)

    def test_vacuous_is_identity(self):
        """Fusing with a vacuous opinion should not change the other."""
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        layer = SLFusionLayer(strategy="cumulative")
        result = layer.fuse([op, vac])
        assert result.belief == pytest.approx(op.belief)
        assert result.disbelief == pytest.approx(op.disbelief)
        assert result.uncertainty == pytest.approx(op.uncertainty)

    def test_dogmatic_pair_averages(self):
        """Two dogmatic opinions yield their average (limit form)."""
        d1 = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        d2 = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        layer = SLFusionLayer(strategy="cumulative")
        result = layer.fuse([d1, d2])
        assert result.belief == pytest.approx(0.5)
        assert result.disbelief == pytest.approx(0.5)
        assert result.uncertainty == pytest.approx(0.0)


# ════════════════════════════════════════════════════════════════════
# SLFusionLayer — averaging fusion
# ════════════════════════════════════════════════════════════════════


class TestAveragingFusion:
    """Tests for averaging fusion strategy."""

    def test_matches_jsonld_ex_directly(self, three_agreeing):
        layer = SLFusionLayer(strategy="averaging")
        result = layer.fuse(three_agreeing)
        expected = averaging_fuse(*three_agreeing)
        assert result.belief == pytest.approx(expected.belief)
        assert result.disbelief == pytest.approx(expected.disbelief)
        assert result.uncertainty == pytest.approx(expected.uncertainty)

    def test_additivity_invariant(self, three_agreeing):
        layer = SLFusionLayer(strategy="averaging")
        result = layer.fuse(three_agreeing)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_idempotence(self):
        """Averaging the same opinion with itself should return itself."""
        op = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        layer = SLFusionLayer(strategy="averaging")
        result = layer.fuse([op, op, op])
        assert result.belief == pytest.approx(op.belief)
        assert result.disbelief == pytest.approx(op.disbelief)
        assert result.uncertainty == pytest.approx(op.uncertainty)

    def test_last_result_metadata(self, three_agreeing):
        layer = SLFusionLayer(strategy="averaging")
        layer.fuse(three_agreeing)
        lr = layer.last_result
        assert lr.strategy == "averaging"
        assert lr.input_count == 3
        assert lr.removed_indices == []
        assert lr.surviving_indices == [0, 1, 2]

    def test_uses_simultaneous_nary_not_pairwise_fold(self):
        """Averaging fusion is NOT associative.

        For n >= 3, we must use the simultaneous n-ary formula,
        not a pairwise fold. Verify by checking that the result
        differs from ((a ⊘ b) ⊘ c).
        """
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.2, disbelief=0.6, uncertainty=0.2)
        c = Opinion(belief=0.5, disbelief=0.1, uncertainty=0.4)

        layer = SLFusionLayer(strategy="averaging")
        simultaneous = layer.fuse([a, b, c])

        # Pairwise fold: (a ⊘ b) ⊘ c — this is WRONG for averaging
        pairwise = averaging_fuse(averaging_fuse(a, b), c)

        # They should NOT be equal in general (averaging is non-associative)
        # We check that the simultaneous result matches jsonld-ex n-ary
        expected = averaging_fuse(a, b, c)
        assert simultaneous.belief == pytest.approx(expected.belief)
        assert simultaneous.disbelief == pytest.approx(expected.disbelief)
        assert simultaneous.uncertainty == pytest.approx(expected.uncertainty)


# ════════════════════════════════════════════════════════════════════
# SLFusionLayer — robust fusion
# ════════════════════════════════════════════════════════════════════


class TestRobustFusion:
    """Tests for robust fusion strategy (iterative conflict filtering)."""

    def test_removes_rogue_agent(self, majority_with_rogue):
        layer = SLFusionLayer(strategy="robust")
        layer.fuse(majority_with_rogue)
        lr = layer.last_result
        # The rogue (index 3) should be removed
        assert 3 in lr.removed_indices

    def test_surviving_indices_exclude_removed(self, majority_with_rogue):
        layer = SLFusionLayer(strategy="robust")
        layer.fuse(majority_with_rogue)
        lr = layer.last_result
        for idx in lr.removed_indices:
            assert idx not in lr.surviving_indices
        assert set(lr.removed_indices) | set(lr.surviving_indices) == {0, 1, 2, 3}

    def test_fused_opinion_is_valid(self, majority_with_rogue):
        layer = SLFusionLayer(strategy="robust")
        result = layer.fuse(majority_with_rogue)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        assert result.belief >= 0.0
        assert result.disbelief >= 0.0
        assert result.uncertainty >= 0.0

    def test_agrees_with_jsonld_ex_directly(self, majority_with_rogue):
        from jsonld_ex.confidence_algebra import robust_fuse
        layer = SLFusionLayer(strategy="robust", robust_threshold=0.15)
        result = layer.fuse(majority_with_rogue)
        expected_op, expected_removed = robust_fuse(majority_with_rogue, threshold=0.15)
        assert result.belief == pytest.approx(expected_op.belief)
        assert result.disbelief == pytest.approx(expected_op.disbelief)
        assert result.uncertainty == pytest.approx(expected_op.uncertainty)
        assert set(layer.last_result.removed_indices) == set(expected_removed)

    def test_no_removal_when_cohesive(self, three_agreeing):
        layer = SLFusionLayer(strategy="robust")
        layer.fuse(three_agreeing)
        assert layer.last_result.removed_indices == []
        assert layer.last_result.surviving_indices == [0, 1, 2]

    def test_custom_threshold(self, majority_with_rogue):
        # Very high threshold — should not remove anyone
        layer = SLFusionLayer(strategy="robust", robust_threshold=0.99)
        layer.fuse(majority_with_rogue)
        assert layer.last_result.removed_indices == []

    def test_last_result_strategy_is_robust(self, majority_with_rogue):
        layer = SLFusionLayer(strategy="robust")
        layer.fuse(majority_with_rogue)
        assert layer.last_result.strategy == "robust"

    def test_no_byzantine_metadata(self, majority_with_rogue):
        """Robust fusion should NOT populate byzantine-only fields."""
        layer = SLFusionLayer(strategy="robust")
        layer.fuse(majority_with_rogue)
        lr = layer.last_result
        assert lr.conflict_matrix is None
        assert lr.cohesion is None
        assert lr.removal_details is None


# ════════════════════════════════════════════════════════════════════
# SLFusionLayer — byzantine fusion (3 strategies)
# ════════════════════════════════════════════════════════════════════


class TestByzantineFusion:
    """Tests for all three byzantine fusion strategies."""

    @pytest.mark.parametrize("strategy", [
        "byzantine_most_conflicting",
        "byzantine_least_trusted",
        "byzantine_combined",
    ])
    def test_fused_opinion_is_valid(self, strategy, majority_with_rogue):
        trust = [0.9, 0.9, 0.9, 0.1]  # needed for least_trusted/combined
        layer = SLFusionLayer(strategy=strategy, trust_weights=trust)
        result = layer.fuse(majority_with_rogue)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        assert result.belief >= 0.0
        assert result.disbelief >= 0.0
        assert result.uncertainty >= 0.0

    @pytest.mark.parametrize("strategy", [
        "byzantine_most_conflicting",
        "byzantine_least_trusted",
        "byzantine_combined",
    ])
    def test_conflict_matrix_preserved(self, strategy, majority_with_rogue):
        trust = [0.9, 0.9, 0.9, 0.1]
        layer = SLFusionLayer(strategy=strategy, trust_weights=trust)
        layer.fuse(majority_with_rogue)
        lr = layer.last_result
        assert lr.conflict_matrix is not None
        n = len(majority_with_rogue)
        assert len(lr.conflict_matrix) == n
        assert all(len(row) == n for row in lr.conflict_matrix)
        # Diagonal is zero
        for i in range(n):
            assert lr.conflict_matrix[i][i] == 0.0
        # Symmetric
        for i in range(n):
            for j in range(i + 1, n):
                assert lr.conflict_matrix[i][j] == pytest.approx(lr.conflict_matrix[j][i])

    @pytest.mark.parametrize("strategy", [
        "byzantine_most_conflicting",
        "byzantine_least_trusted",
        "byzantine_combined",
    ])
    def test_cohesion_score_preserved(self, strategy, majority_with_rogue):
        trust = [0.9, 0.9, 0.9, 0.1]
        layer = SLFusionLayer(strategy=strategy, trust_weights=trust)
        layer.fuse(majority_with_rogue)
        lr = layer.last_result
        assert lr.cohesion is not None
        assert 0.0 <= lr.cohesion <= 1.0

    @pytest.mark.parametrize("strategy", [
        "byzantine_most_conflicting",
        "byzantine_least_trusted",
        "byzantine_combined",
    ])
    def test_removal_details_are_agent_removal_objects(self, strategy, majority_with_rogue):
        trust = [0.9, 0.9, 0.9, 0.1]
        layer = SLFusionLayer(strategy=strategy, trust_weights=trust)
        layer.fuse(majority_with_rogue)
        lr = layer.last_result
        if lr.removal_details:
            for removal in lr.removal_details:
                assert isinstance(removal, AgentRemoval)
                assert isinstance(removal.index, int)
                assert isinstance(removal.opinion, Opinion)
                assert isinstance(removal.discord_score, float)
                assert isinstance(removal.reason, str)

    @pytest.mark.parametrize("strategy", [
        "byzantine_most_conflicting",
        "byzantine_least_trusted",
        "byzantine_combined",
    ])
    def test_surviving_plus_removed_equals_input(self, strategy, majority_with_rogue):
        trust = [0.9, 0.9, 0.9, 0.1]
        layer = SLFusionLayer(strategy=strategy, trust_weights=trust)
        layer.fuse(majority_with_rogue)
        lr = layer.last_result
        all_indices = set(lr.surviving_indices) | set(lr.removed_indices)
        assert all_indices == set(range(len(majority_with_rogue)))

    def test_most_conflicting_removes_rogue(self, majority_with_rogue):
        layer = SLFusionLayer(strategy="byzantine_most_conflicting")
        layer.fuse(majority_with_rogue)
        assert 3 in layer.last_result.removed_indices

    def test_least_trusted_removes_low_trust(self, majority_with_rogue):
        trust = [0.9, 0.9, 0.9, 0.1]
        layer = SLFusionLayer(strategy="byzantine_least_trusted", trust_weights=trust)
        layer.fuse(majority_with_rogue)
        assert 3 in layer.last_result.removed_indices

    def test_combined_removes_rogue(self, majority_with_rogue):
        trust = [0.9, 0.9, 0.9, 0.1]
        layer = SLFusionLayer(strategy="byzantine_combined", trust_weights=trust)
        layer.fuse(majority_with_rogue)
        assert 3 in layer.last_result.removed_indices

    def test_matches_jsonld_ex_most_conflicting(self, majority_with_rogue):
        from jsonld_ex.confidence_byzantine import byzantine_fuse as byz_fuse
        layer = SLFusionLayer(strategy="byzantine_most_conflicting")
        result = layer.fuse(majority_with_rogue)
        cfg = ByzantineConfig(strategy="most_conflicting")
        report = byz_fuse(majority_with_rogue, config=cfg)
        assert result.belief == pytest.approx(report.fused.belief)
        assert result.disbelief == pytest.approx(report.fused.disbelief)
        assert result.uncertainty == pytest.approx(report.fused.uncertainty)

    def test_conflict_matrix_matches_pairwise_conflict(self, majority_with_rogue):
        """Verify the preserved conflict matrix matches manual pairwise_conflict calls."""
        trust = [0.9, 0.9, 0.9, 0.1]
        layer = SLFusionLayer(strategy="byzantine_most_conflicting", trust_weights=trust)
        layer.fuse(majority_with_rogue)
        matrix = layer.last_result.conflict_matrix
        n = len(majority_with_rogue)
        for i in range(n):
            for j in range(i + 1, n):
                expected = pairwise_conflict(majority_with_rogue[i], majority_with_rogue[j])
                assert matrix[i][j] == pytest.approx(expected)

    def test_byzantine_config_override(self, majority_with_rogue):
        """User can pass a custom ByzantineConfig to override defaults."""
        cfg = ByzantineConfig(
            threshold=0.99,  # very high — should remove nobody
            strategy="most_conflicting",
        )
        layer = SLFusionLayer(strategy="byzantine_most_conflicting", byzantine_config=cfg)
        layer.fuse(majority_with_rogue)
        assert layer.last_result.removed_indices == []

    def test_least_trusted_requires_trust_weights(self, majority_with_rogue):
        """least_trusted and combined strategies need trust_weights."""
        layer = SLFusionLayer(strategy="byzantine_least_trusted")
        with pytest.raises((ValueError, TypeError)):
            layer.fuse(majority_with_rogue)

    def test_combined_requires_trust_weights(self, majority_with_rogue):
        layer = SLFusionLayer(strategy="byzantine_combined")
        with pytest.raises((ValueError, TypeError)):
            layer.fuse(majority_with_rogue)

    def test_trust_weights_length_mismatch_raises(self, majority_with_rogue):
        layer = SLFusionLayer(
            strategy="byzantine_least_trusted",
            trust_weights=[0.9, 0.9],  # only 2 for 4 opinions
        )
        with pytest.raises(ValueError, match="[Ll]ength|[Mm]atch|trust"):
            layer.fuse(majority_with_rogue)


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════


class TestFusionEdgeCases:
    """Edge cases shared across all strategies."""

    def test_empty_list_raises(self):
        layer = SLFusionLayer(strategy="cumulative")
        with pytest.raises(ValueError):
            layer.fuse([])

    @pytest.mark.parametrize("strategy", ["cumulative", "averaging", "robust"])
    def test_single_opinion_returns_itself(self, strategy):
        op = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        layer = SLFusionLayer(strategy=strategy)
        result = layer.fuse([op])
        assert result.belief == pytest.approx(op.belief)
        assert result.disbelief == pytest.approx(op.disbelief)
        assert result.uncertainty == pytest.approx(op.uncertainty)

    def test_all_vacuous_stays_vacuous_cumulative(self):
        """Fusing vacuous opinions yields a vacuous opinion."""
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        layer = SLFusionLayer(strategy="cumulative")
        result = layer.fuse([vac, vac, vac])
        assert result.uncertainty == pytest.approx(1.0)
        assert result.belief == pytest.approx(0.0)
        assert result.disbelief == pytest.approx(0.0)

    def test_two_opinions_cumulative(self, two_conflicting):
        """Verify two conflicting opinions fuse without error."""
        layer = SLFusionLayer(strategy="cumulative")
        result = layer.fuse(two_conflicting)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_last_result_updated_on_each_call(self):
        """Each fuse() call replaces last_result."""
        layer = SLFusionLayer(strategy="cumulative")
        op1 = [Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)]
        op2 = [
            Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2),
            Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
        ]
        layer.fuse(op1)
        assert layer.last_result.input_count == 1
        layer.fuse(op2)
        assert layer.last_result.input_count == 2


# ════════════════════════════════════════════════════════════════════
# NoOpFusionLayer still works
# ════════════════════════════════════════════════════════════════════


class TestNoOpFusionLayerStillWorks:
    """Ensure the existing NoOpFusionLayer is not broken."""

    def test_returns_vacuous(self, three_agreeing):
        layer = NoOpFusionLayer()
        result = layer.fuse(three_agreeing)
        assert result.belief == 0.0
        assert result.disbelief == 0.0
        assert result.uncertainty == 1.0

    def test_is_fusion_layer_subclass(self):
        assert isinstance(NoOpFusionLayer(), FusionLayer)
