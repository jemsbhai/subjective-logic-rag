"""SL-RAG Pipeline — main orchestrator.

Wires together all pipeline layers:
    doc_opinions → Trust → Temporal → Conflict Detection → Fusion
    → Deduction → Decision → (Generation)

Each layer is pluggable via dependency injection. Use with_noop_layers()
for a skeleton pipeline where every layer is a passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from jsonld_ex.confidence_algebra import Opinion

from xrag.pipeline.conflict_layer import ConflictLayer, NoOpConflictLayer, SLConflictLayer
from xrag.pipeline.decision_layer import DecisionLayer, NoOpDecisionLayer, SLDecisionLayer
from xrag.pipeline.deduction_layer import DeductionLayer, NoOpDeductionLayer
from xrag.pipeline.fusion_layer import FusionLayer, FusionStrategy, NoOpFusionLayer, SLFusionLayer
from xrag.pipeline.temporal_layer import NoOpTemporalLayer, SLTemporalLayer, TemporalLayer
from xrag.pipeline.trust_layer import NoOpTrustLayer, SLTrustLayer, TrustLayer


@dataclass
class PipelineResult:
    """Final output of the SL-RAG pipeline for a single query.

    Attributes:
        query: The original query string.
        answer: Generated answer, or None if abstained/flagged.
        decision: One of "generate", "abstain", "flag_conflict".
        fused_opinion: Aggregate opinion after fusion + deduction.
        doc_opinions: Per-document opinions after trust + temporal layers.
        conflict_detected: Whether conflict was detected.
        metadata: Intermediate layer outputs for debugging/analysis.
    """

    query: str
    answer: Optional[str]
    decision: str
    fused_opinion: Optional[Opinion]
    doc_opinions: list[Opinion]
    conflict_detected: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class SLRAGPipeline:
    """Main SL-RAG pipeline orchestrator.

    Processes per-document opinions through a sequence of SL layers
    and produces a final decision (generate / abstain / flag_conflict).

    Note: This pipeline operates on *already-estimated* document opinions.
    Opinion estimation (NLI or LLM-judge) and retrieval happen upstream.
    Generation happens downstream. This keeps the pipeline focused on
    the SL uncertainty algebra — the core contribution of the paper.

    Args:
        trust_layer: Applies source reliability trust discounting.
        temporal_layer: Applies temporal decay for stale evidence.
        conflict_layer: Detects contradictions among documents.
        fusion_layer: Fuses multiple document opinions into one.
        deduction_layer: Chains multi-hop reasoning via deduce().
        decision_layer: Decides generate / abstain / flag_conflict.
    """

    def __init__(
        self,
        trust_layer: TrustLayer,
        temporal_layer: TemporalLayer,
        conflict_layer: ConflictLayer,
        fusion_layer: FusionLayer,
        deduction_layer: DeductionLayer,
        decision_layer: DecisionLayer,
    ) -> None:
        self.trust_layer = trust_layer
        self.temporal_layer = temporal_layer
        self.conflict_layer = conflict_layer
        self.fusion_layer = fusion_layer
        self.deduction_layer = deduction_layer
        self.decision_layer = decision_layer

    @classmethod
    def with_noop_layers(cls) -> SLRAGPipeline:
        """Create a pipeline with all no-op layers (skeleton for testing)."""
        return cls(
            trust_layer=NoOpTrustLayer(),
            temporal_layer=NoOpTemporalLayer(),
            conflict_layer=NoOpConflictLayer(),
            fusion_layer=NoOpFusionLayer(),
            deduction_layer=NoOpDeductionLayer(),
            decision_layer=NoOpDecisionLayer(),
        )

    @classmethod
    def with_default_layers(
        cls,
        fusion_strategy: FusionStrategy = "cumulative",
        default_trust: Optional[Opinion] = None,
        half_life: float = 168.0,
        conflict_threshold: float = 0.3,
        tau_abstain: float = 0.7,
        tau_conflict: float = 0.5,
    ) -> SLRAGPipeline:
        """Create a pipeline with real SL layers and sensible defaults.

        Args:
            fusion_strategy: Fusion method (default: cumulative).
            default_trust: Default trust opinion for all docs.
                Default: high trust (b=0.9, d=0.0, u=0.1) for Wikipedia.
            half_life: Temporal decay half-life in hours (default: 168 = 1 week).
            conflict_threshold: Pairwise conflict threshold (default: 0.3).
            tau_abstain: Uncertainty threshold for abstention (default: 0.7).
            tau_conflict: Conflict score threshold for flagging (default: 0.5).

        Returns:
            SLRAGPipeline with real SL layers wired.
        """
        if default_trust is None:
            default_trust = Opinion(
                belief=0.9, disbelief=0.0, uncertainty=0.1, base_rate=0.5,
            )

        return cls(
            trust_layer=SLTrustLayer(default_trust=default_trust),
            temporal_layer=SLTemporalLayer(half_life=half_life),
            conflict_layer=SLConflictLayer(threshold=conflict_threshold),
            fusion_layer=SLFusionLayer(strategy=fusion_strategy),
            deduction_layer=NoOpDeductionLayer(),
            decision_layer=SLDecisionLayer(
                tau_abstain=tau_abstain, tau_conflict=tau_conflict,
            ),
        )

    def run(
        self,
        query: str,
        doc_opinions: list[Opinion],
        sub_questions: Optional[list[str]] = None,
    ) -> PipelineResult:
        """Run the full pipeline for a single query.

        Args:
            query: The question string.
            doc_opinions: Per-document SL opinions from the estimator.
            sub_questions: Optional sub-question decomposition for multi-hop.

        Returns:
            PipelineResult with decision, fused opinion, and metadata.
        """
        # 1. Trust discount
        trusted = self.trust_layer.apply(doc_opinions)

        # 2. Temporal decay
        decayed = self.temporal_layer.apply(trusted)

        # 3. Conflict detection
        conflict_result = self.conflict_layer.detect(decayed)

        # 4. Evidence fusion
        fused = self.fusion_layer.fuse(decayed)

        # 5. Multi-hop deduction
        deduced = self.deduction_layer.deduce(fused, sub_questions=sub_questions)

        # 6. Decision
        decision_result = self.decision_layer.decide(deduced, conflict_result)

        return PipelineResult(
            query=query,
            answer=None,  # Generation is external to this pipeline
            decision=decision_result.decision,
            fused_opinion=deduced,
            doc_opinions=decayed,
            conflict_detected=conflict_result.conflict_detected,
            metadata={
                "trust_opinions": trusted,
                "temporal_opinions": decayed,
                "conflict_result": conflict_result,
                "fused_opinion_pre_deduction": fused,
                "decision_result": decision_result,
            },
        )

    def run_batch(
        self,
        queries: list[str],
        doc_opinions_list: list[list[Opinion]],
        sub_questions_list: Optional[list[Optional[list[str]]]] = None,
    ) -> list[PipelineResult]:
        """Run the pipeline for a batch of queries.

        Args:
            queries: List of query strings.
            doc_opinions_list: List of per-document opinion lists, one per query.
            sub_questions_list: Optional list of sub-question lists, one per query.

        Returns:
            List of PipelineResult, one per query.

        Raises:
            ValueError: If queries and doc_opinions_list have different lengths.
        """
        if len(queries) != len(doc_opinions_list):
            raise ValueError(
                f"queries and doc_opinions_list must have same length, "
                f"got {len(queries)} and {len(doc_opinions_list)}"
            )

        if sub_questions_list is None:
            sub_questions_list = [None] * len(queries)

        return [
            self.run(q, ops, sqs)
            for q, ops, sqs in zip(queries, doc_opinions_list, sub_questions_list)
        ]
