"""Multi-hop deduction layer — chains reasoning across hops.

Uses SL deduce() from jsonld-ex to propagate uncertainty across
multi-hop inference chains.  This is the paper's core scientific
differentiator: no existing UQ method propagates uncertainty across
reasoning hops in a principled, algebraic way.

Design:
    - Linear chains: left-fold deduce() over hop_opinions (HotpotQA, most MuSiQue)
    - DAG deduction: topological sort + cumulative_fuse for multi-parent hops
    - Counterfactual strategies: vacuous (safest), adversarial (pessimistic),
      prior (neutral) — ablatable design choice
    - Full traceability: every intermediate deduced opinion and every
      deduce() call recorded in DeductionResult for Fig 5 and analysis

Key mathematical facts:
    - deduce(ω_x, ω_y|x, ω_y|¬x) generalizes the law of total probability
    - Vacuous antecedent does NOT yield vacuous result — it scales by base rate
    - Deeper chains with imperfect hops generally increase uncertainty
    - All-dogmatic chains collapse to deterministic reasoning

Paper 1 usage:
    - HotpotQA: 2-hop linear chains
    - MuSiQue: 2-4 hop chains (linear or DAG)
    - Ablation A7: NoOpDeductionLayer (flat fusion, no hop chaining)
    - Counterfactual ablation: vacuous vs adversarial vs prior
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Optional

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    deduce,
)


# ═══════════════════════════════════════════════════════════════════
# Counterfactual strategies
# ═══════════════════════════════════════════════════════════════════

CounterfactualFn = Callable[[Opinion], Opinion]


def vacuous_counterfactual(conditional: Opinion) -> Opinion:
    """Vacuous counterfactual: if premise wrong, we know nothing.

    This is the most conservative choice.  If hop 1 was wrong, we
    retrieved with the wrong entity and have zero evidence about hop 2.

    Returns Opinion(b=0, d=0, u=1) with the conditional's base rate.
    """
    return Opinion(
        belief=0.0,
        disbelief=0.0,
        uncertainty=1.0,
        base_rate=conditional.base_rate,
    )


def adversarial_counterfactual(conditional: Opinion) -> Opinion:
    """Adversarial counterfactual: if premise wrong, conclusion likely wrong.

    Maps the conditional's belief into disbelief: "all the evidence that
    supported the conclusion now works against it if the premise was wrong."

        b = 0
        d = conditional.belief
        u = conditional.disbelief + conditional.uncertainty

    Preserves b + d + u = 1 and the conditional's base rate.
    """
    return Opinion(
        belief=0.0,
        disbelief=conditional.belief,
        uncertainty=conditional.disbelief + conditional.uncertainty,
        base_rate=conditional.base_rate,
    )


def prior_counterfactual(conditional: Opinion) -> Opinion:
    """Prior counterfactual: if premise wrong, revert to base rate with high uncertainty.

    Uses Opinion.from_confidence with the base rate and high uncertainty (0.8),
    centering the opinion on the prior while expressing substantial ignorance.
    """
    return Opinion.from_confidence(
        confidence=conditional.base_rate,
        uncertainty=0.8,
        base_rate=conditional.base_rate,
    )


COUNTERFACTUAL_STRATEGIES: dict[str, CounterfactualFn] = {
    "vacuous": vacuous_counterfactual,
    "adversarial": adversarial_counterfactual,
    "prior": prior_counterfactual,
}


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HopInput:
    """A single hop in a multi-hop deduction graph.

    Attributes:
        hop_id:         Unique integer identifier for this hop.
        opinion:        The fused opinion for this hop's retrieval + estimation.
        sub_question:   The sub-question text (metadata, for traceability).
        parent_hop_ids: Tuple of hop_ids this hop depends on.
                        Empty for root hops (first hop or independent branches).
    """

    hop_id: int
    opinion: Opinion
    sub_question: str | None = None
    parent_hop_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DeductionStep:
    """Record of a single deduce() call in the chain.

    Attributes:
        hop_id:          Which hop this step resolves.
        antecedent:      ω_x — the accumulated opinion from parent hops.
        conditional:     ω_y|x — this hop's opinion (given premise was correct).
        counterfactual:  ω_y|¬x — opinion if premise was wrong.
        result:          The deduced opinion ω_y.
    """

    hop_id: int
    antecedent: Opinion
    conditional: Opinion
    counterfactual: Opinion
    result: Opinion


@dataclass(frozen=True)
class DeductionResult:
    """Complete record of a deduction pass.

    Attributes:
        final_opinion:          The final deduced opinion.
        hop_opinions:           Per-hop input opinions.
        steps:                  Ordered list of deduce() calls.
        intermediate_opinions:  Opinion after each hop (for Fig 5).
        chain_length:           Number of hops.
        is_single_hop:          Whether this was a passthrough.
        counterfactual_strategy: Name of the counterfactual strategy used.
        sub_questions:          Sub-question texts (metadata).
    """

    final_opinion: Opinion
    hop_opinions: list[Opinion]
    steps: list[DeductionStep]
    intermediate_opinions: list[Opinion]
    chain_length: int
    is_single_hop: bool
    counterfactual_strategy: str
    sub_questions: list[str] | None


# ═══════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════


class DeductionLayer(abc.ABC):
    """Abstract base for multi-hop deduction layers."""

    @abc.abstractmethod
    def deduce(
        self,
        fused_opinion: Opinion,
        sub_questions: Optional[list[str]] = None,
        **kwargs,
    ) -> Opinion:
        """Apply multi-hop deduction to the fused opinion.

        For single-hop questions, this is typically a passthrough.
        For multi-hop, it chains per-hop opinions via deduce().

        Args:
            fused_opinion: The fused opinion from the fusion layer.
            sub_questions: Optional list of sub-question strings.

        Returns:
            Final opinion after deduction.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# NoOp (unchanged — backward compatible)
# ═══════════════════════════════════════════════════════════════════


class NoOpDeductionLayer(DeductionLayer):
    """No-op deduction layer — passes the fused opinion through unchanged."""

    def deduce(
        self,
        fused_opinion: Opinion,
        sub_questions: Optional[list[str]] = None,
        **kwargs,
    ) -> Opinion:
        return fused_opinion


# ═══════════════════════════════════════════════════════════════════
# Real implementation
# ═══════════════════════════════════════════════════════════════════


class SLDeductionLayer(DeductionLayer):
    """Multi-hop deduction layer using SL deduce() operator.

    Chains per-hop opinions via Jøsang's deduction operator, which
    generalizes the law of total probability to the opinion domain.
    Uncertainty from earlier hops propagates forward — even if hop 2's
    retrieval was clean, the final answer carries hop 1's uncertainty.

    Supports two input modes:
        - hop_opinions (list[Opinion]): Linear chain, left-fold.
        - hop_inputs (list[HopInput]): Arbitrary DAG topology.

    Args:
        counterfactual_fn:           Callable to generate ω_y|¬x from ω_y|x.
        counterfactual_strategy_name: Human-readable name for traceability.
    """

    def __init__(
        self,
        counterfactual_fn: CounterfactualFn | None = None,
        counterfactual_strategy_name: str = "vacuous",
    ) -> None:
        self.counterfactual_fn: CounterfactualFn = (
            counterfactual_fn if counterfactual_fn is not None
            else vacuous_counterfactual
        )
        self.counterfactual_strategy_name = counterfactual_strategy_name
        self.last_result: DeductionResult | None = None

    def deduce(
        self,
        fused_opinion: Opinion,
        sub_questions: Optional[list[str]] = None,
        hop_opinions: list[Opinion] | None = None,
        hop_inputs: list[HopInput] | None = None,
        **kwargs,
    ) -> Opinion:
        """Apply multi-hop deduction.

        Args:
            fused_opinion: The flat-fused opinion (used as passthrough
                           when no hop structure is provided).
            sub_questions: Sub-question texts for traceability.
            hop_opinions: Per-hop opinions for linear chain deduction.
            hop_inputs: Per-hop inputs with DAG topology.

        Returns:
            Final deduced opinion.

        Raises:
            ValueError: If both hop_opinions and hop_inputs are provided.
        """
        if hop_opinions is not None and hop_inputs is not None:
            raise ValueError(
                "Both hop_opinions and hop_inputs provided — "
                "these are mutually exclusive. Use hop_opinions for "
                "linear chains and hop_inputs for DAG topology."
            )

        if hop_inputs is not None:
            return self._deduce_dag(hop_inputs, sub_questions)
        elif hop_opinions is not None and len(hop_opinions) >= 2:
            return self._deduce_linear(hop_opinions, sub_questions)
        else:
            # Single-hop passthrough
            return self._passthrough(fused_opinion, hop_opinions, sub_questions)

    # ── Passthrough ───────────────────────────────────────────────

    def _passthrough(
        self,
        fused_opinion: Opinion,
        hop_opinions: list[Opinion] | None,
        sub_questions: list[str] | None,
    ) -> Opinion:
        """Single-hop: no deduction needed."""
        hop_ops = hop_opinions if hop_opinions else [fused_opinion]
        self.last_result = DeductionResult(
            final_opinion=fused_opinion,
            hop_opinions=hop_ops,
            steps=[],
            intermediate_opinions=[fused_opinion],
            chain_length=len(hop_ops),
            is_single_hop=True,
            counterfactual_strategy=self.counterfactual_strategy_name,
            sub_questions=sub_questions,
        )
        return fused_opinion

    # ── Linear chain ──────────────────────────────────────────────

    def _deduce_linear(
        self,
        hop_opinions: list[Opinion],
        sub_questions: list[str] | None,
    ) -> Opinion:
        """Left-fold deduce() over a linear chain of hop opinions."""
        steps: list[DeductionStep] = []
        intermediates: list[Opinion] = [hop_opinions[0]]
        acc = hop_opinions[0]

        for i in range(1, len(hop_opinions)):
            conditional = hop_opinions[i]
            cf = self.counterfactual_fn(conditional)
            result = deduce(acc, conditional, cf)
            steps.append(DeductionStep(
                hop_id=i,
                antecedent=acc,
                conditional=conditional,
                counterfactual=cf,
                result=result,
            ))
            acc = result
            intermediates.append(acc)

        self.last_result = DeductionResult(
            final_opinion=acc,
            hop_opinions=list(hop_opinions),
            steps=steps,
            intermediate_opinions=intermediates,
            chain_length=len(hop_opinions),
            is_single_hop=False,
            counterfactual_strategy=self.counterfactual_strategy_name,
            sub_questions=sub_questions,
        )
        return acc

    # ── DAG deduction ─────────────────────────────────────────────

    def _deduce_dag(
        self,
        hop_inputs: list[HopInput],
        sub_questions: list[str] | None,
    ) -> Opinion:
        """DAG deduction with topological ordering.

        Algorithm:
            1. Topological sort of hops by dependency.
            2. For root hops (no parents): deduced opinion = raw opinion.
            3. For non-root hops:
               a. Collect deduced opinions of all parents.
               b. If multiple parents: cumulative_fuse into compound antecedent.
               c. deduce(antecedent, hop_opinion, counterfactual(hop_opinion)).
            4. Final result = deduced opinion of the sink hop (last in topo order).
        """
        # Build lookup
        hop_map: dict[int, HopInput] = {h.hop_id: h for h in hop_inputs}
        deduced: dict[int, Opinion] = {}
        steps: list[DeductionStep] = []
        intermediates: list[Opinion] = []

        # Topological sort (Kahn's algorithm)
        order = self._topological_sort(hop_inputs)

        for hop_id in order:
            hop = hop_map[hop_id]

            if not hop.parent_hop_ids:
                # Root hop — use raw opinion
                deduced[hop_id] = hop.opinion
                intermediates.append(hop.opinion)
            else:
                # Non-root: gather parent deduced opinions
                parent_opinions = [deduced[pid] for pid in hop.parent_hop_ids]

                # Fuse multiple parents into compound antecedent
                if len(parent_opinions) == 1:
                    antecedent = parent_opinions[0]
                else:
                    antecedent = cumulative_fuse(*parent_opinions)

                # Deduce
                conditional = hop.opinion
                cf = self.counterfactual_fn(conditional)
                result = deduce(antecedent, conditional, cf)

                steps.append(DeductionStep(
                    hop_id=hop_id,
                    antecedent=antecedent,
                    conditional=conditional,
                    counterfactual=cf,
                    result=result,
                ))
                deduced[hop_id] = result
                intermediates.append(result)

        # Final opinion is the last hop in topological order
        final = deduced[order[-1]]

        self.last_result = DeductionResult(
            final_opinion=final,
            hop_opinions=[h.opinion for h in hop_inputs],
            steps=steps,
            intermediate_opinions=intermediates,
            chain_length=len(hop_inputs),
            is_single_hop=False,
            counterfactual_strategy=self.counterfactual_strategy_name,
            sub_questions=sub_questions,
        )
        return final

    @staticmethod
    def _topological_sort(hop_inputs: list[HopInput]) -> list[int]:
        """Kahn's algorithm for topological ordering of hops.

        Returns hop_ids in dependency order (parents before children).

        Raises:
            ValueError: If the graph contains a cycle.
        """
        hop_map = {h.hop_id: h for h in hop_inputs}
        all_ids = set(hop_map.keys())

        # Compute in-degree
        in_degree: dict[int, int] = {hid: 0 for hid in all_ids}
        children: dict[int, list[int]] = {hid: [] for hid in all_ids}

        for hop in hop_inputs:
            for pid in hop.parent_hop_ids:
                children[pid].append(hop.hop_id)
                in_degree[hop.hop_id] += 1

        # Queue of zero in-degree nodes
        queue = [hid for hid in all_ids if in_degree[hid] == 0]
        # Sort for deterministic order among peers
        queue.sort()

        order: list[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in sorted(children[node]):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(all_ids):
            raise ValueError(
                "Cycle detected in hop dependency graph. "
                f"Processed {len(order)} of {len(all_ids)} hops."
            )

        return order
