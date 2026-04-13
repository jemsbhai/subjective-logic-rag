"""Corruption suites for UQ-RAGBench — controlled perturbations.

Provides 6 atomic corruption types that map to specific SL operators
and hallucination modes, plus a composable pipeline for realistic
multi-corruption scenarios.

Corruption types:
    1. Distractor injection   → tests fusion (belief vs uncertainty)
    2. Contradiction injection → tests conflict detection (pairwise_conflict)
    3. Evidence removal        → tests abstention (u > τ_abstain)
    4. Timestamp backdating    → tests temporal decay (decay_opinion)
    5. Adversarial injection   → tests Byzantine fusion (robust_fuse)
    6. Prompt injection        → tests trust discount + decision layer

CRITICAL DESIGN PRINCIPLE: The pipeline receives a plain RetrievalResult
with NO corruption labels. CorruptedResult metadata is for evaluation
ONLY — computing conflict detection P/R, abstention quality, etc.

Corruption levels are continuous [0.0, 1.0]. The paper reports curves
(corruption fraction on x-axis, metric on y-axis) with specific points
at 0%, 25%, 50%, 75% for tables.

Target selection policies (configurable, all three reported in paper):
    - random:         realistic deployment scenario (default)
    - gold_first:     hardest — removes the answer first
    - non_gold_first: easiest — preserves the answer

All corruptions are seeded for reproducibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from xrag.benchmarks.retriever import RetrievedPassage, RetrievalResult


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════

CorruptionType = Literal[
    "distractor",
    "contradiction",
    "evidence_removal",
    "timestamp_backdate",
    "adversarial",
    "prompt_injection",
]

CORRUPTION_TYPES: tuple[CorruptionType, ...] = (
    "distractor",
    "contradiction",
    "evidence_removal",
    "timestamp_backdate",
    "adversarial",
    "prompt_injection",
)

TargetPolicy = Literal["random", "gold_first", "non_gold_first"]

ContradictionFn = Callable[[RetrievedPassage], RetrievedPassage]


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CorruptionConfig:
    """Configuration for a single corruption operation.

    Attributes:
        corruption_type:  Which corruption to apply.
        fraction:         Fraction of passages to corrupt [0.0, 1.0].
        target_policy:    How to select which passages to corrupt.
        seed:             RNG seed for reproducibility.
    """

    corruption_type: CorruptionType
    fraction: float
    target_policy: TargetPolicy = "random"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.fraction < 0.0 or self.fraction > 1.0:
            raise ValueError(
                f"fraction must be in [0.0, 1.0], got {self.fraction}"
            )


@dataclass(frozen=True)
class CorruptionEntry:
    """Record of a single passage corruption — for evaluation ONLY.

    Attributes:
        index:            Position in the corrupted result's passage list.
        corruption_type:  Which corruption was applied.
        original_passage: The passage that was replaced/removed (None if N/A).
        injected_passage: The passage that was injected (None for removal).
        metadata:         Extra info (e.g., elapsed_hours for backdating).
    """

    index: int
    corruption_type: str
    original_passage: RetrievedPassage | None = None
    injected_passage: RetrievedPassage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorruptedResult:
    """Result of applying corruption to a RetrievalResult.

    The retrieval_result field is what the pipeline sees — a plain
    RetrievalResult with NO corruption labels. The corruption_log
    and other fields are for evaluation ONLY.

    Attributes:
        retrieval_result:  The corrupted result (pipeline-facing, no labels).
        original_result:   The original uncorrupted result.
        corruption_log:    Per-passage corruption records (evaluation only).
        elapsed_times:     Per-passage elapsed times for temporal layer
                           (None for non-backdated passages).
    """

    retrieval_result: RetrievalResult
    original_result: RetrievalResult
    corruption_log: list[CorruptionEntry]
    elapsed_times: list[float | None] | None = None


# ═══════════════════════════════════════════════════════════════════
# Target selection
# ═══════════════════════════════════════════════════════════════════


def _select_targets(
    passages: list[RetrievedPassage],
    n: int,
    policy: TargetPolicy,
    rng: random.Random,
) -> list[int]:
    """Select which passage indices to corrupt.

    Args:
        passages: The passage list.
        n:        Number of passages to select.
        policy:   Selection policy.
        rng:      Seeded RNG.

    Returns:
        List of indices to corrupt, in no particular order.
    """
    if n == 0 or len(passages) == 0:
        return []

    n = min(n, len(passages))

    gold_indices = [i for i, p in enumerate(passages) if p.has_answer]
    non_gold_indices = [i for i, p in enumerate(passages) if not p.has_answer]

    if policy == "gold_first":
        # Take gold first, then overflow to non-gold
        priority = list(gold_indices)
        rng.shuffle(priority)
        overflow = list(non_gold_indices)
        rng.shuffle(overflow)
        pool = priority + overflow
    elif policy == "non_gold_first":
        # Take non-gold first, then overflow to gold
        priority = list(non_gold_indices)
        rng.shuffle(priority)
        overflow = list(gold_indices)
        rng.shuffle(overflow)
        pool = priority + overflow
    else:
        # Random
        pool = list(range(len(passages)))
        rng.shuffle(pool)

    return pool[:n]


def _compute_n(total: int, fraction: float) -> int:
    """Compute number of passages to corrupt."""
    return round(fraction * total)


# ═══════════════════════════════════════════════════════════════════
# 1. Distractor injection
# ═══════════════════════════════════════════════════════════════════


def inject_distractors(
    result: RetrievalResult,
    pool: list[RetrievedPassage],
    config: CorruptionConfig,
) -> CorruptedResult:
    """Replace selected passages with topically similar but irrelevant ones.

    Tests fusion (belief vs uncertainty separation).

    Args:
        result: Original retrieval result.
        pool:   Distractor passages to draw from.
        config: Corruption configuration.

    Returns:
        CorruptedResult with distractor passages injected.

    Raises:
        ValueError: If pool is too small for the requested corruption.
    """
    n = _compute_n(len(result.passages), config.fraction)
    if n > len(pool):
        raise ValueError(
            f"Insufficient pool: need {n} distractors but pool has {len(pool)}."
        )

    rng = random.Random(config.seed)
    targets = _select_targets(result.passages, n, config.target_policy, rng)

    # Sample from pool without replacement
    pool_sample = rng.sample(pool, n)

    passages = list(result.passages)
    log: list[CorruptionEntry] = []

    for target_idx, replacement in zip(sorted(targets), pool_sample):
        log.append(CorruptionEntry(
            index=target_idx,
            corruption_type="distractor",
            original_passage=passages[target_idx],
            injected_passage=replacement,
        ))
        passages[target_idx] = replacement

    return CorruptedResult(
        retrieval_result=RetrievalResult(query=result.query, passages=passages),
        original_result=result,
        corruption_log=log,
    )


# ═══════════════════════════════════════════════════════════════════
# 2. Contradiction injection
# ═══════════════════════════════════════════════════════════════════


def inject_contradictions(
    result: RetrievalResult,
    contradiction_fn: ContradictionFn,
    config: CorruptionConfig,
) -> CorruptedResult:
    """Replace selected passages with counterfactual versions.

    Tests conflict detection (pairwise_conflict).

    Args:
        result:           Original retrieval result.
        contradiction_fn: Callable that takes a passage and returns
                          its counterfactual version.
        config:           Corruption configuration.

    Returns:
        CorruptedResult with contradiction passages injected.
    """
    n = _compute_n(len(result.passages), config.fraction)
    rng = random.Random(config.seed)
    targets = _select_targets(result.passages, n, config.target_policy, rng)

    passages = list(result.passages)
    log: list[CorruptionEntry] = []

    for target_idx in sorted(targets):
        original = passages[target_idx]
        counterfactual = contradiction_fn(original)
        log.append(CorruptionEntry(
            index=target_idx,
            corruption_type="contradiction",
            original_passage=original,
            injected_passage=counterfactual,
        ))
        passages[target_idx] = counterfactual

    return CorruptedResult(
        retrieval_result=RetrievalResult(query=result.query, passages=passages),
        original_result=result,
        corruption_log=log,
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Evidence removal
# ═══════════════════════════════════════════════════════════════════


def remove_evidence(
    result: RetrievalResult,
    config: CorruptionConfig,
) -> CorruptedResult:
    """Remove selected passages entirely.

    Tests abstention (u > τ_abstain). Unlike other corruptions,
    this reduces the passage count.

    Args:
        result: Original retrieval result.
        config: Corruption configuration.

    Returns:
        CorruptedResult with passages removed.
    """
    n = _compute_n(len(result.passages), config.fraction)
    rng = random.Random(config.seed)
    targets = set(_select_targets(result.passages, n, config.target_policy, rng))

    passages: list[RetrievedPassage] = []
    log: list[CorruptionEntry] = []

    for i, p in enumerate(result.passages):
        if i in targets:
            log.append(CorruptionEntry(
                index=i,
                corruption_type="evidence_removal",
                original_passage=p,
                injected_passage=None,
            ))
        else:
            passages.append(p)

    return CorruptedResult(
        retrieval_result=RetrievalResult(query=result.query, passages=passages),
        original_result=result,
        corruption_log=log,
    )


# ═══════════════════════════════════════════════════════════════════
# 4. Timestamp backdating
# ═══════════════════════════════════════════════════════════════════


def backdate_timestamps(
    result: RetrievalResult,
    max_age_hours: float,
    config: CorruptionConfig,
) -> CorruptedResult:
    """Assign old timestamps to selected passages.

    Tests temporal decay (decay_opinion). Does NOT remove or replace
    passages — only adds temporal metadata. The elapsed_times field
    on the result provides per-passage elapsed times for the
    temporal layer (None for non-backdated passages).

    Args:
        result:        Original retrieval result.
        max_age_hours: Maximum age in hours for backdated passages.
        config:        Corruption configuration.

    Returns:
        CorruptedResult with elapsed_times populated.
    """
    n = _compute_n(len(result.passages), config.fraction)
    rng = random.Random(config.seed)
    targets = set(_select_targets(result.passages, n, config.target_policy, rng))

    elapsed_times: list[float | None] = [None] * len(result.passages)
    log: list[CorruptionEntry] = []

    for i in range(len(result.passages)):
        if i in targets:
            age = rng.uniform(1.0, max_age_hours)
            elapsed_times[i] = age
            log.append(CorruptionEntry(
                index=i,
                corruption_type="timestamp_backdate",
                original_passage=result.passages[i],
                injected_passage=result.passages[i],  # passage unchanged
                metadata={"elapsed_hours": age},
            ))

    return CorruptedResult(
        retrieval_result=RetrievalResult(
            query=result.query, passages=list(result.passages),
        ),
        original_result=result,
        corruption_log=log,
        elapsed_times=elapsed_times,
    )


# ═══════════════════════════════════════════════════════════════════
# 5. Adversarial injection
# ═══════════════════════════════════════════════════════════════════


def inject_adversarial(
    result: RetrievalResult,
    pool: list[RetrievedPassage],
    config: CorruptionConfig,
) -> CorruptedResult:
    """Replace selected passages with plausible-but-wrong content.

    Tests Byzantine fusion (robust_fuse, byzantine_fuse).

    Args:
        result: Original retrieval result.
        pool:   Adversarial passages to draw from.
        config: Corruption configuration.

    Returns:
        CorruptedResult with adversarial passages injected.

    Raises:
        ValueError: If pool is too small.
    """
    n = _compute_n(len(result.passages), config.fraction)
    if n > len(pool):
        raise ValueError(
            f"Insufficient pool: need {n} adversarial passages but pool has {len(pool)}."
        )

    rng = random.Random(config.seed)
    targets = _select_targets(result.passages, n, config.target_policy, rng)
    pool_sample = rng.sample(pool, n)

    passages = list(result.passages)
    log: list[CorruptionEntry] = []

    for target_idx, replacement in zip(sorted(targets), pool_sample):
        log.append(CorruptionEntry(
            index=target_idx,
            corruption_type="adversarial",
            original_passage=passages[target_idx],
            injected_passage=replacement,
        ))
        passages[target_idx] = replacement

    return CorruptedResult(
        retrieval_result=RetrievalResult(query=result.query, passages=passages),
        original_result=result,
        corruption_log=log,
    )


# ═══════════════════════════════════════════════════════════════════
# 6. Prompt injection
# ═══════════════════════════════════════════════════════════════════


def inject_prompt_injection(
    result: RetrievalResult,
    pool: list[RetrievedPassage],
    config: CorruptionConfig,
) -> CorruptedResult:
    """Replace selected passages with instruction-hijack text.

    Tests trust discount + decision layer. Models the threat where
    retrieved passages contain adversarial instructions that attempt
    to override the user's query.

    Args:
        result: Original retrieval result.
        pool:   Prompt injection passages to draw from.
        config: Corruption configuration.

    Returns:
        CorruptedResult with prompt injection passages injected.

    Raises:
        ValueError: If pool is too small.
    """
    n = _compute_n(len(result.passages), config.fraction)
    if n > len(pool):
        raise ValueError(
            f"Insufficient pool: need {n} injection passages but pool has {len(pool)}."
        )

    rng = random.Random(config.seed)
    targets = _select_targets(result.passages, n, config.target_policy, rng)
    pool_sample = rng.sample(pool, n)

    passages = list(result.passages)
    log: list[CorruptionEntry] = []

    for target_idx, replacement in zip(sorted(targets), pool_sample):
        log.append(CorruptionEntry(
            index=target_idx,
            corruption_type="prompt_injection",
            original_passage=passages[target_idx],
            injected_passage=replacement,
        ))
        passages[target_idx] = replacement

    return CorruptedResult(
        retrieval_result=RetrievalResult(query=result.query, passages=passages),
        original_result=result,
        corruption_log=log,
    )


# ═══════════════════════════════════════════════════════════════════
# Composable pipeline
# ═══════════════════════════════════════════════════════════════════


# Map from step name to the corruption function
_INJECTION_FUNCTIONS = {
    "distractor": inject_distractors,
    "adversarial": inject_adversarial,
    "prompt_injection": inject_prompt_injection,
}


class CorruptionPipeline:
    """Composable pipeline that chains multiple corruption operations.

    Each step applies a corruption to the result of the previous step.
    The final CorruptedResult preserves the original (pre-any-corruption)
    result and accumulates all corruption log entries.

    Args:
        steps: List of (corruption_name, kwargs) tuples.
               Each kwargs dict is passed to the corresponding function.
    """

    def __init__(self, steps: list[tuple[str, dict]]) -> None:
        self.steps = steps

    def apply(self, result: RetrievalResult) -> CorruptedResult:
        """Apply all corruption steps in sequence.

        Args:
            result: The original uncorrupted retrieval result.

        Returns:
            CorruptedResult with accumulated corruption log.
        """
        original = result
        current = result
        all_log: list[CorruptionEntry] = []
        all_elapsed: list[float | None] | None = None

        for step_name, kwargs in self.steps:
            step_result = self._apply_step(step_name, current, kwargs)
            all_log.extend(step_result.corruption_log)
            current = step_result.retrieval_result

            # Merge elapsed times
            if step_result.elapsed_times is not None:
                if all_elapsed is None:
                    all_elapsed = step_result.elapsed_times
                else:
                    # Merge: keep non-None values from both
                    merged: list[float | None] = []
                    for i in range(len(step_result.elapsed_times)):
                        if i < len(all_elapsed) and all_elapsed[i] is not None:
                            merged.append(all_elapsed[i])
                        else:
                            merged.append(step_result.elapsed_times[i])
                    all_elapsed = merged

        return CorruptedResult(
            retrieval_result=current,
            original_result=original,
            corruption_log=all_log,
            elapsed_times=all_elapsed,
        )

    def _apply_step(
        self,
        step_name: str,
        result: RetrievalResult,
        kwargs: dict,
    ) -> CorruptedResult:
        """Apply a single corruption step."""
        config = kwargs["config"]

        if step_name in _INJECTION_FUNCTIONS:
            pool = kwargs["pool"]
            return _INJECTION_FUNCTIONS[step_name](result, pool, config)
        elif step_name == "contradiction":
            return inject_contradictions(result, kwargs["contradiction_fn"], config)
        elif step_name == "evidence_removal":
            return remove_evidence(result, config)
        elif step_name == "timestamp_backdate":
            return backdate_timestamps(result, kwargs["max_age_hours"], config)
        else:
            raise ValueError(f"Unknown corruption step: {step_name!r}")


# ═══════════════════════════════════════════════════════════════════
# Named composite: low recall
# ═══════════════════════════════════════════════════════════════════


def simulate_low_recall(
    result: RetrievalResult,
    distractor_pool: list[RetrievedPassage],
    removal_fraction: float = 0.5,
    distractor_fraction: float = 0.3,
    seed: int = 0,
) -> CorruptedResult:
    """Simulate a low-recall retriever: remove gold + add distractors.

    Models the scenario where the retriever fails to find relevant
    documents and returns mostly irrelevant ones instead.

    Args:
        result:              Original retrieval result.
        distractor_pool:     Distractor passages to draw from.
        removal_fraction:    Fraction of passages to remove (gold_first).
        distractor_fraction: Fraction of remaining to replace with distractors.
        seed:                RNG seed.

    Returns:
        CorruptedResult with combined removal + distractor corruption.
    """
    pipeline = CorruptionPipeline(steps=[
        ("evidence_removal", dict(config=CorruptionConfig(
            corruption_type="evidence_removal",
            fraction=removal_fraction,
            target_policy="gold_first",
            seed=seed,
        ))),
        ("distractor", dict(
            pool=distractor_pool,
            config=CorruptionConfig(
                corruption_type="distractor",
                fraction=distractor_fraction,
                target_policy="random",
                seed=seed + 1,
            ),
        )),
    ])
    return pipeline.apply(result)
