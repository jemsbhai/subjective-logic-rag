"""NLI-based opinion estimators: DeBERTa MNLI → SL opinions.

Architecture:
    NLIEstimator (ABC — shared model loading, inference, doc-doc conflict)
        ├── NLIRelevanceEstimator    (Stage 1: no disbelief)
        └── NLIFaithfulnessEstimator (Stage 2: full 3-class)

Stage 1 (Relevance): NLI(document, query)
    Used pre-generation to assess document relevance to the query.
    Questions are not declarative, so NLI contradiction is unreliable.
    Contradiction signal is routed to uncertainty, not disbelief.

    b = entailment × (1 − base_u)
    d = 0
    u = base_u + (neutral + contradiction) × (1 − base_u)

Stage 2 (Faithfulness): NLI(document, generated_answer)
    Used post-generation to assess whether documents support the answer.
    Both inputs are declarative, so full 3-class mapping is valid.

    b = entailment × (1 − base_u)
    d = contradiction × (1 − base_u)
    u = base_u + neutral × (1 − base_u)

Doc-doc conflict: NLI(doc_a, doc_b)
    Both inputs are declarative → uses faithfulness mapping.
    Available on all subclasses via the shared NLIEstimator base.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

import torch
from jsonld_ex.confidence_algebra import Opinion

from xrag.opinion_estimation.base import BaseOpinionEstimator, EstimationResult

# Default NLI model — DeBERTa-v3-large fine-tuned on MNLI+Fever+ANLI+LingNLI+WANLI
# Label order: [entailment, neutral, contradiction] (indices 0, 1, 2)
_DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# Default evidence weight for base_u = 1/(W+1) mapping
_DEFAULT_EVIDENCE_WEIGHT = 10.0

# Default base rate for binary opinion domain
_DEFAULT_BASE_RATE = 0.5


class NLIEstimator(BaseOpinionEstimator):
    """Abstract base for NLI-based opinion estimators.

    Provides shared model loading, NLI inference, and document-document
    conflict estimation. Subclasses implement the NLI-to-opinion mapping
    via ``_nli_scores_to_opinion()`` and the ``estimate``/``estimate_batch``
    interface.

    Not directly instantiable — use NLIRelevanceEstimator or
    NLIFaithfulnessEstimator.

    Args:
        model_name: HuggingFace model identifier for the NLI model.
        evidence_weight: W in the mapping formula. Higher → lower base uncertainty.
            Must be positive.
        device: Torch device string ('cpu', 'cuda', 'cuda:0', etc.).
        batch_size: Maximum batch size for NLI inference.
        _lazy_load: If True, defer model loading until first inference call.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        evidence_weight: float = _DEFAULT_EVIDENCE_WEIGHT,
        device: str | None = None,
        batch_size: int = 32,
        _lazy_load: bool = False,
    ) -> None:
        if evidence_weight <= 0:
            raise ValueError(
                f"evidence_weight must be positive, got {evidence_weight}"
            )

        self.model_name = model_name
        self.evidence_weight = evidence_weight
        self.batch_size = batch_size

        # Resolve device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model and tokenizer — loaded lazily or eagerly
        self._model = None
        self._tokenizer = None
        if not _lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Load the NLI model and tokenizer from HuggingFace."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self._model.eval()

    def _ensure_model_loaded(self) -> None:
        """Load model if not already loaded (lazy init)."""
        if self._model is None or self._tokenizer is None:
            self._load_model()

    # ------------------------------------------------------------------
    # NLI inference (shared)
    # ------------------------------------------------------------------

    def _predict_nli(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[float, float, float]]:
        """Run NLI inference on (premise, hypothesis) pairs.

        Args:
            pairs: List of (premise, hypothesis) tuples.

        Returns:
            List of (entailment, neutral, contradiction) probability tuples.
        """
        self._ensure_model_loaded()

        if len(pairs) == 0:
            return []

        all_scores: list[tuple[float, float, float]] = []

        # Process in batches
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            premises = [p[0] for p in batch_pairs]
            hypotheses = [p[1] for p in batch_pairs]

            inputs = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()

            # MoritzLaurer DeBERTa label order: 0=entailment, 1=neutral, 2=contradiction
            for prob_row in probs:
                entailment = prob_row[0]
                neutral = prob_row[1]
                contradiction = prob_row[2]
                all_scores.append((entailment, neutral, contradiction))

        return all_scores

    # ------------------------------------------------------------------
    # Abstract mapping (subclasses implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def _nli_scores_to_opinion(
        self,
        entailment: float,
        neutral: float,
        contradiction: float,
    ) -> Opinion:
        """Map NLI probability triple to an SL opinion.

        Subclasses implement the specific mapping formula.

        Args:
            entailment: P(entailment) from NLI model.
            neutral: P(neutral) from NLI model.
            contradiction: P(contradiction) from NLI model.

        Returns:
            SL Opinion with base_rate = 0.5.
        """
        ...

    # ------------------------------------------------------------------
    # Document-document conflict estimation (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def _faithfulness_mapping(
        entailment: float,
        neutral: float,
        contradiction: float,
        evidence_weight: float,
    ) -> Opinion:
        """Full 3-class faithfulness mapping (used for doc-doc pairs).

        Both inputs are declarative → contradiction is meaningful.

        Args:
            entailment: P(entailment).
            neutral: P(neutral).
            contradiction: P(contradiction).
            evidence_weight: W for base_u = 1/(W+1).

        Returns:
            SL Opinion.
        """
        # Normalize for float32 drift
        total = entailment + neutral + contradiction
        if total > 0:
            entailment /= total
            neutral /= total
            contradiction /= total

        base_u = 1.0 / (evidence_weight + 1.0)
        scale = 1.0 - base_u

        b = entailment * scale
        d = contradiction * scale
        u = base_u + neutral * scale

        return Opinion(
            belief=b,
            disbelief=d,
            uncertainty=u,
            base_rate=_DEFAULT_BASE_RATE,
        )

    def estimate_document_pair(
        self, doc_a: str, doc_b: str,
    ) -> EstimationResult:
        """Estimate opinion for a document-document pair.

        Uses faithfulness mapping since both inputs are declarative.
        The opinion represents "doc_a supports/contradicts doc_b".

        Args:
            doc_a: First document (premise).
            doc_b: Second document (hypothesis).

        Returns:
            EstimationResult with SL opinion and raw NLI scores.
        """
        scores = self._predict_nli([(doc_a, doc_b)])
        entail, neutral, contradict = scores[0]
        opinion = self._faithfulness_mapping(
            entail, neutral, contradict, self.evidence_weight
        )
        return EstimationResult(
            opinion=opinion,
            nli_scores=(entail, neutral, contradict),
            metadata={"model_name": self.model_name, "mode": "doc_pair"},
        )

    def estimate_document_pairs_batch(
        self,
        docs_a: Sequence[str],
        docs_b: Sequence[str],
    ) -> list[EstimationResult]:
        """Estimate opinions for a batch of document-document pairs.

        Args:
            docs_a: List of first documents (premises).
            docs_b: List of second documents (hypotheses).

        Returns:
            List of EstimationResult, one per pair.

        Raises:
            ValueError: If docs_a and docs_b have different lengths.
        """
        if len(docs_a) != len(docs_b):
            raise ValueError(
                f"docs_a and docs_b must have same length, "
                f"got {len(docs_a)} and {len(docs_b)}"
            )

        if len(docs_a) == 0:
            return []

        pairs = list(zip(docs_a, docs_b))
        all_scores = self._predict_nli(pairs)

        results = []
        for entail, neutral, contradict in all_scores:
            opinion = self._faithfulness_mapping(
                entail, neutral, contradict, self.evidence_weight
            )
            results.append(
                EstimationResult(
                    opinion=opinion,
                    nli_scores=(entail, neutral, contradict),
                    metadata={"model_name": self.model_name, "mode": "doc_pair"},
                )
            )
        return results

    # ------------------------------------------------------------------
    # Shared estimate / estimate_batch logic
    # ------------------------------------------------------------------

    def estimate(self, query: str, document: str) -> EstimationResult:
        """Estimate opinion for a single (query, document) pair.

        Args:
            query: The query or claim string (hypothesis).
            document: The document string (premise).

        Returns:
            EstimationResult with SL opinion and raw NLI scores.
        """
        scores = self._predict_nli([(document, query)])
        entail, neutral, contradict = scores[0]
        opinion = self._nli_scores_to_opinion(entail, neutral, contradict)
        return EstimationResult(
            opinion=opinion,
            nli_scores=(entail, neutral, contradict),
            metadata={"model_name": self.model_name},
        )

    def estimate_batch(
        self,
        queries: Sequence[str],
        documents: Sequence[str],
    ) -> list[EstimationResult]:
        """Estimate opinions for a batch of (query, document) pairs.

        Args:
            queries: List of query or claim strings (hypotheses).
            documents: List of document strings (premises).

        Returns:
            List of EstimationResult, one per pair.

        Raises:
            ValueError: If queries and documents have different lengths.
        """
        if len(queries) != len(documents):
            raise ValueError(
                f"queries and documents must have same length, "
                f"got {len(queries)} and {len(documents)}"
            )

        if len(queries) == 0:
            return []

        pairs = [(doc, query) for query, doc in zip(queries, documents)]
        all_scores = self._predict_nli(pairs)

        results = []
        for entail, neutral, contradict in all_scores:
            opinion = self._nli_scores_to_opinion(entail, neutral, contradict)
            results.append(
                EstimationResult(
                    opinion=opinion,
                    nli_scores=(entail, neutral, contradict),
                    metadata={"model_name": self.model_name},
                )
            )
        return results


class NLIRelevanceEstimator(NLIEstimator):
    """Stage 1: Relevance estimation — NLI(document, query).

    Used pre-generation. Questions are not declarative, so NLI contradiction
    is unreliable against question-form queries. Contradiction signal is
    routed to uncertainty (not disbelief).

    Mapping:
        base_u = 1 / (W + 1)
        b = entailment × (1 − base_u)
        d = 0
        u = base_u + (neutral + contradiction) × (1 − base_u)
    """

    def _nli_scores_to_opinion(
        self,
        entailment: float,
        neutral: float,
        contradiction: float,
    ) -> Opinion:
        # Normalize for float32 drift
        total = entailment + neutral + contradiction
        if total > 0:
            entailment /= total
            neutral /= total
            contradiction /= total

        base_u = 1.0 / (self.evidence_weight + 1.0)
        scale = 1.0 - base_u

        b = entailment * scale
        d = 0.0
        u = base_u + (neutral + contradiction) * scale

        return Opinion(
            belief=b,
            disbelief=d,
            uncertainty=u,
            base_rate=_DEFAULT_BASE_RATE,
        )


class NLIFaithfulnessEstimator(NLIEstimator):
    """Stage 2: Faithfulness estimation — NLI(document, generated_answer).

    Used post-generation. Both inputs are declarative, so full 3-class
    NLI mapping is epistemically valid.

    Mapping:
        base_u = 1 / (W + 1)
        b = entailment × (1 − base_u)
        d = contradiction × (1 − base_u)
        u = base_u + neutral × (1 − base_u)
    """

    def _nli_scores_to_opinion(
        self,
        entailment: float,
        neutral: float,
        contradiction: float,
    ) -> Opinion:
        return self._faithfulness_mapping(
            entailment, neutral, contradiction, self.evidence_weight
        )
