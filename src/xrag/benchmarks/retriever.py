"""Retriever wrapper with standardized interface.

Provides three implementations:
- PrecomputedRetriever: loads pre-computed retrieval results from JSONL files
  (e.g. 'Lost in the Middle' Contriever-MSMARCO results for NQ-Open)
- GoldRetriever: wraps gold context paragraphs from QAExample as retrieval results
  (for HotpotQA/MuSiQue oracle/upper-bound experiments)
- ContrieverRetriever: live retrieval stub (interface defined, implementation deferred
  to Colab notebook with full Wikipedia FAISS index)
"""

from __future__ import annotations

import abc
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from xrag.benchmarks.data_loader import Paragraph, QAExample


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedPassage:
    """A single retrieved passage.

    Attributes:
        id: Passage identifier (from corpus).
        title: Title of the source document/article.
        text: Full passage text.
        score: Retrieval relevance score (higher = more relevant).
        has_answer: Whether this passage contains a gold answer, if known.
    """

    id: str
    title: str
    text: str
    score: float
    has_answer: Optional[bool] = None


@dataclass
class RetrievalResult:
    """Result of retrieving passages for a single query.

    Attributes:
        query: The original query string.
        passages: Retrieved passages, sorted by score descending.
    """

    query: str
    passages: list[RetrievedPassage]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Retriever(abc.ABC):
    """Abstract base class for retrievers."""

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Retrieve passages for a single query.

        Args:
            query: The query string.
            top_k: Maximum number of passages to return.

        Returns:
            RetrievalResult with passages sorted by score descending.
        """
        ...

    @abc.abstractmethod
    def retrieve_batch(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        """Retrieve passages for multiple queries.

        Args:
            queries: List of query strings.
            top_k: Maximum number of passages per query.

        Returns:
            List of RetrievalResult, one per query.
        """
        ...


# ---------------------------------------------------------------------------
# PrecomputedRetriever
# ---------------------------------------------------------------------------


class PrecomputedRetriever(Retriever):
    """Loads pre-computed retrieval results from JSONL files.

    Supports the 'Lost in the Middle' format:
        {"question": str, "answers": [...], "ctxs": [{id, title, text, score, hasanswer, ...}]}

    Handles both plain .jsonl and gzipped .jsonl.gz files.
    """

    def __init__(self, filepath: str) -> None:
        self._index: dict[str, list[RetrievedPassage]] = {}
        self._load(filepath)

    def _load(self, filepath: str) -> None:
        path = Path(filepath)
        open_fn = gzip.open if path.suffix == ".gz" else open

        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                question = row["question"]
                passages = []
                for ctx in row.get("ctxs", []):
                    passages.append(
                        RetrievedPassage(
                            id=str(ctx["id"]),
                            title=ctx.get("title", ""),
                            text=ctx.get("text", ""),
                            score=float(ctx["score"]),
                            has_answer=ctx.get("hasanswer"),
                        )
                    )
                # Sort by score descending
                passages.sort(key=lambda p: p.score, reverse=True)
                self._index[question] = passages

    @property
    def num_queries(self) -> int:
        """Number of queries loaded."""
        return len(self._index)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        passages = self._index.get(query, [])
        return RetrievalResult(query=query, passages=passages[:top_k])

    def retrieve_batch(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        return [self.retrieve(q, top_k=top_k) for q in queries]


# ---------------------------------------------------------------------------
# GoldRetriever
# ---------------------------------------------------------------------------


class GoldRetriever(Retriever):
    """Uses gold context paragraphs from QAExample as retrieval results.

    Supporting paragraphs get score=1.0, non-supporting get score=0.0.
    Passages are sorted by score descending (supporting first).

    Useful for oracle/upper-bound experiments on HotpotQA and MuSiQue.
    """

    def __init__(self, examples: list[QAExample]) -> None:
        self._index: dict[str, list[RetrievedPassage]] = {}
        for ex in examples:
            if ex.paragraphs is None:
                continue
            passages = []
            for i, para in enumerate(ex.paragraphs):
                passages.append(
                    RetrievedPassage(
                        id=f"{ex.id}_p{i}",
                        title=para.title,
                        text=para.text,
                        score=1.0 if para.is_supporting else 0.0,
                        has_answer=None,
                    )
                )
            # Sort: supporting first
            passages.sort(key=lambda p: p.score, reverse=True)
            self._index[ex.question] = passages

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        passages = self._index.get(query, [])
        return RetrievalResult(query=query, passages=passages[:top_k])

    def retrieve_batch(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        return [self.retrieve(q, top_k=top_k) for q in queries]


# ---------------------------------------------------------------------------
# ContrieverRetriever (stub)
# ---------------------------------------------------------------------------


class ContrieverRetriever(Retriever):
    """Live retrieval using Contriever model + FAISS index.

    This is a stub — the interface is defined but live retrieval
    is deferred to a Colab notebook with the full Wikipedia FAISS index.
    """

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        raise NotImplementedError(
            "ContrieverRetriever requires a FAISS index and Contriever model. "
            "Use PrecomputedRetriever with pre-computed results, or run live "
            "retrieval via the Colab notebook."
        )

    def retrieve_batch(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        raise NotImplementedError(
            "ContrieverRetriever requires a FAISS index and Contriever model. "
            "Use PrecomputedRetriever with pre-computed results, or run live "
            "retrieval via the Colab notebook."
        )
