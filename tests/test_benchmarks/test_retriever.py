"""TDD tests for retriever wrapper.

Tests the standardized retrieval interface with three implementations:
- PrecomputedRetriever: loads pre-computed retrieval results from JSONL
- GoldRetriever: wraps gold context from QAExample.paragraphs
- ContrieverRetriever: live retrieval stub (not yet implemented)

Red phase: all tests should FAIL until implementation is written.
"""

from __future__ import annotations

import abc
import json
import os
import tempfile
from dataclasses import fields
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Section 1: Data structure tests
# ---------------------------------------------------------------------------


class TestRetrievedPassage:
    """Tests for the RetrievedPassage dataclass."""

    def test_import(self):
        from xrag.benchmarks.retriever import RetrievedPassage

        assert RetrievedPassage is not None

    def test_required_fields(self):
        from xrag.benchmarks.retriever import RetrievedPassage

        p = RetrievedPassage(id="1", title="Title", text="Content", score=0.95)
        assert p.id == "1"
        assert p.title == "Title"
        assert p.text == "Content"
        assert p.score == 0.95

    def test_has_answer_defaults_none(self):
        from xrag.benchmarks.retriever import RetrievedPassage

        p = RetrievedPassage(id="1", title="T", text="X", score=0.5)
        assert p.has_answer is None

    def test_has_answer_explicit(self):
        from xrag.benchmarks.retriever import RetrievedPassage

        p = RetrievedPassage(id="1", title="T", text="X", score=0.5, has_answer=True)
        assert p.has_answer is True

    def test_field_names(self):
        from xrag.benchmarks.retriever import RetrievedPassage

        names = {f.name for f in fields(RetrievedPassage)}
        assert names == {"id", "title", "text", "score", "has_answer"}


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_import(self):
        from xrag.benchmarks.retriever import RetrievalResult

        assert RetrievalResult is not None

    def test_construction(self):
        from xrag.benchmarks.retriever import RetrievalResult, RetrievedPassage

        passages = [
            RetrievedPassage(id="1", title="T1", text="X1", score=0.9),
            RetrievedPassage(id="2", title="T2", text="X2", score=0.7),
        ]
        result = RetrievalResult(query="What is X?", passages=passages)
        assert result.query == "What is X?"
        assert len(result.passages) == 2

    def test_field_names(self):
        from xrag.benchmarks.retriever import RetrievalResult

        names = {f.name for f in fields(RetrievalResult)}
        assert names == {"query", "passages"}


# ---------------------------------------------------------------------------
# Section 2: Abstract Retriever interface tests
# ---------------------------------------------------------------------------


class TestRetrieverInterface:
    """Tests for the abstract Retriever base class."""

    def test_import(self):
        from xrag.benchmarks.retriever import Retriever

        assert Retriever is not None

    def test_is_abstract(self):
        from xrag.benchmarks.retriever import Retriever

        assert abc.ABC in Retriever.__mro__

    def test_cannot_instantiate(self):
        from xrag.benchmarks.retriever import Retriever

        with pytest.raises(TypeError):
            Retriever()

    def test_has_retrieve_method(self):
        from xrag.benchmarks.retriever import Retriever

        assert hasattr(Retriever, "retrieve")

    def test_has_retrieve_batch_method(self):
        from xrag.benchmarks.retriever import Retriever

        assert hasattr(Retriever, "retrieve_batch")

    def test_retrieve_is_abstract(self):
        from xrag.benchmarks.retriever import Retriever

        assert getattr(Retriever.retrieve, "__isabstractmethod__", False)

    def test_retrieve_batch_is_abstract(self):
        from xrag.benchmarks.retriever import Retriever

        assert getattr(Retriever.retrieve_batch, "__isabstractmethod__", False)


# ---------------------------------------------------------------------------
# Section 3: Concrete class import tests
# ---------------------------------------------------------------------------


class TestPrecomputedRetrieverBasics:
    def test_import(self):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        assert PrecomputedRetriever is not None

    def test_is_retriever(self):
        from xrag.benchmarks.retriever import PrecomputedRetriever, Retriever

        assert issubclass(PrecomputedRetriever, Retriever)


class TestGoldRetrieverBasics:
    def test_import(self):
        from xrag.benchmarks.retriever import GoldRetriever

        assert GoldRetriever is not None

    def test_is_retriever(self):
        from xrag.benchmarks.retriever import GoldRetriever, Retriever

        assert issubclass(GoldRetriever, Retriever)


class TestContrieverRetrieverBasics:
    def test_import(self):
        from xrag.benchmarks.retriever import ContrieverRetriever

        assert ContrieverRetriever is not None

    def test_is_retriever(self):
        from xrag.benchmarks.retriever import ContrieverRetriever, Retriever

        assert issubclass(ContrieverRetriever, Retriever)


# ---------------------------------------------------------------------------
# Section 4: PrecomputedRetriever tests
# ---------------------------------------------------------------------------


def _make_lost_in_middle_jsonl_content():
    """Create mock JSONL content in 'Lost in the Middle' format."""
    rows = [
        {
            "question": "who got the first nobel prize in physics",
            "answers": ["Wilhelm Conrad Röntgen"],
            "ctxs": [
                {
                    "id": "71445",
                    "title": "Nobel Prize in Physics",
                    "text": "The first Nobel Prize in Physics was awarded to Wilhelm Conrad Röntgen.",
                    "score": "1.0510446",
                    "hasanswer": True,
                    "isgold": True,
                    "original_retrieval_index": 0,
                },
                {
                    "id": "83221",
                    "title": "Physics",
                    "text": "Physics is the natural science of matter.",
                    "score": "0.8342",
                    "hasanswer": False,
                    "isgold": False,
                    "original_retrieval_index": 1,
                },
                {
                    "id": "10092",
                    "title": "Alfred Nobel",
                    "text": "Alfred Nobel established the Nobel Prizes.",
                    "score": "0.7112",
                    "hasanswer": False,
                    "isgold": False,
                    "original_retrieval_index": 2,
                },
            ],
        },
        {
            "question": "what is the capital of france",
            "answers": ["Paris"],
            "ctxs": [
                {
                    "id": "22001",
                    "title": "France",
                    "text": "Paris is the capital and largest city of France.",
                    "score": "1.2301",
                    "hasanswer": True,
                    "isgold": True,
                    "original_retrieval_index": 0,
                },
                {
                    "id": "22050",
                    "title": "Paris",
                    "text": "Paris is known as the City of Light.",
                    "score": "0.9801",
                    "hasanswer": False,
                    "isgold": False,
                    "original_retrieval_index": 1,
                },
            ],
        },
    ]
    return rows


@pytest.fixture
def precomputed_jsonl_file(tmp_path):
    """Write mock JSONL to a temp file and return its path."""
    filepath = tmp_path / "nq-contriever-results.jsonl"
    rows = _make_lost_in_middle_jsonl_content()
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(filepath)


class TestPrecomputedRetrieverLoad:
    """Tests for PrecomputedRetriever loading and querying."""

    def test_instantiate_with_file(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        assert retriever is not None

    def test_retrieve_returns_retrieval_result(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever, RetrievalResult

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        result = retriever.retrieve("who got the first nobel prize in physics")
        assert isinstance(result, RetrievalResult)

    def test_retrieve_correct_query(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        result = retriever.retrieve("who got the first nobel prize in physics")
        assert result.query == "who got the first nobel prize in physics"

    def test_retrieve_returns_passages(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever, RetrievedPassage

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        result = retriever.retrieve("who got the first nobel prize in physics")
        assert len(result.passages) == 3
        assert all(isinstance(p, RetrievedPassage) for p in result.passages)

    def test_passage_fields_mapped(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        result = retriever.retrieve("who got the first nobel prize in physics")
        p0 = result.passages[0]
        assert p0.id == "71445"
        assert p0.title == "Nobel Prize in Physics"
        assert "Röntgen" in p0.text
        assert p0.score == pytest.approx(1.0510446)
        assert p0.has_answer is True

    def test_retrieve_top_k(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        result = retriever.retrieve("who got the first nobel prize in physics", top_k=2)
        assert len(result.passages) == 2

    def test_retrieve_unknown_query_returns_empty(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        result = retriever.retrieve("completely unknown query")
        assert len(result.passages) == 0

    def test_retrieve_second_query(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        result = retriever.retrieve("what is the capital of france")
        assert len(result.passages) == 2
        assert result.passages[0].title == "France"

    def test_passages_sorted_by_score_descending(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        result = retriever.retrieve("who got the first nobel prize in physics")
        scores = [p.score for p in result.passages]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_batch(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever, RetrievalResult

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        queries = [
            "who got the first nobel prize in physics",
            "what is the capital of france",
        ]
        results = retriever.retrieve_batch(queries, top_k=10)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].query == queries[0]
        assert results[1].query == queries[1]

    def test_retrieve_batch_top_k(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        results = retriever.retrieve_batch(
            ["who got the first nobel prize in physics"], top_k=1
        )
        assert len(results[0].passages) == 1

    def test_num_queries_property(self, precomputed_jsonl_file):
        from xrag.benchmarks.retriever import PrecomputedRetriever

        retriever = PrecomputedRetriever(precomputed_jsonl_file)
        assert retriever.num_queries == 2


class TestPrecomputedRetrieverGzip:
    """Test that PrecomputedRetriever handles gzipped JSONL files."""

    def test_load_gzipped_file(self, tmp_path):
        import gzip

        from xrag.benchmarks.retriever import PrecomputedRetriever

        filepath = tmp_path / "results.jsonl.gz"
        rows = _make_lost_in_middle_jsonl_content()
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        retriever = PrecomputedRetriever(str(filepath))
        result = retriever.retrieve("what is the capital of france")
        assert len(result.passages) == 2


# ---------------------------------------------------------------------------
# Section 5: GoldRetriever tests
# ---------------------------------------------------------------------------


class TestGoldRetriever:
    """Tests for GoldRetriever using gold context from QAExample."""

    def _make_examples(self):
        from xrag.benchmarks.data_loader import Paragraph, QAExample

        return [
            QAExample(
                id="hotpot-1",
                question="Were A and B the same nationality?",
                answers=["yes"],
                paragraphs=[
                    Paragraph(title="Doc A", text="A is American.", is_supporting=True),
                    Paragraph(title="Doc B", text="B is American.", is_supporting=True),
                    Paragraph(title="Distractor", text="Irrelevant.", is_supporting=False),
                ],
            ),
            QAExample(
                id="hotpot-2",
                question="Who wrote Hamlet?",
                answers=["Shakespeare"],
                paragraphs=[
                    Paragraph(title="Shakespeare", text="He wrote Hamlet.", is_supporting=True),
                ],
            ),
        ]

    def test_instantiate(self):
        from xrag.benchmarks.retriever import GoldRetriever

        examples = self._make_examples()
        retriever = GoldRetriever(examples)
        assert retriever is not None

    def test_retrieve_returns_result(self):
        from xrag.benchmarks.retriever import GoldRetriever, RetrievalResult

        examples = self._make_examples()
        retriever = GoldRetriever(examples)
        result = retriever.retrieve("Were A and B the same nationality?")
        assert isinstance(result, RetrievalResult)

    def test_retrieve_returns_all_paragraphs(self):
        from xrag.benchmarks.retriever import GoldRetriever

        examples = self._make_examples()
        retriever = GoldRetriever(examples)
        result = retriever.retrieve("Were A and B the same nationality?")
        assert len(result.passages) == 3

    def test_supporting_score_higher(self):
        """Supporting passages should have score=1.0, non-supporting score=0.0."""
        from xrag.benchmarks.retriever import GoldRetriever

        examples = self._make_examples()
        retriever = GoldRetriever(examples)
        result = retriever.retrieve("Were A and B the same nationality?")
        supporting = [p for p in result.passages if p.score == 1.0]
        non_supporting = [p for p in result.passages if p.score == 0.0]
        assert len(supporting) == 2
        assert len(non_supporting) == 1

    def test_retrieve_top_k(self):
        from xrag.benchmarks.retriever import GoldRetriever

        examples = self._make_examples()
        retriever = GoldRetriever(examples)
        result = retriever.retrieve("Were A and B the same nationality?", top_k=2)
        assert len(result.passages) == 2

    def test_retrieve_unknown_query_returns_empty(self):
        from xrag.benchmarks.retriever import GoldRetriever

        examples = self._make_examples()
        retriever = GoldRetriever(examples)
        result = retriever.retrieve("totally unknown question")
        assert len(result.passages) == 0

    def test_retrieve_batch(self):
        from xrag.benchmarks.retriever import GoldRetriever

        examples = self._make_examples()
        retriever = GoldRetriever(examples)
        queries = ["Were A and B the same nationality?", "Who wrote Hamlet?"]
        results = retriever.retrieve_batch(queries)
        assert len(results) == 2
        assert len(results[0].passages) == 3
        assert len(results[1].passages) == 1

    def test_rejects_example_without_paragraphs(self):
        """GoldRetriever should skip examples with paragraphs=None."""
        from xrag.benchmarks.data_loader import QAExample
        from xrag.benchmarks.retriever import GoldRetriever

        examples = [
            QAExample(id="nq-1", question="Q?", answers=["A"]),  # paragraphs=None
        ]
        retriever = GoldRetriever(examples)
        result = retriever.retrieve("Q?")
        assert len(result.passages) == 0


# ---------------------------------------------------------------------------
# Section 6: ContrieverRetriever stub tests
# ---------------------------------------------------------------------------


class TestContrieverRetrieverStub:
    """Tests for ContrieverRetriever stub — should raise NotImplementedError."""

    def test_instantiate(self):
        from xrag.benchmarks.retriever import ContrieverRetriever

        retriever = ContrieverRetriever()
        assert retriever is not None

    def test_retrieve_raises_not_implemented(self):
        from xrag.benchmarks.retriever import ContrieverRetriever

        retriever = ContrieverRetriever()
        with pytest.raises(NotImplementedError):
            retriever.retrieve("test query")

    def test_retrieve_batch_raises_not_implemented(self):
        from xrag.benchmarks.retriever import ContrieverRetriever

        retriever = ContrieverRetriever()
        with pytest.raises(NotImplementedError):
            retriever.retrieve_batch(["test query"])
