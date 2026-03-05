"""TDD tests for benchmark data loaders.

Tests the standardized data loading interface for four QA benchmarks:
- Natural Questions (NQ-Open)
- PopQA
- HotpotQA
- MuSiQue

Red phase: all tests should FAIL until implementation is written.
"""

from __future__ import annotations

import abc
from dataclasses import fields
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Section 1: Data structure tests
# ---------------------------------------------------------------------------


class TestParagraph:
    """Tests for the Paragraph dataclass."""

    def test_import(self):
        from xrag.benchmarks.data_loader import Paragraph

        assert Paragraph is not None

    def test_required_fields(self):
        from xrag.benchmarks.data_loader import Paragraph

        p = Paragraph(title="Test Title", text="Some text content.")
        assert p.title == "Test Title"
        assert p.text == "Some text content."

    def test_is_supporting_defaults_false(self):
        from xrag.benchmarks.data_loader import Paragraph

        p = Paragraph(title="T", text="X")
        assert p.is_supporting is False

    def test_is_supporting_explicit(self):
        from xrag.benchmarks.data_loader import Paragraph

        p = Paragraph(title="T", text="X", is_supporting=True)
        assert p.is_supporting is True

    def test_field_names(self):
        from xrag.benchmarks.data_loader import Paragraph

        names = {f.name for f in fields(Paragraph)}
        assert names == {"title", "text", "is_supporting"}


class TestSubQuestion:
    """Tests for the SubQuestion dataclass (multi-hop decomposition)."""

    def test_import(self):
        from xrag.benchmarks.data_loader import SubQuestion

        assert SubQuestion is not None

    def test_required_fields(self):
        from xrag.benchmarks.data_loader import SubQuestion

        sq = SubQuestion(question="Who directed X?", answer="Spielberg")
        assert sq.question == "Who directed X?"
        assert sq.answer == "Spielberg"

    def test_paragraph_support_idx_defaults_none(self):
        from xrag.benchmarks.data_loader import SubQuestion

        sq = SubQuestion(question="Q", answer="A")
        assert sq.paragraph_support_idx is None

    def test_paragraph_support_idx_explicit(self):
        from xrag.benchmarks.data_loader import SubQuestion

        sq = SubQuestion(question="Q", answer="A", paragraph_support_idx=3)
        assert sq.paragraph_support_idx == 3

    def test_field_names(self):
        from xrag.benchmarks.data_loader import SubQuestion

        names = {f.name for f in fields(SubQuestion)}
        assert names == {"question", "answer", "paragraph_support_idx"}


class TestQAExample:
    """Tests for the QAExample dataclass."""

    def test_import(self):
        from xrag.benchmarks.data_loader import QAExample

        assert QAExample is not None

    def test_minimal_construction(self):
        """QAExample with only required fields (NQ-Open style)."""
        from xrag.benchmarks.data_loader import QAExample

        ex = QAExample(
            id="nq-001",
            question="What is the capital of France?",
            answers=["Paris"],
        )
        assert ex.id == "nq-001"
        assert ex.question == "What is the capital of France?"
        assert ex.answers == ["Paris"]

    def test_paragraphs_defaults_none(self):
        """Datasets without gold context should have paragraphs=None."""
        from xrag.benchmarks.data_loader import QAExample

        ex = QAExample(id="x", question="Q", answers=["A"])
        assert ex.paragraphs is None

    def test_sub_questions_defaults_none(self):
        """Non-multi-hop datasets have no sub-questions."""
        from xrag.benchmarks.data_loader import QAExample

        ex = QAExample(id="x", question="Q", answers=["A"])
        assert ex.sub_questions is None

    def test_answerable_defaults_true(self):
        from xrag.benchmarks.data_loader import QAExample

        ex = QAExample(id="x", question="Q", answers=["A"])
        assert ex.answerable is True

    def test_metadata_defaults_empty_dict(self):
        from xrag.benchmarks.data_loader import QAExample

        ex = QAExample(id="x", question="Q", answers=["A"])
        assert ex.metadata == {}

    def test_full_construction_multihop(self):
        """QAExample with all fields populated (MuSiQue style)."""
        from xrag.benchmarks.data_loader import Paragraph, QAExample, SubQuestion

        paragraphs = [
            Paragraph(title="Doc A", text="Content A", is_supporting=True),
            Paragraph(title="Doc B", text="Content B", is_supporting=False),
        ]
        sub_qs = [
            SubQuestion(question="Who wrote X?", answer="Author A", paragraph_support_idx=0),
            SubQuestion(question="When was Author A born?", answer="1960", paragraph_support_idx=1),
        ]
        ex = QAExample(
            id="musique-001",
            question="When was the author of X born?",
            answers=["1960"],
            paragraphs=paragraphs,
            sub_questions=sub_qs,
            answerable=True,
            metadata={"num_hops": 2},
        )
        assert len(ex.paragraphs) == 2
        assert len(ex.sub_questions) == 2
        assert ex.answerable is True
        assert ex.metadata["num_hops"] == 2

    def test_multiple_gold_answers(self):
        """NQ-Open can have multiple acceptable answers."""
        from xrag.benchmarks.data_loader import QAExample

        ex = QAExample(
            id="nq-002",
            question="Who wrote Hamlet?",
            answers=["William Shakespeare", "Shakespeare"],
        )
        assert len(ex.answers) == 2

    def test_unanswerable_example(self):
        """MuSiQue has unanswerable questions."""
        from xrag.benchmarks.data_loader import QAExample

        ex = QAExample(
            id="musique-unans-001",
            question="Unanswerable question",
            answers=[],
            answerable=False,
        )
        assert ex.answerable is False
        assert ex.answers == []

    def test_field_names(self):
        from xrag.benchmarks.data_loader import QAExample

        names = {f.name for f in fields(QAExample)}
        expected = {
            "id", "question", "answers", "paragraphs",
            "metadata", "sub_questions", "answerable",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# Section 2: Abstract DatasetLoader interface tests
# ---------------------------------------------------------------------------


class TestDatasetLoaderInterface:
    """Tests for the abstract DatasetLoader base class."""

    def test_import(self):
        from xrag.benchmarks.data_loader import DatasetLoader

        assert DatasetLoader is not None

    def test_is_abstract(self):
        from xrag.benchmarks.data_loader import DatasetLoader

        assert abc.ABC in DatasetLoader.__mro__

    def test_cannot_instantiate(self):
        from xrag.benchmarks.data_loader import DatasetLoader

        with pytest.raises(TypeError):
            DatasetLoader()

    def test_has_load_method(self):
        from xrag.benchmarks.data_loader import DatasetLoader

        assert hasattr(DatasetLoader, "load")

    def test_has_dataset_name_property(self):
        from xrag.benchmarks.data_loader import DatasetLoader

        assert hasattr(DatasetLoader, "dataset_name")

    def test_load_is_abstract(self):
        from xrag.benchmarks.data_loader import DatasetLoader

        # load should be an abstract method
        assert getattr(DatasetLoader.load, "__isabstractmethod__", False)

    def test_dataset_name_is_abstract(self):
        from xrag.benchmarks.data_loader import DatasetLoader

        # dataset_name should be abstract property
        assert isinstance(
            DatasetLoader.__dict__.get("dataset_name"),
            property,
        ) or getattr(
            DatasetLoader.dataset_name.fget, "__isabstractmethod__", False
        )


# ---------------------------------------------------------------------------
# Section 3: Concrete loader import and instantiation tests
# ---------------------------------------------------------------------------


class TestNQLoaderBasics:
    """Tests for NQLoader class basics."""

    def test_import(self):
        from xrag.benchmarks.data_loader import NQLoader

        assert NQLoader is not None

    def test_instantiate(self):
        from xrag.benchmarks.data_loader import NQLoader

        loader = NQLoader()
        assert loader is not None

    def test_is_dataset_loader(self):
        from xrag.benchmarks.data_loader import DatasetLoader, NQLoader

        assert issubclass(NQLoader, DatasetLoader)

    def test_dataset_name(self):
        from xrag.benchmarks.data_loader import NQLoader

        loader = NQLoader()
        assert loader.dataset_name == "natural_questions"


class TestPopQALoaderBasics:
    """Tests for PopQALoader class basics."""

    def test_import(self):
        from xrag.benchmarks.data_loader import PopQALoader

        assert PopQALoader is not None

    def test_instantiate(self):
        from xrag.benchmarks.data_loader import PopQALoader

        loader = PopQALoader()
        assert loader is not None

    def test_is_dataset_loader(self):
        from xrag.benchmarks.data_loader import DatasetLoader, PopQALoader

        assert issubclass(PopQALoader, DatasetLoader)

    def test_dataset_name(self):
        from xrag.benchmarks.data_loader import PopQALoader

        loader = PopQALoader()
        assert loader.dataset_name == "popqa"


class TestHotpotQALoaderBasics:
    """Tests for HotpotQALoader class basics."""

    def test_import(self):
        from xrag.benchmarks.data_loader import HotpotQALoader

        assert HotpotQALoader is not None

    def test_instantiate(self):
        from xrag.benchmarks.data_loader import HotpotQALoader

        loader = HotpotQALoader()
        assert loader is not None

    def test_is_dataset_loader(self):
        from xrag.benchmarks.data_loader import DatasetLoader, HotpotQALoader

        assert issubclass(HotpotQALoader, DatasetLoader)

    def test_dataset_name(self):
        from xrag.benchmarks.data_loader import HotpotQALoader

        loader = HotpotQALoader()
        assert loader.dataset_name == "hotpotqa"


class TestMuSiQueLoaderBasics:
    """Tests for MuSiQueLoader class basics."""

    def test_import(self):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        assert MuSiQueLoader is not None

    def test_instantiate(self):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        loader = MuSiQueLoader()
        assert loader is not None

    def test_is_dataset_loader(self):
        from xrag.benchmarks.data_loader import DatasetLoader, MuSiQueLoader

        assert issubclass(MuSiQueLoader, DatasetLoader)

    def test_dataset_name(self):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        loader = MuSiQueLoader()
        assert loader.dataset_name == "musique"


# ---------------------------------------------------------------------------
# Section 4: Mocked loading tests — NQ-Open
# ---------------------------------------------------------------------------


def _make_mock_nq_dataset():
    """Create mock HuggingFace dataset mimicking NQ-Open schema."""
    return [
        {
            "question": "What is the capital of France?",
            "answer": ["Paris"],
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": ["William Shakespeare", "Shakespeare"],
        },
        {
            "question": "When was the Eiffel Tower built?",
            "answer": ["1889"],
        },
    ]


class TestNQLoaderLoad:
    """Tests for NQLoader.load() with mocked HuggingFace data."""

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_returns_list_of_qa_examples(self, mock_load):
        from xrag.benchmarks.data_loader import NQLoader, QAExample

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation")
        assert isinstance(examples, list)
        assert all(isinstance(ex, QAExample) for ex in examples)

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_correct_count(self, mock_load):
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation")
        assert len(examples) == 3

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_question_mapping(self, mock_load):
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation")
        assert examples[0].question == "What is the capital of France?"

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_answers_is_list(self, mock_load):
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation")
        assert isinstance(examples[0].answers, list)
        assert examples[0].answers == ["Paris"]

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_multiple_answers(self, mock_load):
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation")
        assert len(examples[1].answers) == 2

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_paragraphs_is_none(self, mock_load):
        """NQ-Open does not provide gold context paragraphs."""
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation")
        assert examples[0].paragraphs is None

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_sub_questions_is_none(self, mock_load):
        """NQ-Open is single-hop — no sub-questions."""
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation")
        assert examples[0].sub_questions is None

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_id_is_string(self, mock_load):
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation")
        assert isinstance(examples[0].id, str)
        assert len(examples[0].id) > 0

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_max_samples(self, mock_load):
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        examples = loader.load(split="validation", max_samples=2)
        assert len(examples) == 2

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_calls_hf_load_dataset(self, mock_load):
        from xrag.benchmarks.data_loader import NQLoader

        mock_load.return_value = {"validation": _make_mock_nq_dataset()}
        loader = NQLoader()
        loader.load(split="validation")
        mock_load.assert_called_once()
        call_args = mock_load.call_args
        assert "google-research-datasets/nq_open" in call_args[0] or \
               "google-research-datasets/nq_open" == call_args[1].get("path", call_args[0][0])


# ---------------------------------------------------------------------------
# Section 5: Mocked loading tests — PopQA
# ---------------------------------------------------------------------------


def _make_mock_popqa_dataset():
    """Create mock HuggingFace dataset mimicking PopQA schema."""
    return [
        {
            "id": 4222362,
            "subj": "George Rankin",
            "prop": "occupation",
            "obj": "politician",
            "s_pop": 142,
            "o_pop": 25692,
            "question": "What is George Rankin's occupation?",
            "possible_answers": '["politician", "political leader"]',
            "s_uri": "http://www.wikidata.org/entity/Q5543720",
            "o_uri": "http://www.wikidata.org/entity/Q82955",
            "s_wiki_title": "George Rankin",
            "o_wiki_title": "Politician",
            "subj_id": 1850297,
            "prop_id": 22,
            "obj_id": 2834605,
            "s_aliases": '["George James Rankin"]',
            "o_aliases": '["political leader"]',
        },
        {
            "id": 276787,
            "subj": "Scooter Braun",
            "prop": "occupation",
            "obj": "talent manager",
            "s_pop": 66280,
            "o_pop": 4624,
            "question": "What is Scooter Braun's occupation?",
            "possible_answers": '["talent manager", "artist manager"]',
            "s_uri": "http://www.wikidata.org/entity/Q1189670",
            "o_uri": "http://www.wikidata.org/entity/Q1320883",
            "s_wiki_title": "Scooter Braun",
            "o_wiki_title": "Talent manager",
            "subj_id": 111929,
            "prop_id": 22,
            "obj_id": 169656,
            "s_aliases": '["Scott Samuel Braun"]',
            "o_aliases": '["artist manager"]',
        },
    ]


class TestPopQALoaderLoad:
    """Tests for PopQALoader.load() with mocked HuggingFace data."""

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_returns_list_of_qa_examples(self, mock_load):
        from xrag.benchmarks.data_loader import PopQALoader, QAExample

        mock_load.return_value = {"test": _make_mock_popqa_dataset()}
        loader = PopQALoader()
        examples = loader.load(split="test")
        assert isinstance(examples, list)
        assert all(isinstance(ex, QAExample) for ex in examples)

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_question_mapping(self, mock_load):
        from xrag.benchmarks.data_loader import PopQALoader

        mock_load.return_value = {"test": _make_mock_popqa_dataset()}
        loader = PopQALoader()
        examples = loader.load(split="test")
        assert examples[0].question == "What is George Rankin's occupation?"

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_answers_parsed_from_possible_answers(self, mock_load):
        from xrag.benchmarks.data_loader import PopQALoader

        mock_load.return_value = {"test": _make_mock_popqa_dataset()}
        loader = PopQALoader()
        examples = loader.load(split="test")
        assert isinstance(examples[0].answers, list)
        assert "politician" in examples[0].answers

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_paragraphs_is_none(self, mock_load):
        """PopQA does not provide gold context paragraphs."""
        from xrag.benchmarks.data_loader import PopQALoader

        mock_load.return_value = {"test": _make_mock_popqa_dataset()}
        loader = PopQALoader()
        examples = loader.load(split="test")
        assert examples[0].paragraphs is None

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_sub_questions_is_none(self, mock_load):
        from xrag.benchmarks.data_loader import PopQALoader

        mock_load.return_value = {"test": _make_mock_popqa_dataset()}
        loader = PopQALoader()
        examples = loader.load(split="test")
        assert examples[0].sub_questions is None

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_metadata_contains_popularity(self, mock_load):
        """PopQA's entity popularity is scientifically important for analysis."""
        from xrag.benchmarks.data_loader import PopQALoader

        mock_load.return_value = {"test": _make_mock_popqa_dataset()}
        loader = PopQALoader()
        examples = loader.load(split="test")
        assert "s_pop" in examples[0].metadata
        assert examples[0].metadata["s_pop"] == 142

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_metadata_contains_relation_type(self, mock_load):
        from xrag.benchmarks.data_loader import PopQALoader

        mock_load.return_value = {"test": _make_mock_popqa_dataset()}
        loader = PopQALoader()
        examples = loader.load(split="test")
        assert "prop" in examples[0].metadata
        assert examples[0].metadata["prop"] == "occupation"

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_max_samples(self, mock_load):
        from xrag.benchmarks.data_loader import PopQALoader

        mock_load.return_value = {"test": _make_mock_popqa_dataset()}
        loader = PopQALoader()
        examples = loader.load(split="test", max_samples=1)
        assert len(examples) == 1


# ---------------------------------------------------------------------------
# Section 6: Mocked loading tests — HotpotQA
# ---------------------------------------------------------------------------


def _make_mock_hotpotqa_dataset():
    """Create mock HuggingFace dataset mimicking HotpotQA distractor schema."""
    return [
        {
            "id": "5a8b57f25542995d1e6f1371",
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "answer": "yes",
            "type": "comparison",
            "level": "hard",
            "supporting_facts": {
                "title": ["Scott Derrickson", "Ed Wood"],
                "sent_id": [0, 0],
            },
            "context": {
                "title": ["Scott Derrickson", "Ed Wood", "Irrelevant Doc"],
                "sentences": [
                    ["Scott Derrickson is an American filmmaker."],
                    ["Edward Davis Wood Jr. was an American filmmaker."],
                    ["This is an irrelevant distractor document."],
                ],
            },
        },
        {
            "id": "5a8c7595554299585d9e36b6",
            "question": "What government position was held by the woman who was combative?",
            "answer": "Secretary of State",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {
                "title": ["Condoleezza Rice"],
                "sent_id": [0],
            },
            "context": {
                "title": ["Condoleezza Rice", "Another Doc"],
                "sentences": [
                    ["Condoleezza Rice served as Secretary of State."],
                    ["Another document with some content."],
                ],
            },
        },
    ]


class TestHotpotQALoaderLoad:
    """Tests for HotpotQALoader.load() with mocked HuggingFace data."""

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_returns_list_of_qa_examples(self, mock_load):
        from xrag.benchmarks.data_loader import HotpotQALoader, QAExample

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        assert isinstance(examples, list)
        assert all(isinstance(ex, QAExample) for ex in examples)

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_answer_is_list(self, mock_load):
        """HotpotQA has single gold answers but we wrap in list for consistency."""
        from xrag.benchmarks.data_loader import HotpotQALoader

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        assert isinstance(examples[0].answers, list)
        assert examples[0].answers == ["yes"]

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_paragraphs_populated(self, mock_load):
        """HotpotQA provides gold context — paragraphs must NOT be None."""
        from xrag.benchmarks.data_loader import HotpotQALoader, Paragraph

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        assert examples[0].paragraphs is not None
        assert len(examples[0].paragraphs) > 0
        assert all(isinstance(p, Paragraph) for p in examples[0].paragraphs)

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_paragraph_titles(self, mock_load):
        from xrag.benchmarks.data_loader import HotpotQALoader

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        titles = {p.title for p in examples[0].paragraphs}
        assert "Scott Derrickson" in titles
        assert "Ed Wood" in titles

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_paragraph_text_joined(self, mock_load):
        """Each paragraph's sentences should be joined into a single text string."""
        from xrag.benchmarks.data_loader import HotpotQALoader

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        for p in examples[0].paragraphs:
            assert isinstance(p.text, str)
            assert len(p.text) > 0

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_supporting_facts_marked(self, mock_load):
        """Paragraphs that are supporting facts should have is_supporting=True."""
        from xrag.benchmarks.data_loader import HotpotQALoader

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        supporting = [p for p in examples[0].paragraphs if p.is_supporting]
        non_supporting = [p for p in examples[0].paragraphs if not p.is_supporting]
        assert len(supporting) >= 1
        assert len(non_supporting) >= 1  # distractor docs

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_metadata_contains_type(self, mock_load):
        from xrag.benchmarks.data_loader import HotpotQALoader

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        assert "type" in examples[0].metadata
        assert examples[0].metadata["type"] == "comparison"

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_metadata_contains_level(self, mock_load):
        from xrag.benchmarks.data_loader import HotpotQALoader

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        assert "level" in examples[0].metadata
        assert examples[0].metadata["level"] == "hard"

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_sub_questions_is_none(self, mock_load):
        """HotpotQA does not provide explicit question decompositions."""
        from xrag.benchmarks.data_loader import HotpotQALoader

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation")
        assert examples[0].sub_questions is None

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_max_samples(self, mock_load):
        from xrag.benchmarks.data_loader import HotpotQALoader

        mock_load.return_value = {"validation": _make_mock_hotpotqa_dataset()}
        loader = HotpotQALoader()
        examples = loader.load(split="validation", max_samples=1)
        assert len(examples) == 1


# ---------------------------------------------------------------------------
# Section 7: Mocked loading tests — MuSiQue
# ---------------------------------------------------------------------------


def _make_mock_musique_dataset():
    """Create mock HuggingFace dataset mimicking MuSiQue schema."""
    return [
        {
            "id": "2hop__482757_12019",
            "question": "When was the institute that owned The Collegian founded?",
            "answer": "1960",
            "answer_aliases": [],
            "answerable": True,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Pakistan Super League",
                    "paragraph_text": "PSL is a Twenty20 cricket league.",
                    "is_supporting": False,
                },
                {
                    "idx": 1,
                    "title": "The Collegian (HBU)",
                    "paragraph_text": "The Collegian is the bi-weekly student publication of HBU.",
                    "is_supporting": True,
                },
                {
                    "idx": 2,
                    "title": "Houston",
                    "paragraph_text": "Houston Baptist University was founded in 1960.",
                    "is_supporting": True,
                },
            ],
            "question_decomposition": [
                {
                    "id": 482757,
                    "question": "The Collegian >> owned by",
                    "answer": "Houston Baptist University",
                    "paragraph_support_idx": 1,
                },
                {
                    "id": 12019,
                    "question": "When was #1 founded?",
                    "answer": "1960",
                    "paragraph_support_idx": 2,
                },
            ],
        },
        {
            "id": "2hop__unans_001",
            "question": "An unanswerable multi-hop question?",
            "answer": "",
            "answer_aliases": [],
            "answerable": False,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Some Doc",
                    "paragraph_text": "Some irrelevant content.",
                    "is_supporting": False,
                },
            ],
            "question_decomposition": [
                {
                    "id": 99999,
                    "question": "Sub-question 1",
                    "answer": "unknown",
                    "paragraph_support_idx": None,
                },
            ],
        },
    ]


class TestMuSiQueLoaderLoad:
    """Tests for MuSiQueLoader.load() with mocked HuggingFace data."""

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_returns_list_of_qa_examples(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader, QAExample

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        assert isinstance(examples, list)
        assert all(isinstance(ex, QAExample) for ex in examples)

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_answer_and_aliases_merged(self, mock_load):
        """answers should include main answer + aliases."""
        from xrag.benchmarks.data_loader import MuSiQueLoader

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        assert "1960" in examples[0].answers

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_paragraphs_populated(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader, Paragraph

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        assert examples[0].paragraphs is not None
        assert len(examples[0].paragraphs) == 3
        assert all(isinstance(p, Paragraph) for p in examples[0].paragraphs)

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_paragraph_is_supporting(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        supporting = [p for p in examples[0].paragraphs if p.is_supporting]
        assert len(supporting) == 2

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_sub_questions_populated(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader, SubQuestion

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        assert examples[0].sub_questions is not None
        assert len(examples[0].sub_questions) == 2
        assert all(isinstance(sq, SubQuestion) for sq in examples[0].sub_questions)

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_sub_question_fields(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        sq0 = examples[0].sub_questions[0]
        assert sq0.question == "The Collegian >> owned by"
        assert sq0.answer == "Houston Baptist University"
        assert sq0.paragraph_support_idx == 1

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_answerable_flag_true(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        assert examples[0].answerable is True

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_answerable_flag_false(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        assert examples[1].answerable is False

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_unanswerable_has_empty_answers(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        # Unanswerable: answers should be empty list
        assert examples[1].answers == []

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_metadata_contains_num_hops(self, mock_load):
        """num_hops derived from question_decomposition length."""
        from xrag.benchmarks.data_loader import MuSiQueLoader

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation")
        assert "num_hops" in examples[0].metadata
        assert examples[0].metadata["num_hops"] == 2

    @patch("xrag.benchmarks.data_loader.load_dataset")
    def test_max_samples(self, mock_load):
        from xrag.benchmarks.data_loader import MuSiQueLoader

        mock_load.return_value = {"validation": _make_mock_musique_dataset()}
        loader = MuSiQueLoader()
        examples = loader.load(split="validation", max_samples=1)
        assert len(examples) == 1


# ---------------------------------------------------------------------------
# Section 8: Integration tests (real HuggingFace downloads)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationNQ:
    """Integration tests for NQLoader with real HuggingFace data."""

    def test_load_small_sample(self):
        from xrag.benchmarks.data_loader import NQLoader, QAExample

        loader = NQLoader()
        examples = loader.load(split="validation", max_samples=5)
        assert len(examples) == 5
        for ex in examples:
            assert isinstance(ex, QAExample)
            assert isinstance(ex.question, str) and len(ex.question) > 0
            assert isinstance(ex.answers, list) and len(ex.answers) >= 1
            assert ex.paragraphs is None
            assert ex.sub_questions is None
            assert ex.answerable is True


@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationPopQA:
    """Integration tests for PopQALoader with real HuggingFace data."""

    def test_load_small_sample(self):
        from xrag.benchmarks.data_loader import PopQALoader, QAExample

        loader = PopQALoader()
        examples = loader.load(split="test", max_samples=5)
        assert len(examples) == 5
        for ex in examples:
            assert isinstance(ex, QAExample)
            assert isinstance(ex.question, str) and len(ex.question) > 0
            assert isinstance(ex.answers, list) and len(ex.answers) >= 1
            assert ex.paragraphs is None
            assert "s_pop" in ex.metadata


@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationHotpotQA:
    """Integration tests for HotpotQALoader with real HuggingFace data."""

    def test_load_small_sample(self):
        from xrag.benchmarks.data_loader import HotpotQALoader, Paragraph, QAExample

        loader = HotpotQALoader()
        examples = loader.load(split="validation", max_samples=5)
        assert len(examples) == 5
        for ex in examples:
            assert isinstance(ex, QAExample)
            assert isinstance(ex.question, str) and len(ex.question) > 0
            assert isinstance(ex.answers, list) and len(ex.answers) >= 1
            assert ex.paragraphs is not None
            assert len(ex.paragraphs) > 0
            assert all(isinstance(p, Paragraph) for p in ex.paragraphs)
            assert "type" in ex.metadata
            assert ex.metadata["type"] in ("bridge", "comparison")
            assert "level" in ex.metadata


@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationMuSiQue:
    """Integration tests for MuSiQueLoader with real HuggingFace data."""

    def test_load_small_sample(self):
        from xrag.benchmarks.data_loader import MuSiQueLoader, Paragraph, QAExample, SubQuestion

        loader = MuSiQueLoader()
        examples = loader.load(split="validation", max_samples=5)
        assert len(examples) == 5
        for ex in examples:
            assert isinstance(ex, QAExample)
            assert isinstance(ex.question, str) and len(ex.question) > 0
            assert ex.paragraphs is not None
            assert len(ex.paragraphs) > 0
            assert all(isinstance(p, Paragraph) for p in ex.paragraphs)
            assert ex.sub_questions is not None
            assert all(isinstance(sq, SubQuestion) for sq in ex.sub_questions)
            assert "num_hops" in ex.metadata
            assert isinstance(ex.answerable, bool)
            if ex.answerable:
                assert len(ex.answers) >= 1
