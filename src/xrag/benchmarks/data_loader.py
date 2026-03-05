"""Standardized data loaders for QA benchmark datasets.

Provides a unified interface across four benchmarks:
- Natural Questions (NQ-Open): single-hop, open-domain QA
- PopQA: single-hop, long-tail entity QA
- HotpotQA: 2-hop multi-hop QA
- MuSiQue: 2-4 hop compositional QA

All loaders emit QAExample instances with a consistent schema.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Paragraph:
    """A context paragraph (document chunk) for a QA example.

    Attributes:
        title: Document or section title.
        text: Full paragraph text content.
        is_supporting: Whether this paragraph is a gold supporting fact.
    """

    title: str
    text: str
    is_supporting: bool = False


@dataclass
class SubQuestion:
    """A sub-question in a multi-hop decomposition.

    Attributes:
        question: The sub-question text.
        answer: The gold answer to this sub-question.
        paragraph_support_idx: Index into the parent QAExample.paragraphs
            that supports this sub-question, or None if unavailable.
    """

    question: str
    answer: str
    paragraph_support_idx: Optional[int] = None


@dataclass
class QAExample:
    """A standardized question-answer example across all benchmarks.

    Attributes:
        id: Unique identifier for this example.
        question: The question text.
        answers: List of all acceptable gold answers.
        paragraphs: Gold context paragraphs if provided by the dataset,
            or None if the dataset does not supply context (e.g. NQ-Open, PopQA).
        metadata: Dataset-specific metadata (e.g. popularity, question type, level).
        sub_questions: Multi-hop question decomposition if available, or None.
        answerable: Whether the question is answerable (MuSiQue has unanswerable Qs).
    """

    id: str
    question: str
    answers: list[str]
    paragraphs: Optional[list[Paragraph]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sub_questions: Optional[list[SubQuestion]] = None
    answerable: bool = True


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DatasetLoader(abc.ABC):
    """Abstract base class for benchmark dataset loaders."""

    @property
    @abc.abstractmethod
    def dataset_name(self) -> str:
        """Canonical name for this dataset."""
        ...

    @abc.abstractmethod
    def load(self, split: str, max_samples: Optional[int] = None) -> list[QAExample]:
        """Load examples from the dataset.

        Args:
            split: Dataset split to load (e.g. 'validation', 'test').
            max_samples: If set, return at most this many examples.

        Returns:
            List of QAExample instances.
        """
        ...


# ---------------------------------------------------------------------------
# NQ-Open
# ---------------------------------------------------------------------------


class NQLoader(DatasetLoader):
    """Loader for Natural Questions Open (google-research-datasets/nq_open).

    Schema: question (str), answer (list[str]).
    No gold context paragraphs provided.
    """

    HF_DATASET_ID = "google-research-datasets/nq_open"

    @property
    def dataset_name(self) -> str:
        return "natural_questions"

    def load(self, split: str = "validation", max_samples: Optional[int] = None) -> list[QAExample]:
        ds = load_dataset(self.HF_DATASET_ID)
        data = ds[split]

        examples = []
        for i, row in enumerate(data):
            if max_samples is not None and i >= max_samples:
                break
            examples.append(
                QAExample(
                    id=f"nq-{i}",
                    question=row["question"],
                    answers=list(row["answer"]),
                )
            )
        return examples


# ---------------------------------------------------------------------------
# PopQA
# ---------------------------------------------------------------------------


class PopQALoader(DatasetLoader):
    """Loader for PopQA (akariasai/PopQA).

    Schema: question, possible_answers (JSON string), s_pop, prop, subj, obj, etc.
    No gold context paragraphs provided.
    """

    HF_DATASET_ID = "akariasai/PopQA"

    @property
    def dataset_name(self) -> str:
        return "popqa"

    def load(self, split: str = "test", max_samples: Optional[int] = None) -> list[QAExample]:
        ds = load_dataset(self.HF_DATASET_ID)
        data = ds[split]

        examples = []
        for i, row in enumerate(data):
            if max_samples is not None and i >= max_samples:
                break

            # possible_answers is stored as a JSON string in the CSV
            raw_answers = row["possible_answers"]
            if isinstance(raw_answers, str):
                answers = json.loads(raw_answers)
            else:
                answers = list(raw_answers)

            examples.append(
                QAExample(
                    id=f"popqa-{row['id']}",
                    question=row["question"],
                    answers=answers,
                    metadata={
                        "s_pop": row["s_pop"],
                        "o_pop": row["o_pop"],
                        "prop": row["prop"],
                        "subj": row["subj"],
                        "obj": row["obj"],
                    },
                )
            )
        return examples


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------


class HotpotQALoader(DatasetLoader):
    """Loader for HotpotQA distractor setting (hotpotqa/hotpot_qa).

    Schema: question, answer, context (title + sentences), supporting_facts,
    type (bridge/comparison), level (easy/medium/hard).
    """

    HF_DATASET_ID = "hotpotqa/hotpot_qa"
    HF_CONFIG = "distractor"

    @property
    def dataset_name(self) -> str:
        return "hotpotqa"

    def load(self, split: str = "validation", max_samples: Optional[int] = None) -> list[QAExample]:
        ds = load_dataset(self.HF_DATASET_ID, self.HF_CONFIG)
        data = ds[split]

        examples = []
        for i, row in enumerate(data):
            if max_samples is not None and i >= max_samples:
                break

            # Build supporting_facts title set for is_supporting marking
            sf_titles = set(row["supporting_facts"]["title"])

            # Build paragraphs from context
            paragraphs = []
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
                text = " ".join(sentences)
                paragraphs.append(
                    Paragraph(
                        title=title,
                        text=text,
                        is_supporting=(title in sf_titles),
                    )
                )

            examples.append(
                QAExample(
                    id=row["id"],
                    question=row["question"],
                    answers=[row["answer"]],
                    paragraphs=paragraphs,
                    metadata={
                        "type": row["type"],
                        "level": row["level"],
                    },
                )
            )
        return examples


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------


class MuSiQueLoader(DatasetLoader):
    """Loader for MuSiQue (dgslibisey/MuSiQue).

    Schema: question, answer, answer_aliases, paragraphs (with is_supporting),
    question_decomposition (sub-questions), answerable flag.
    """

    HF_DATASET_ID = "dgslibisey/MuSiQue"

    @property
    def dataset_name(self) -> str:
        return "musique"

    def load(self, split: str = "validation", max_samples: Optional[int] = None) -> list[QAExample]:
        ds = load_dataset(self.HF_DATASET_ID)
        data = ds[split]

        examples = []
        for i, row in enumerate(data):
            if max_samples is not None and i >= max_samples:
                break

            # Build answers: main answer + aliases, filtering empty strings
            answers = []
            if row["answer"] and row["answer"].strip():
                answers.append(row["answer"])
            if row["answer_aliases"]:
                for alias in row["answer_aliases"]:
                    if alias and alias.strip() and alias not in answers:
                        answers.append(alias)

            # For unanswerable questions, answers should be empty
            answerable = row["answerable"]
            if not answerable:
                answers = []

            # Build paragraphs
            paragraphs = []
            for para in row["paragraphs"]:
                paragraphs.append(
                    Paragraph(
                        title=para["title"],
                        text=para["paragraph_text"],
                        is_supporting=para["is_supporting"],
                    )
                )

            # Build sub-questions
            sub_questions = []
            for sq in row["question_decomposition"]:
                sub_questions.append(
                    SubQuestion(
                        question=sq["question"],
                        answer=sq["answer"],
                        paragraph_support_idx=sq.get("paragraph_support_idx"),
                    )
                )

            examples.append(
                QAExample(
                    id=row["id"],
                    question=row["question"],
                    answers=answers,
                    paragraphs=paragraphs,
                    sub_questions=sub_questions,
                    answerable=answerable,
                    metadata={
                        "num_hops": len(sub_questions),
                    },
                )
            )
        return examples
