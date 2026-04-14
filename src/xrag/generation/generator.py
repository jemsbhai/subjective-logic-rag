"""LLM generation for SL-RAG pipeline.

Provides a Generator ABC and HuggingFaceGenerator for local inference
with models like Llama 3.1 8B Instruct.

Design:
    - Generator ABC: generate(query, passages) → GenerationResult
    - HuggingFaceGenerator: loads a local model with optional 4-bit quantization
    - RAGPromptTemplate: configurable prompt formatting
    - GenerationResult: answer, prompt, token_logprobs (for softmax entropy baseline)

Models live at D:\\cc\\models\\ (portable SSD, not primary drive).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

from xrag.benchmarks.retriever import RetrievedPassage


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class GenerationResult:
    """Result of LLM generation.

    Attributes:
        answer:         The generated answer string.
        prompt:         The full prompt sent to the model.
        token_logprobs: Per-token log probabilities (for softmax entropy baseline).
                        None if not requested.
        metadata:       Model info, timing, token counts, etc.
    """

    answer: str
    prompt: str
    token_logprobs: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Prompt template
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RAGPromptTemplate:
    """Configurable prompt template for RAG generation.

    Attributes:
        template:          Full prompt template with {query} and {context} placeholders.
        passage_format:    Format string for each passage with {idx}, {title}, {text}.
        passage_separator: String between passages.
    """

    template: str = (
        "Given the following context, answer the question concisely.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n"
        "Answer:"
    )
    passage_format: str = "[{idx}] {title}: {text}"
    passage_separator: str = "\n"

    def format(
        self,
        query: str,
        passages: list[RetrievedPassage],
    ) -> str:
        """Format the prompt with query and passages.

        Args:
            query:    The question string.
            passages: Retrieved passages to include as context.

        Returns:
            Formatted prompt string.
        """
        if passages:
            context_parts = []
            for i, p in enumerate(passages):
                part = self.passage_format.format(
                    idx=i + 1,
                    title=p.title,
                    text=p.text,
                )
                context_parts.append(part)
            context = self.passage_separator.join(context_parts)
        else:
            context = "(No context provided)"

        return self.template.format(query=query, context=context)


DEFAULT_RAG_PROMPT_TEMPLATE = RAGPromptTemplate()


# ═══════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════


class Generator(abc.ABC):
    """Abstract base for LLM generators."""

    @abc.abstractmethod
    def generate(
        self,
        query: str,
        passages: list[RetrievedPassage],
        max_new_tokens: int = 128,
        return_logprobs: bool = False,
        abstain: bool = False,
        **kwargs,
    ) -> GenerationResult:
        """Generate an answer from query and context passages.

        Args:
            query:           The question string.
            passages:        Retrieved passages as context.
            max_new_tokens:  Maximum tokens to generate.
            return_logprobs: Whether to return per-token log probs.
            abstain:         If True, return empty answer (pipeline decided to abstain).

        Returns:
            GenerationResult with answer, prompt, and optional logprobs.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# HuggingFace generator
# ═══════════════════════════════════════════════════════════════════


class HuggingFaceGenerator(Generator):
    """Local LLM generator using HuggingFace transformers.

    Loads a model from a local path with optional 4-bit quantization
    for fitting large models on consumer GPUs (e.g., RTX 4090 16GB).

    Args:
        model_path:      Path to the model directory.
        device:          Device to load on ("cuda", "cpu", "auto").
        load_in_4bit:    Whether to use 4-bit quantization (requires bitsandbytes).
        prompt_template: RAG prompt template for formatting.
        max_new_tokens:  Default max tokens for generation.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        load_in_4bit: bool = False,
        prompt_template: RAGPromptTemplate | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.prompt_template = prompt_template or DEFAULT_RAG_PROMPT_TEMPLATE
        self.max_new_tokens = max_new_tokens
        self._loaded = False
        self.model = None
        self.tokenizer = None

        self._load_model(load_in_4bit)

    def _load_model(self, load_in_4bit: bool) -> None:
        """Load model and tokenizer from local path."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            if self.device != "cpu":
                load_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        self.model.eval()
        self._loaded = True

    def generate(
        self,
        query: str,
        passages: list[RetrievedPassage],
        max_new_tokens: int | None = None,
        return_logprobs: bool = False,
        abstain: bool = False,
        **kwargs,
    ) -> GenerationResult:
        """Generate an answer using the loaded model.

        Args:
            query:           The question string.
            passages:        Retrieved passages as context.
            max_new_tokens:  Max tokens (overrides default if set).
            return_logprobs: Whether to return per-token log probs.
            abstain:         If True, skip generation and return empty answer.

        Returns:
            GenerationResult.
        """
        import torch

        prompt = self.prompt_template.format(query=query, passages=passages)

        if abstain:
            return GenerationResult(
                answer="",
                prompt=prompt,
                token_logprobs=None,
                metadata={"model_path": self.model_path, "abstained": True},
            )

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        input_len = input_ids.shape[1]

        # Generate
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,  # greedy for reproducibility
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if return_logprobs:
            generate_kwargs["output_scores"] = True
            generate_kwargs["return_dict_in_generate"] = True

        with torch.no_grad():
            output = self.model.generate(input_ids, **generate_kwargs)

        # Extract generated tokens and decode
        if return_logprobs:
            generated_ids = output.sequences[0][input_len:]
            # Compute log probs from scores
            token_logprobs = []
            for i, score in enumerate(output.scores):
                log_probs = torch.nn.functional.log_softmax(score[0], dim=-1)
                token_id = generated_ids[i]
                token_logprobs.append(log_probs[token_id].item())
        else:
            generated_ids = output[0][input_len:]
            token_logprobs = None

        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            token_logprobs=token_logprobs,
            metadata={
                "model_path": self.model_path,
                "input_tokens": input_len,
                "output_tokens": len(generated_ids),
                "abstained": False,
            },
        )
