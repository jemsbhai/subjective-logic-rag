"""Tests for faithfulness_metrics.py — claim decomposition and entailment scoring.

Two decomposition methods:
    1. "sentence" — nltk sentence tokenizer (no model needed)
    2. "llm" — LLM-based atomic claim decomposition (callable interface)

Faithfulness precision = fraction of claims entailed by at least one evidence passage.

The entailment function is injected as a callable, keeping this module
decoupled from any specific NLI model.
"""

import pytest

from xrag.evaluation.faithfulness_metrics import (
    decompose_claims,
    faithfulness_precision,
    faithfulness_result,
    FaithfulnessResult,
)


# =============================================================================
# decompose_claims — sentence splitting
# =============================================================================


class TestDecomposeClaimsSentence:
    """Sentence-based claim decomposition using nltk."""

    def test_single_sentence(self):
        text = "Barack Obama was the 44th president."
        claims = decompose_claims(text, method="sentence")
        assert len(claims) == 1
        assert claims[0] == "Barack Obama was the 44th president."

    def test_two_sentences(self):
        text = "The sky is blue. Water is wet."
        claims = decompose_claims(text, method="sentence")
        assert len(claims) == 2
        assert claims[0] == "The sky is blue."
        assert claims[1] == "Water is wet."

    def test_three_sentences(self):
        text = "First claim. Second claim. Third claim."
        claims = decompose_claims(text, method="sentence")
        assert len(claims) == 3

    def test_empty_string(self):
        claims = decompose_claims("", method="sentence")
        assert claims == []

    def test_whitespace_only(self):
        claims = decompose_claims("   \n\t  ", method="sentence")
        assert claims == []

    def test_strips_whitespace_from_claims(self):
        text = "  First claim.   Second claim.  "
        claims = decompose_claims(text, method="sentence")
        for claim in claims:
            assert claim == claim.strip()
            assert len(claim) > 0

    def test_handles_question_marks(self):
        text = "Is this a question? Yes, it is."
        claims = decompose_claims(text, method="sentence")
        assert len(claims) == 2

    def test_handles_exclamation(self):
        text = "This is great! Really impressive."
        claims = decompose_claims(text, method="sentence")
        assert len(claims) == 2

    def test_abbreviations_handled(self):
        """Common abbreviations like 'Dr.' or 'U.S.' should not split."""
        text = "Dr. Smith works at the U.S. embassy. He is a diplomat."
        claims = decompose_claims(text, method="sentence")
        # Should be 2, not more (nltk handles common abbreviations)
        assert len(claims) == 2

    def test_returns_list_of_strings(self):
        claims = decompose_claims("Test sentence.", method="sentence")
        assert isinstance(claims, list)
        assert all(isinstance(c, str) for c in claims)

    def test_multiline_input(self):
        text = "First line.\nSecond line.\nThird line."
        claims = decompose_claims(text, method="sentence")
        assert len(claims) == 3

    def test_no_trailing_empty_claims(self):
        """Should not produce empty strings as claims."""
        text = "A sentence. Another one. "
        claims = decompose_claims(text, method="sentence")
        assert all(len(c) > 0 for c in claims)


# =============================================================================
# decompose_claims — LLM-based
# =============================================================================


class TestDecomposeClaimsLLM:
    """LLM-based claim decomposition using a callable."""

    def test_uses_provided_function(self):
        """The decompose_fn should be called with the answer text."""
        called_with = []

        def mock_decompose(text: str) -> list[str]:
            called_with.append(text)
            return ["claim 1", "claim 2"]

        result = decompose_claims(
            "The answer text.", method="llm", decompose_fn=mock_decompose
        )
        assert called_with == ["The answer text."]
        assert result == ["claim 1", "claim 2"]

    def test_returns_fn_output(self):
        def mock_fn(text: str) -> list[str]:
            return ["atomic claim A", "atomic claim B", "atomic claim C"]

        result = decompose_claims("anything", method="llm", decompose_fn=mock_fn)
        assert len(result) == 3

    def test_requires_decompose_fn(self):
        """method='llm' without decompose_fn should raise."""
        with pytest.raises((ValueError, TypeError)):
            decompose_claims("text", method="llm")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="[Mm]ethod"):
            decompose_claims("text", method="invalid")


# =============================================================================
# faithfulness_precision — entailment scoring
# =============================================================================


class TestFaithfulnessPrecision:
    """Faithfulness precision = fraction of claims entailed by evidence."""

    def _always_entails(self, premise: str, hypothesis: str) -> bool:
        return True

    def _never_entails(self, premise: str, hypothesis: str) -> bool:
        return False

    def test_all_claims_supported(self):
        claims = ["claim 1", "claim 2", "claim 3"]
        evidence = ["evidence passage"]
        result = faithfulness_precision(claims, evidence, self._always_entails)
        assert result == pytest.approx(1.0)

    def test_no_claims_supported(self):
        claims = ["claim 1", "claim 2", "claim 3"]
        evidence = ["evidence passage"]
        result = faithfulness_precision(claims, evidence, self._never_entails)
        assert result == pytest.approx(0.0)

    def test_partial_support(self):
        """2 of 4 claims supported → precision = 0.5."""
        call_count = [0]

        def selective_entails(premise: str, hypothesis: str) -> bool:
            call_count[0] += 1
            return hypothesis in ("claim 1", "claim 3")

        claims = ["claim 1", "claim 2", "claim 3", "claim 4"]
        evidence = ["evidence"]
        result = faithfulness_precision(claims, evidence, selective_entails)
        assert result == pytest.approx(0.5)

    def test_claim_supported_by_any_evidence(self):
        """A claim is supported if ANY evidence passage entails it."""
        def first_evidence_only(premise: str, hypothesis: str) -> bool:
            return premise == "evidence 1"

        claims = ["claim A"]
        evidence = ["evidence 1", "evidence 2", "evidence 3"]
        result = faithfulness_precision(claims, evidence, first_evidence_only)
        assert result == pytest.approx(1.0)

    def test_each_claim_checked_against_all_evidence(self):
        """Verify that each claim is checked against every evidence passage."""
        checked_pairs = []

        def tracking_entails(premise: str, hypothesis: str) -> bool:
            checked_pairs.append((premise, hypothesis))
            return False

        claims = ["c1", "c2"]
        evidence = ["e1", "e2", "e3"]
        faithfulness_precision(claims, evidence, tracking_entails)
        # Should have checked 2 claims × 3 evidence = 6 pairs
        assert len(checked_pairs) == 6
        # Verify all pairs present
        for c in claims:
            for e in evidence:
                assert (e, c) in checked_pairs

    def test_short_circuits_on_first_entailment(self):
        """Once a claim is supported by one passage, skip remaining passages."""
        call_count = [0]

        def counting_entails(premise: str, hypothesis: str) -> bool:
            call_count[0] += 1
            return premise == "e1"  # First evidence always entails

        claims = ["claim"]
        evidence = ["e1", "e2", "e3"]
        result = faithfulness_precision(claims, evidence, counting_entails)
        assert result == pytest.approx(1.0)
        # Should stop after finding entailment from e1
        assert call_count[0] == 1

    def test_empty_claims(self):
        """No claims → precision = 1.0 (vacuous truth: all zero claims are supported)."""
        result = faithfulness_precision([], ["evidence"], self._never_entails)
        assert result == pytest.approx(1.0)

    def test_empty_evidence(self):
        """No evidence → nothing can entail → precision = 0.0."""
        result = faithfulness_precision(["claim"], [], self._always_entails)
        assert result == pytest.approx(0.0)

    def test_both_empty(self):
        """No claims, no evidence → vacuous truth."""
        result = faithfulness_precision([], [], self._never_entails)
        assert result == pytest.approx(1.0)

    def test_returns_float(self):
        result = faithfulness_precision(["c"], ["e"], self._always_entails)
        assert isinstance(result, float)

    def test_bounded_0_1(self):
        result = faithfulness_precision(
            ["c1", "c2", "c3"], ["e1"], self._always_entails
        )
        assert 0.0 <= result <= 1.0


# =============================================================================
# faithfulness_result — full result with per-claim verdicts
# =============================================================================


class TestFaithfulnessResult:
    """Full faithfulness evaluation returning per-claim details."""

    def _keyword_entails(self, premise: str, hypothesis: str) -> bool:
        """Simple mock: entails if any word from hypothesis appears in premise."""
        hyp_words = set(hypothesis.lower().split())
        prem_words = set(premise.lower().split())
        return len(hyp_words & prem_words) > 0

    def test_returns_dataclass(self):
        result = faithfulness_result(
            answer="The sky is blue.",
            evidence=["The sky appears blue due to Rayleigh scattering."],
            entailment_fn=self._keyword_entails,
            method="sentence",
        )
        assert isinstance(result, FaithfulnessResult)

    def test_has_required_fields(self):
        result = faithfulness_result(
            answer="Claim one. Claim two.",
            evidence=["evidence"],
            entailment_fn=self._keyword_entails,
            method="sentence",
        )
        assert hasattr(result, "claims")
        assert hasattr(result, "per_claim_supported")
        assert hasattr(result, "precision")
        assert hasattr(result, "method")

    def test_per_claim_verdicts_length(self):
        result = faithfulness_result(
            answer="First. Second. Third.",
            evidence=["evidence"],
            entailment_fn=self._keyword_entails,
            method="sentence",
        )
        assert len(result.claims) == 3
        assert len(result.per_claim_supported) == 3

    def test_per_claim_verdicts_are_bools(self):
        result = faithfulness_result(
            answer="Supported claim. Unsupported claim.",
            evidence=["supported"],
            entailment_fn=self._keyword_entails,
            method="sentence",
        )
        assert all(isinstance(v, bool) for v in result.per_claim_supported)

    def test_precision_matches_verdicts(self):
        """Precision should equal mean of per_claim_supported."""

        def half_entails(premise: str, hypothesis: str) -> bool:
            return "first" in hypothesis.lower()

        result = faithfulness_result(
            answer="First claim. Second claim.",
            evidence=["anything"],
            entailment_fn=half_entails,
            method="sentence",
        )
        expected_precision = sum(result.per_claim_supported) / len(
            result.per_claim_supported
        )
        assert result.precision == pytest.approx(expected_precision)

    def test_method_recorded(self):
        result = faithfulness_result(
            answer="Claim.",
            evidence=["evidence"],
            entailment_fn=self._keyword_entails,
            method="sentence",
        )
        assert result.method == "sentence"

    def test_llm_method_with_decompose_fn(self):
        def mock_decompose(text: str) -> list[str]:
            return ["atomic claim 1", "atomic claim 2"]

        result = faithfulness_result(
            answer="Complex answer.",
            evidence=["atomic claim 1 is in the evidence"],
            entailment_fn=self._keyword_entails,
            method="llm",
            decompose_fn=mock_decompose,
        )
        assert result.method == "llm"
        assert len(result.claims) == 2

    def test_empty_answer(self):
        result = faithfulness_result(
            answer="",
            evidence=["evidence"],
            entailment_fn=self._keyword_entails,
            method="sentence",
        )
        assert result.claims == []
        assert result.per_claim_supported == []
        assert result.precision == pytest.approx(1.0)  # vacuous
