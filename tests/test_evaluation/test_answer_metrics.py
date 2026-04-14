"""Tests for answer_metrics.py — SQuAD-canonical EM and token F1.

Reference: Official SQuAD v1.1/v2.0 evaluation scripts.
Normalization order: lower → remove_punc → remove_articles → white_space_fix.
F1: bag-of-words precision/recall over normalized tokens.
EM/F1 over multiple gold answers: take max.
"""

import pytest

from xrag.evaluation.answer_metrics import (
    normalize_answer,
    exact_match,
    token_f1,
    batch_em,
    batch_f1,
)


# =============================================================================
# normalize_answer — must exactly match SQuAD canonical script
# =============================================================================


class TestNormalizeAnswer:
    """Verify normalization matches the official SQuAD eval script exactly."""

    # --- Lowercasing ---

    def test_lowercases(self):
        assert normalize_answer("Barack Obama") == "barack obama"

    def test_all_caps(self):
        assert normalize_answer("NASA") == "nasa"

    def test_mixed_case(self):
        assert normalize_answer("McCarthyism") == "mccarthyism"

    # --- Punctuation removal ---

    def test_removes_period(self):
        assert normalize_answer("end.") == "end"

    def test_removes_comma(self):
        assert normalize_answer("a, b, c") == "b c"  # 'a' is article

    def test_removes_apostrophe(self):
        # SQuAD script removes apostrophe: "it's" → "its"
        assert normalize_answer("it's") == "its"

    def test_removes_hyphen(self):
        # SQuAD script removes hyphens: "well-known" → "wellknown"
        assert normalize_answer("well-known") == "wellknown"

    def test_removes_colon(self):
        assert normalize_answer("Answer: yes") == "answer yes"

    def test_removes_parentheses(self):
        assert normalize_answer("(hello)") == "hello"

    def test_removes_quotes(self):
        assert normalize_answer('"hello"') == "hello"

    def test_all_punctuation_removed(self):
        # Every character in string.punctuation should be removed
        assert normalize_answer("!@#$%^&*()") == ""

    # --- Article removal ---

    def test_removes_article_a(self):
        assert normalize_answer("a dog") == "dog"

    def test_removes_article_an(self):
        assert normalize_answer("an apple") == "apple"

    def test_removes_article_the(self):
        assert normalize_answer("the cat") == "cat"

    def test_removes_articles_case_insensitive(self):
        assert normalize_answer("The A An") == ""

    def test_does_not_remove_article_substring(self):
        # "the" inside "theorem" should NOT be removed (word boundary \b)
        assert normalize_answer("theorem") == "theorem"

    def test_does_not_remove_a_in_middle(self):
        # "a" inside "cat" should NOT be removed
        assert normalize_answer("cat") == "cat"

    def test_does_not_remove_an_in_word(self):
        assert normalize_answer("answer") == "answer"

    def test_article_at_end(self):
        # "the" at end of string should still be removed
        assert normalize_answer("give me the") == "give me"

    def test_multiple_articles(self):
        assert normalize_answer("the cat and a dog") == "cat and dog"

    # --- Whitespace normalization ---

    def test_collapses_multiple_spaces(self):
        assert normalize_answer("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert normalize_answer("  hello  ") == "hello"

    def test_tabs_and_newlines(self):
        assert normalize_answer("hello\tworld\n") == "hello world"

    # --- Combined normalization ---

    def test_squad_canonical_example(self):
        """The classic test: all four steps together."""
        assert normalize_answer("The   Quick, Brown Fox!") == "quick brown fox"

    def test_full_sentence(self):
        assert normalize_answer("It's a well-known fact.") == "its wellknown fact"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_only_articles_and_punctuation(self):
        assert normalize_answer("a, the, an!") == ""

    def test_numeric(self):
        assert normalize_answer("42") == "42"

    def test_numeric_with_punctuation(self):
        assert normalize_answer("1,000,000") == "1000000"

    def test_date_format(self):
        assert normalize_answer("January 1, 2020") == "january 1 2020"

    def test_unicode_preserved(self):
        """Non-ASCII characters should pass through (not in string.punctuation)."""
        assert normalize_answer("café") == "café"


# =============================================================================
# exact_match — max over gold answers, uses normalized comparison
# =============================================================================


class TestExactMatch:
    """EM = 1 iff normalize(pred) == normalize(gold) for at least one gold."""

    def test_exact_match_identical(self):
        assert exact_match("Barack Obama", ["Barack Obama"]) is True

    def test_exact_match_case_insensitive(self):
        assert exact_match("barack obama", ["Barack Obama"]) is True

    def test_exact_match_with_article(self):
        assert exact_match("the United States", ["United States"]) is True

    def test_exact_match_with_punctuation(self):
        assert exact_match("U.S.A.", ["USA"]) is True

    def test_no_match(self):
        assert exact_match("France", ["Germany"]) is False

    def test_multiple_golds_first_matches(self):
        assert exact_match("NYC", ["NYC", "New York City", "New York"]) is True

    def test_multiple_golds_last_matches(self):
        assert exact_match("New York", ["NYC", "New York City", "New York"]) is True

    def test_multiple_golds_none_match(self):
        assert exact_match("Boston", ["NYC", "New York City", "New York"]) is False

    def test_partial_match_is_not_em(self):
        assert exact_match("Barack", ["Barack Obama"]) is False

    def test_superset_is_not_em(self):
        assert exact_match("President Barack Obama", ["Barack Obama"]) is False

    def test_empty_prediction_empty_gold(self):
        """Both empty should match (SQuAD v2 unanswerable convention)."""
        assert exact_match("", [""]) is True

    def test_empty_prediction_nonempty_gold(self):
        assert exact_match("", ["Barack Obama"]) is False

    def test_nonempty_prediction_empty_gold(self):
        assert exact_match("Barack Obama", [""]) is False

    def test_whitespace_normalization_makes_match(self):
        assert exact_match("Barack  Obama", ["Barack Obama"]) is True

    def test_gold_list_must_be_nonempty(self):
        """An empty gold_answers list should raise, not silently return."""
        with pytest.raises((ValueError, TypeError)):
            exact_match("anything", [])

    def test_returns_bool(self):
        result = exact_match("test", ["test"])
        assert isinstance(result, bool)


# =============================================================================
# token_f1 — bag-of-words F1 over normalized tokens, max over golds
# =============================================================================


class TestTokenF1:
    """F1 = 2*P*R/(P+R) over normalized token bags, max over gold answers."""

    def test_perfect_match(self):
        assert token_f1("Barack Obama", ["Barack Obama"]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("France", ["Germany"]) == pytest.approx(0.0)

    def test_partial_overlap_precision_recall(self):
        # pred: "Barack Hussein Obama" → tokens: [barack, hussein, obama]
        # gold: "Barack Obama" → tokens: [barack, obama]
        # common = 2, pred_len = 3, gold_len = 2
        # P = 2/3, R = 2/2 = 1.0, F1 = 2*(2/3)*1/(2/3+1) = (4/3)/(5/3) = 4/5 = 0.8
        assert token_f1("Barack Hussein Obama", ["Barack Obama"]) == pytest.approx(0.8)

    def test_partial_overlap_reverse(self):
        # pred: "Barack Obama" → [barack, obama]
        # gold: "Barack Hussein Obama" → [barack, hussein, obama]
        # common = 2, P = 2/2 = 1.0, R = 2/3, F1 = 2*1*(2/3)/(1+2/3) = (4/3)/(5/3) = 4/5 = 0.8
        assert token_f1("Barack Obama", ["Barack Hussein Obama"]) == pytest.approx(0.8)

    def test_single_token_match(self):
        # pred: "Obama" → [obama], gold: "Barack Obama" → [barack, obama]
        # common = 1, P = 1/1 = 1.0, R = 1/2 = 0.5, F1 = 2*1*0.5/1.5 = 2/3
        assert token_f1("Obama", ["Barack Obama"]) == pytest.approx(2 / 3)

    def test_max_over_gold_answers(self):
        # gold1: "NYC" → [nyc], gold2: "New York City" → [new, york, city]
        # pred: "New York City"
        # F1 with gold1: pred=[new,york,city], gold=[nyc], common=0 → 0.0
        # F1 with gold2: pred=[new,york,city], gold=[new,york,city], common=3 → 1.0
        assert token_f1("New York City", ["NYC", "New York City"]) == pytest.approx(1.0)

    def test_articles_removed_before_scoring(self):
        # "The United States" normalizes to "united states"
        # "United States" normalizes to "united states"
        assert token_f1("The United States", ["United States"]) == pytest.approx(1.0)

    def test_punctuation_removed_before_scoring(self):
        assert token_f1("U.S.A.", ["USA"]) == pytest.approx(1.0)

    def test_empty_prediction_nonempty_gold(self):
        # SQuAD convention: if pred is empty and gold is not, F1 = 0
        assert token_f1("", ["Barack Obama"]) == pytest.approx(0.0)

    def test_nonempty_prediction_empty_gold(self):
        # SQuAD convention: if gold is empty and pred is not, F1 = 0
        assert token_f1("Barack Obama", [""]) == pytest.approx(0.0)

    def test_both_empty(self):
        # SQuAD v2 convention: both empty → F1 = 1.0 (correct abstention)
        assert token_f1("", [""]) == pytest.approx(1.0)

    def test_duplicate_tokens_in_prediction(self):
        # pred: "the the cat" → normalize → "cat" (both "the" removed)
        # gold: "cat" → "cat"
        assert token_f1("the the cat", ["cat"]) == pytest.approx(1.0)

    def test_repeated_words_counted_correctly(self):
        # pred: "very very good" → [very, very, good]
        # gold: "very good" → [very, good]
        # Counter intersection: {very:1, good:1} = 2
        # P = 2/3, R = 2/2 = 1.0, F1 = (4/3)/(5/3) = 0.8
        assert token_f1("very very good", ["very good"]) == pytest.approx(0.8)

    def test_returns_float(self):
        result = token_f1("test", ["test"])
        assert isinstance(result, float)

    def test_f1_bounded_0_1(self):
        """F1 should always be in [0, 1]."""
        cases = [
            ("Barack Obama", ["Barack Obama"]),
            ("totally wrong", ["Barack Obama"]),
            ("Barack", ["Barack Obama"]),
            ("President Barack Hussein Obama Jr.", ["Obama"]),
        ]
        for pred, golds in cases:
            f1 = token_f1(pred, golds)
            assert 0.0 <= f1 <= 1.0, f"F1={f1} out of bounds for pred={pred}, golds={golds}"

    def test_gold_list_must_be_nonempty(self):
        with pytest.raises((ValueError, TypeError)):
            token_f1("anything", [])

    def test_symmetry_of_f1_single_gold(self):
        """F1 is symmetric: F1(A, B) == F1(B, A) for single gold."""
        f1_ab = token_f1("quick brown fox", ["lazy brown dog"])
        f1_ba = token_f1("lazy brown dog", ["quick brown fox"])
        assert f1_ab == pytest.approx(f1_ba)


# =============================================================================
# batch_em / batch_f1 — aggregation over datasets
# =============================================================================


class TestBatchEM:
    """batch_em returns mean EM over a dataset."""

    def test_all_correct(self):
        preds = ["Barack Obama", "Paris", "42"]
        golds = [["Barack Obama"], ["Paris"], ["42"]]
        assert batch_em(preds, golds) == pytest.approx(1.0)

    def test_all_wrong(self):
        preds = ["wrong1", "wrong2", "wrong3"]
        golds = [["right1"], ["right2"], ["right3"]]
        assert batch_em(preds, golds) == pytest.approx(0.0)

    def test_half_correct(self):
        preds = ["Barack Obama", "wrong", "42", "wrong"]
        golds = [["Barack Obama"], ["Paris"], ["42"], ["London"]]
        assert batch_em(preds, golds) == pytest.approx(0.5)

    def test_single_example(self):
        assert batch_em(["test"], [["test"]]) == pytest.approx(1.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises((ValueError, TypeError)):
            batch_em(["a", "b"], [["a"]])

    def test_empty_raises(self):
        with pytest.raises((ValueError, TypeError)):
            batch_em([], [])

    def test_returns_float(self):
        result = batch_em(["test"], [["test"]])
        assert isinstance(result, float)

    def test_multiple_gold_answers(self):
        """Each example can have multiple acceptable gold answers."""
        preds = ["NYC", "The Big Apple"]
        golds = [
            ["New York City", "NYC", "New York"],
            ["Big Apple", "The Big Apple"],
        ]
        assert batch_em(preds, golds) == pytest.approx(1.0)


class TestBatchF1:
    """batch_f1 returns mean token F1 over a dataset."""

    def test_all_perfect(self):
        preds = ["Barack Obama", "Paris"]
        golds = [["Barack Obama"], ["Paris"]]
        assert batch_f1(preds, golds) == pytest.approx(1.0)

    def test_all_zero(self):
        preds = ["xyz", "abc"]
        golds = [["Barack Obama"], ["Paris"]]
        assert batch_f1(preds, golds) == pytest.approx(0.0)

    def test_mixed_f1_averaged(self):
        # Example 1: perfect match → F1 = 1.0
        # Example 2: no overlap → F1 = 0.0
        preds = ["Barack Obama", "xyz"]
        golds = [["Barack Obama"], ["Paris"]]
        assert batch_f1(preds, golds) == pytest.approx(0.5)

    def test_mismatched_lengths_raises(self):
        with pytest.raises((ValueError, TypeError)):
            batch_f1(["a", "b"], [["a"]])

    def test_empty_raises(self):
        with pytest.raises((ValueError, TypeError)):
            batch_f1([], [])

    def test_returns_float(self):
        result = batch_f1(["test"], [["test"]])
        assert isinstance(result, float)

    def test_result_bounded_0_1(self):
        preds = ["quick brown", "totally wrong", "exact"]
        golds = [["quick brown fox"], ["something else"], ["exact match"]]
        result = batch_f1(preds, golds)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Regression tests — known values from SQuAD-like benchmarks
# =============================================================================


class TestRegressionKnownValues:
    """
    Hand-computed test cases to guard against regressions.
    If any of these fail, our metrics diverge from the standard.
    """

    def test_squad_style_normalization_regression(self):
        """Verify exact normalization matches SQuAD eval script output."""
        # From the SQuAD eval script behavior:
        # Input: "The quick, brown Fox!"
        # lower: "the quick, brown fox!"
        # remove_punc: "the quick brown fox"
        # remove_articles: " quick brown fox"  (the → space)
        # white_space_fix: "quick brown fox"
        assert normalize_answer("The quick, brown Fox!") == "quick brown fox"

    def test_nq_style_answer(self):
        """Natural Questions often have full-sentence gold answers."""
        pred = "January 20, 2009"
        gold = ["January 20, 2009", "January 20 2009", "20 January 2009"]
        assert exact_match(pred, gold) is True

    def test_hotpotqa_bridge_answer(self):
        """HotpotQA bridge questions have entity answers."""
        pred = "the United Kingdom"
        gold = ["United Kingdom", "UK", "the United Kingdom"]
        assert exact_match(pred, gold) is True
        assert token_f1(pred, gold) == pytest.approx(1.0)

    def test_f1_hand_computed(self):
        """
        pred: "the cat sat on the mat" → normalize → "cat sat on mat"
        gold: "a cat sat on a rug"   → normalize → "cat sat on rug"
        pred_tokens = [cat, sat, on, mat]   (4 tokens)
        gold_tokens = [cat, sat, on, rug]   (4 tokens)
        common = {cat:1, sat:1, on:1} = 3
        P = 3/4 = 0.75, R = 3/4 = 0.75
        F1 = 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
        """
        f1 = token_f1("the cat sat on the mat", ["a cat sat on a rug"])
        assert f1 == pytest.approx(0.75)

    def test_em_with_normalization_makes_match(self):
        """
        pred: "The U.S.A."
        gold: "USA"
        normalize(pred) = "usa"
        normalize(gold) = "usa"
        → EM = True
        """
        assert exact_match("The U.S.A.", ["USA"]) is True

    def test_f1_completely_disjoint(self):
        """No shared tokens → F1 = 0."""
        assert token_f1("alpha beta gamma", ["delta epsilon zeta"]) == pytest.approx(0.0)

    def test_batch_em_known_dataset(self):
        """Simulate a mini benchmark run with known result."""
        preds = [
            "Barack Obama",     # EM=1 (matches gold 1)
            "the white house",  # EM=1 (articles removed → "white house")
            "Germany",          # EM=0 (gold is France)
            "",                 # EM=1 (unanswerable, gold is "")
            "42",               # EM=1 (matches)
        ]
        golds = [
            ["Barack Obama", "Obama"],
            ["White House", "The White House"],
            ["France"],
            [""],
            ["42", "forty-two"],
        ]
        # 4 out of 5 correct → 0.8
        assert batch_em(preds, golds) == pytest.approx(0.8)
