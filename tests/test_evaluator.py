"""Tests for benchmark/evaluator.py — score_answer, token_f1, keyword_coverage."""
from __future__ import annotations

import pytest

from benchmark.evaluator import (
    _normalize,
    _tokens,
    token_f1,
    keyword_coverage,
    score_answer,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------
class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert _normalize("cat, dog!") == "cat  dog "

    def test_keeps_digits(self):
        result = _normalize("abc 123")
        assert "123" in result

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_only_punctuation(self):
        result = _normalize("!!!???")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# _tokens
# ---------------------------------------------------------------------------
class TestTokens:
    def test_basic_extraction(self):
        tokens = _tokens("The quick brown fox")
        # 'the' is a stopword; 'quick', 'brown', 'fox' should remain
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "the" not in tokens

    def test_stopwords_removed(self):
        tokens = _tokens("a an the is are was were")
        assert tokens == set()

    def test_empty_string(self):
        assert _tokens("") == set()

    def test_all_stopwords(self):
        assert _tokens("is the a an") == set()

    def test_punctuation_stripped(self):
        tokens = _tokens("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------
class TestTokenF1:
    def test_identical_strings(self):
        assert token_f1("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("apple banana cherry", "delta echo foxtrot") == pytest.approx(0.0)

    def test_partial_overlap(self):
        f1 = token_f1("quick brown fox", "quick brown dog")
        assert 0.0 < f1 < 1.0

    def test_empty_prediction(self):
        assert token_f1("", "some reference text") == pytest.approx(0.0)

    def test_empty_reference(self):
        assert token_f1("some prediction text", "") == pytest.approx(0.0)

    def test_both_empty(self):
        assert token_f1("", "") == pytest.approx(0.0)

    def test_all_stopwords_prediction(self):
        # After removing stopwords, pred_tokens is empty → 0.0
        assert token_f1("is the a", "quick brown fox") == pytest.approx(0.0)

    def test_symmetry(self):
        a = "machine learning model training"
        b = "deep learning neural network training"
        # F1 is symmetric
        assert token_f1(a, b) == pytest.approx(token_f1(b, a))

    def test_perfect_subset(self):
        # prediction has exactly the tokens of reference → recall = 1
        f1 = token_f1("brown fox", "quick brown fox")
        assert 0.0 < f1 <= 1.0

    def test_f1_formula(self):
        # Manually verify the formula
        pred = _tokens("quick brown fox")
        ref = _tokens("quick brown dog")
        common = pred & ref
        precision = len(common) / len(pred)
        recall = len(common) / len(ref)
        expected = 2 * precision * recall / (precision + recall)
        assert token_f1("quick brown fox", "quick brown dog") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# keyword_coverage
# ---------------------------------------------------------------------------
class TestKeywordCoverage:
    def test_all_keywords_found(self):
        assert keyword_coverage("Python machine learning framework", ["python", "learning"]) == pytest.approx(1.0)

    def test_no_keywords_found(self):
        assert keyword_coverage("completely irrelevant text", ["python", "machine"]) == pytest.approx(0.0)

    def test_partial_keywords(self):
        cov = keyword_coverage("Python is great", ["python", "java"])
        assert cov == pytest.approx(0.5)

    def test_empty_keywords_list(self):
        # No keywords → perfect coverage by definition
        assert keyword_coverage("any text here", []) == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert keyword_coverage("Python Machine Learning", ["python", "machine"]) == pytest.approx(1.0)

    def test_keyword_not_whole_word(self):
        # 'learn' should match inside 'learning'
        cov = keyword_coverage("deep learning approach", ["learn"])
        assert cov == pytest.approx(1.0)

    def test_multiple_keywords_all_missing(self):
        assert keyword_coverage("hello world", ["python", "java", "rust"]) == pytest.approx(0.0)

    def test_single_keyword_found(self):
        assert keyword_coverage("using Python for AI", ["python"]) == pytest.approx(1.0)

    def test_single_keyword_not_found(self):
        assert keyword_coverage("using Rust for systems", ["python"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_answer
# ---------------------------------------------------------------------------
class TestScoreAnswer:
    def test_no_keywords_returns_f1(self):
        pred = "machine learning model"
        ref = "machine learning model"
        score = score_answer(pred, ref)
        assert score == pytest.approx(token_f1(pred, ref))

    def test_with_keywords_combined(self):
        pred = "Python machine learning framework"
        ref = "machine learning Python"
        keywords = ["python", "learning"]
        score = score_answer(pred, ref, keywords)
        f1 = token_f1(pred, ref)
        kw = keyword_coverage(pred, keywords)
        expected = 0.6 * f1 + 0.4 * kw
        assert score == pytest.approx(expected)

    def test_score_in_range(self):
        score = score_answer("hello world test", "test hello", ["hello"])
        assert 0.0 <= score <= 1.0

    def test_empty_keywords_list_behaves_as_no_keywords(self):
        # keywords=[] is falsy → pure F1
        pred = "quick brown fox"
        ref = "quick brown fox"
        score_none = score_answer(pred, ref, None)
        score_empty = score_answer(pred, ref, [])
        assert score_none == pytest.approx(score_empty)

    def test_perfect_prediction(self):
        pred = "the neural network trained successfully on the dataset"
        ref = pred
        score = score_answer(pred, ref)
        assert score == pytest.approx(1.0)

    def test_completely_wrong_prediction(self):
        score = score_answer("apple banana cherry", "delta echo foxtrot", ["delta"])
        # F1 = 0.0, keyword coverage = 0.0
        assert score == pytest.approx(0.0)

    def test_weights_sum_correctly(self):
        pred = "neural network training"
        ref = "neural network architecture"
        keywords = ["neural", "training"]
        f1 = token_f1(pred, ref)
        kw = keyword_coverage(pred, keywords)
        expected = 0.6 * f1 + 0.4 * kw
        assert score_answer(pred, ref, keywords) == pytest.approx(expected)
