from __future__ import annotations

import re
from typing import List, Set


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def _tokens(text: str) -> Set[str]:
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "out", "off", "over", "under",
        "again", "further", "then", "once", "and", "but", "or", "nor",
        "so", "yet", "both", "either", "neither", "not", "only", "own",
        "same", "than", "too", "very", "just", "because", "if", "while",
        "that", "this", "these", "those", "it", "its", "they", "them",
        "their", "which", "who", "whom", "what", "when", "where", "how",
        "all", "each", "every", "more", "most", "other", "such", "no",
        "also", "i", "you", "he", "she", "we",
    }
    return {t for t in _normalize(text).split() if t and t not in stopwords}


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (ignoring stopwords)."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(prediction: str, keywords: List[str]) -> float:
    """Fraction of expected keywords found in the prediction."""
    if not keywords:
        return 1.0
    pred_norm = _normalize(prediction)
    found = sum(1 for kw in keywords if _normalize(kw) in pred_norm)
    return found / len(keywords)


def score_answer(prediction: str, reference: str, keywords: List[str] | None = None) -> float:
    """
    Combined accuracy score in [0, 1].
    60% token F1 + 40% keyword coverage (if keywords provided, else pure F1).
    """
    f1 = token_f1(prediction, reference)
    if keywords:
        kw = keyword_coverage(prediction, keywords)
        return 0.6 * f1 + 0.4 * kw
    return f1
