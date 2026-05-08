"""
Backward-compatibility shim.
Re-exports everything from reranker_updated and provides the original
function-based API (rerank_candidates, cosine_similarity) used by main.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.linalg import norm

# Re-export all classes/constants from the updated module
from .reranker_updated import *  # noqa: F401, F403

try:
    from logger import get_logger
    from metrics import VECTOR_LOOKUP_FAILURES
    _logger = get_logger(__name__)
    _METRICS = True
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)
    _METRICS = False

_RERANK_ENDPOINT = "/ml/search-rerank"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with zero-padding for mismatched dimensions."""
    max_dim = max(len(a), len(b))
    a_p, b_p = np.zeros(max_dim), np.zeros(max_dim)
    a_p[: len(a)] = a
    b_p[: len(b)] = b
    na, nb = norm(a_p), norm(b_p)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a_p, b_p) / (na * nb))


def rerank_candidates(
    user_vector: np.ndarray,
    candidates: List[Dict],
    product_vectors: Dict,
    weights: Optional[Dict] = None,
    search_intent: Optional[Dict] = None,
) -> List[Dict]:
    """
    Original function-based reranker used by /ml/search-rerank.
    Blends vector similarity, intent, and pricing signals.
    """
    if weights is None:
        weights = {"vector": 0.45, "intent": 0.30, "pricing": 0.25}

    vec_dim = len(next(iter(product_vectors.values()))) if product_vectors else len(user_vector)

    prices = [c.get("price") for c in candidates if c.get("price") is not None]
    min_price = min(prices) if prices else None
    max_price = max(prices) if prices else None

    intent_label = intent_confidence = None
    if search_intent:
        intent_label = search_intent.get("label")
        intent_confidence = float(search_intent.get("confidence", 0.0))

    results = []
    for candidate in candidates:
        product_id = candidate["product_id"]
        keyword_score = float(candidate.get("keyword_score") or 0.0)
        semantic_score = candidate.get("semantic_score")
        price = candidate.get("price")

        if _METRICS and product_id not in product_vectors:
            VECTOR_LOOKUP_FAILURES.labels(endpoint=_RERANK_ENDPOINT).inc()

        product_vec = product_vectors.get(product_id, np.zeros(vec_dim))
        cosine_score = cosine_similarity(user_vector, product_vec)

        sparse_score = keyword_score
        vector_score = 0.7 * cosine_score + 0.3 * sparse_score

        if price is None or min_price is None or max_price is None or max_price == min_price:
            pricing_score = 0.5
        else:
            norm_price = (price - min_price) / (max_price - min_price)
            if intent_label == "price_sensitive":
                pricing_score = 1.0 - norm_price
            elif intent_label == "premium":
                pricing_score = norm_price
            else:
                pricing_score = 0.5

        if intent_label in ("price_sensitive", "premium"):
            intent_alignment = pricing_score
        else:
            intent_alignment = float(semantic_score) if semantic_score is not None else sparse_score

        intent_score = (intent_confidence or 0.0) * intent_alignment

        final_score = (
            weights.get("vector", 0.45) * vector_score
            + weights.get("intent", 0.30) * intent_score
            + weights.get("pricing", 0.25) * pricing_score
        )

        results.append({
            "product_id":        product_id,
            "final_score":       float(final_score),
            "vector_score":      float(vector_score),
            "cosine_score":      float(cosine_score),
            "keyword_score":     float(sparse_score),
            "intent_score":      float(intent_score),
            "pricing_score":     float(pricing_score),
            "intent_label":      intent_label,
            "intent_confidence": float(intent_confidence or 0.0),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results
