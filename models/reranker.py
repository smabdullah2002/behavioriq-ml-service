"""Search re-ranker using user vectors and keyword scores."""

import numpy as np
from numpy.linalg import norm
from typing import List, Dict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def rerank_candidates(user_vector: np.ndarray,
                      candidates: List[Dict],
                      product_vectors: Dict,
                      weights: Dict = None) -> List[Dict]:
    """
    Re-rank search candidates using user vector and blended scoring.
    
    Args:
        user_vector: user behavioral embedding
        candidates: list of {"product_id": str, "keyword_score": float}
        product_vectors: dict mapping product_id to vector
        weights: blend weights {"kw": 0.5, "cosine": 0.3, "popularity": 0.2}
    
    Returns:
        list of {"product_id": str, "final_score": float} sorted by score
    """
    if weights is None:
        weights = {"kw": 0.5, "cosine": 0.3, "popularity": 0.2}
    
    results = []
    
    for candidate in candidates:
        product_id = candidate["product_id"]
        keyword_score = candidate.get("keyword_score", 0.5)
        popularity_score = candidate.get("popularity_score", 0.5)
        
        # Get product vector
        product_vec = product_vectors.get(product_id, np.zeros(len(user_vector)))
        
        # Compute cosine similarity
        cosine_score = cosine_similarity(user_vector, product_vec)
        
        # Blend scores
        final_score = (
            weights.get("kw", 0.5) * keyword_score +
            weights.get("cosine", 0.3) * cosine_score +
            weights.get("popularity", 0.2) * popularity_score
        )
        
        results.append({
            "product_id": product_id,
            "final_score": float(final_score)
        })
    
    # Sort by final_score descending
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results
