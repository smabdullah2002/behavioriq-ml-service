"""TF-IDF embedder for product vectors and user vectors."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List

from logger import get_logger

logger = get_logger(__name__)


class ProductEmbedder:
    """Builds and manages TF-IDF product embeddings."""
    
    def __init__(self):
        """Initialize the embedder."""
        self.vectorizer = None
        self.product_vectors = {}  # dict: product_id -> vector
        self.corpus = []  # list of product texts
    
    def fit(self, products: List[Dict]) -> None:
        """
        Fit the vectorizer on product corpus.
        
        Args:
            products: list of dicts with 'id', 'name', 'desc', 'category'
        """
        self.corpus = [
            f"{p.get('name', '')} {p.get('desc', '')} {p.get('category', '')}"
            for p in products
        ]
        self.vectorizer = TfidfVectorizer(max_features=100)
        vectors = self.vectorizer.fit_transform(self.corpus)
        
        for i, p in enumerate(products):
            self.product_vectors[p['id']] = vectors[i].toarray().flatten()

        logger.info(
            "Embedder fitted | products=%d | vocab_size=%d",
            len(products),
            len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
        )
    
    def _dim(self) -> int:
        """Return actual vector dimension from fitted data."""
        if self.product_vectors:
            return len(next(iter(self.product_vectors.values())))
        return 100

    def get_product_vector(self, product_id: str) -> np.ndarray:
        """Get pre-computed vector for a product."""
        return self.product_vectors.get(product_id, np.zeros(self._dim()))

    def build_user_vector(self, product_ids: List[str],
                          weights: List[float] = None) -> np.ndarray:
        """
        Build user behavioral vector from product vectors.

        Args:
            product_ids: list of recently viewed product IDs
            weights: optional per-product weights

        Returns:
            user vector as numpy array
        """
        if not product_ids:
            return np.zeros(self._dim())

        if weights is None:
            weights = [1.0] * len(product_ids)

        weighted_sum = np.zeros(self._dim())
        total_weight = 0
        
        for pid, w in zip(product_ids, weights):
            vec = self.get_product_vector(pid)
            weighted_sum += vec * w
            total_weight += w
        
        if total_weight == 0:
            return np.zeros(100)
        
        return weighted_sum / total_weight
