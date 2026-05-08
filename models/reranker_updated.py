"""
BehaviorIQ Reranker — Updated Full Pipeline

Pipeline:
  raw query
    -> Step 1: Semantic + keyword search  (Pinecone hybrid: dense + sparse)
    -> Step 2: Search intent analysis     (rule-based, confidence-scored)
    -> Step 3: BehaviorIQ reranker        (intent + vectors + pricing + churn)

Run standalone:
    python reranker_updated.py
    # or:
    uvicorn reranker_updated:app --port 8002 --reload

Example request:
    POST /search
    {"query": "cheap running shoes", "churn_score": 0.8}
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("behaviouriq.reranker_updated")

# ── Environment (reads from ml-service/.env — one level up from models/) ─────
_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

PINECONE_API_KEY: str = os.getenv("PINECONE_API", "")
HF_API_KEY: str = os.getenv("HF_API_KEY", "")
OFFLINE_MODE: bool = not bool(PINECONE_API_KEY and HF_API_KEY)

# ── Constants ─────────────────────────────────────────────────────────────────
INDEX_NAME = "behavioriq-products"
EMBEDDING_DIM = 384          # sentence-transformers/all-MiniLM-L6-v2

HF_EMBED_URL = (
    "https://api-inference.huggingface.co/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Base scoring weights (config-driven; intent overrides applied at runtime)
BASE_WEIGHTS: Dict[str, float] = {"vector": 0.45, "intent": 0.30, "pricing": 0.25}

# Within vector_score: how much dense vs. sparse contributes
W_DENSE: float = 0.70
W_SPARSE: float = 0.30

INTENT_CONFIDENCE_THRESHOLD: float = 0.40  # below → fall back to exploratory
TOP_CANDIDATES: int = 100                   # retrieved from Pinecone before rerank

# Per-intent weight overrides — shifts emphasis when intent is strong
_INTENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "price_sensitive": {"vector": 0.35, "intent": 0.25, "pricing": 0.40},
    "premium":         {"vector": 0.40, "intent": 0.25, "pricing": 0.35},
    "urgent_buy":      {"vector": 0.45, "intent": 0.40, "pricing": 0.15},
    "gift":            {"vector": 0.50, "intent": 0.35, "pricing": 0.15},
    "comparison":      {"vector": 0.55, "intent": 0.30, "pricing": 0.15},
    "exploratory":     {"vector": 0.50, "intent": 0.25, "pricing": 0.25},
}

# Keyword patterns for intent detection
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "price_sensitive": [
        "cheap", "cheapest", "affordable", "budget", "low price", "low-price",
        "inexpensive", "discount", "sale", "bargain", "value", "economical",
        "cost-effective", "on sale", "deal", "least expensive", "low cost",
    ],
    "premium": [
        "premium", "luxury", "high-end", "best quality", "top rated", "elite",
        "professional", "pro", "high quality", "finest", "top tier", "best",
    ],
    "comparison": [
        "compare", "vs", "versus", "difference between", "better than",
        "alternative", "which is better", "best option",
    ],
    "gift": [
        "gift", "present", "birthday", "anniversary", "for him", "for her",
        "surprise", "gifting",
    ],
    "urgent_buy": [
        "urgent", "fast shipping", "quick", "same day", "overnight",
        "asap", "right now", "immediately", "today",
    ],
}

# Category keyword → category slug for slot extraction
_CATEGORY_HINTS: Dict[str, str] = {
    "running shoe": "running_shoes",
    "running shoes": "running_shoes",
    "trail shoe": "trail_shoes",
    "trail shoes": "trail_shoes",
    "casual shoe": "casual_shoes",
    "sneaker": "casual_shoes",
    "sandal": "sandals",
    "shirt": "sports_shirt",
    "tee": "sports_shirt",
    "shorts": "sports_shorts",
    "tights": "sports_pants",
    "legging": "sports_pants",
    "kids": "kids_shoes",
}

_KNOWN_BRANDS = [
    "nike", "adidas", "brooks", "asics", "new balance", "hoka",
    "saucony", "reebok", "puma", "vans", "converse", "salomon",
    "merrell", "lululemon", "gymshark", "under armour", "mizuno",
    "birkenstock", "teva", "crocs", "havaianas", "garmin", "balega",
]


# ═════════════════════════════════════════════════════════════════════════════
# Step 1a: Cloud Embedder (HF SentenceTransformers + TF-IDF fallback)
# ═════════════════════════════════════════════════════════════════════════════

class CloudEmbedder:
    """
    Generates dense embeddings via HuggingFace SentenceTransformers API.
    Falls back to TF-IDF when API is unavailable or keys are missing.
    Uses a simple in-memory dict cache keyed by MD5 of text.
    """

    def __init__(self, api_key: str = "", offline: bool = False):
        self.api_key = api_key
        self.offline = offline or not api_key
        self._cache: Dict[str, List[float]] = {}
        self._tfidf = None

        if self.offline:
            logger.warning("CloudEmbedder: no HF key — TF-IDF fallback active")

    def fit_fallback(self, texts: List[str]) -> None:
        """Fit TF-IDF fallback on corpus. Must be called before embed() in offline mode."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(max_features=EMBEDDING_DIM, sublinear_tf=True)
        self._tfidf.fit(texts)
        logger.info("TF-IDF fallback fitted on %d texts", len(texts))

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a dense embedding per text, using cache when possible."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_idx, miss_texts = [], []

        for i, t in enumerate(texts):
            key = hashlib.md5(t.encode()).hexdigest()
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                miss_idx.append(i)
                miss_texts.append(t)

        if miss_texts:
            embeds = (
                self._tfidf_embed(miss_texts)
                if self.offline
                else self._hf_embed(miss_texts)
            )
            for i, (orig_idx, text) in enumerate(zip(miss_idx, miss_texts)):
                vec = embeds[i]
                results[orig_idx] = vec
                self._cache[hashlib.md5(text.encode()).hexdigest()] = vec

        return results  # type: ignore[return-value]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _hf_embed(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        for attempt in range(retries):
            try:
                resp = requests.post(HF_EMBED_URL, headers=headers, json=payload, timeout=45)
                if resp.status_code == 200:
                    data = resp.json()
                    return self._parse_hf_response(data, len(texts))
                if resp.status_code in (503, 429):
                    wait = 2 ** attempt
                    logger.warning("HF API %s — retrying in %ss", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                logger.error("HF API error %s: %s", resp.status_code, resp.text[:200])
            except Exception as exc:
                logger.error("HF API exception: %s", exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        logger.warning("HF API failed after %d retries - falling back to TF-IDF", retries)
        return self._tfidf_embed(texts)

    @staticmethod
    def _parse_hf_response(data: list, expected: int) -> List[List[float]]:
        """
        HF feature-extraction returns either:
          - list[list[float]]      — sentence embeddings (sentence-transformers)
          - list[list[list[float]]]— per-token embeddings; mean-pool needed
        """
        if not data:
            return [[0.0] * EMBEDDING_DIM] * expected
        sample = data[0]
        if isinstance(sample[0], list):
            # Per-token: mean-pool over token dim
            return [
                np.mean(token_embeds, axis=0).tolist()
                for token_embeds in data
            ]
        # Already sentence embeddings
        return data

    def _tfidf_embed(self, texts: List[str]) -> List[List[float]]:
        if self._tfidf is None or not hasattr(self._tfidf, "vocabulary_"):
            return [[0.0] * EMBEDDING_DIM] * len(texts)
        mat = self._tfidf.transform(texts).toarray()
        result = []
        for row in mat:
            if len(row) >= EMBEDDING_DIM:
                result.append(row[:EMBEDDING_DIM].tolist())
            else:
                padded = np.zeros(EMBEDDING_DIM)
                padded[: len(row)] = row
                result.append(padded.tolist())
        return result


# ═════════════════════════════════════════════════════════════════════════════
# Step 1b: Sparse Encoder (BM25-style TF-IDF for Pinecone sparse_values)
# ═════════════════════════════════════════════════════════════════════════════

class SparseEncoder:
    """
    Produces Pinecone-compatible sparse_values dicts for keyword matching.
    Uses sklearn TfidfVectorizer with sublinear TF as the underlying scorer.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(max_features=8000, sublinear_tf=True, ngram_range=(1, 2))

    def fit(self, texts: List[str]) -> None:
        self._vec.fit(texts)
        logger.info("SparseEncoder fitted | vocab=%d", len(self._vec.vocabulary_))

    def encode(self, text: str) -> Dict:
        """Return {"indices": [...], "values": [...]} for Pinecone."""
        mat = self._vec.transform([text]).tocoo()
        if mat.nnz == 0:
            return {"indices": [], "values": []}
        return {
            "indices": mat.col.tolist(),
            "values": mat.data.tolist(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Step 1c: Pinecone Manager (hybrid index: dense + sparse + metadata)
# ═════════════════════════════════════════════════════════════════════════════

class PineconeManager:
    """
    Manages a Pinecone hybrid index (metric=dotproduct).
    Falls back to an in-memory store when Pinecone is unavailable.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dim: int,
        offline: bool = False,
    ):
        self.index_name = index_name
        self.dim = dim
        self.offline = offline
        self._index = None
        # In-memory store used by offline mode AND as a local cache for metadata
        self._store: Dict[str, Dict] = {}

        if not offline and api_key:
            self._connect(api_key)

    def _connect(self, api_key: str) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=api_key)
            existing = [idx.name for idx in pc.list_indexes()]

            if self.index_name not in existing:
                logger.info("Creating Pinecone index '%s' (dim=%d, metric=dotproduct)…", self.index_name, self.dim)
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dim,
                    metric="dotproduct",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                # Poll until ready
                for _ in range(30):
                    if pc.describe_index(self.index_name).status.get("ready"):
                        break
                    time.sleep(2)
                logger.info("Pinecone index ready.")

            self._index = pc.Index(self.index_name)
            logger.info("Connected to Pinecone index '%s'", self.index_name)
        except Exception as exc:
            logger.error("Pinecone connection failed: %s — using offline fallback", exc)
            self.offline = True

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_products(
        self,
        products: List[Dict],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict],
        batch_size: int = 100,
    ) -> None:
        """Upsert products with their hybrid vectors and metadata."""
        pinecone_vectors = []

        for product, dense, sparse in zip(products, dense_vectors, sparse_vectors):
            meta = {
                "name":       product.get("name", ""),
                "desc":       product.get("desc", ""),
                "category":   product.get("category", ""),
                "brand":      product.get("brand", ""),
                "price":      float(product.get("price", 0)),
                "rating":     float(product.get("rating", 3.5)),
                "popularity": float(product.get("popularity", 0.5)),
                "stock":      bool(product.get("stock", True)),
                "discount":   float(product.get("discount", 0.0)),
            }
            # Always keep in local store for offline fallback & metadata lookup
            self._store[product["id"]] = {"dense": dense, "sparse": sparse, "metadata": meta}

            if not self.offline:
                pinecone_vectors.append({
                    "id":            product["id"],
                    "values":        dense,
                    "sparse_values": sparse,
                    "metadata":      meta,
                })

        if pinecone_vectors and self._index:
            for i in range(0, len(pinecone_vectors), batch_size):
                self._index.upsert(vectors=pinecone_vectors[i: i + batch_size])
            logger.info("Upserted %d vectors to Pinecone index '%s'", len(pinecone_vectors), self.index_name)
        else:
            logger.info("Offline: %d products stored in memory", len(self._store))

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        dense_vector: List[float],
        sparse_vector: Dict,
        top_k: int = TOP_CANDIDATES,
        stock_only: bool = True,
    ) -> List[Dict]:
        """Return top_k candidates with dense_sim, sparse_score, and metadata."""
        if not self.offline and self._index:
            return self._pinecone_query(dense_vector, sparse_vector, top_k, stock_only)
        return self._offline_query(dense_vector, sparse_vector, top_k, stock_only)

    def _pinecone_query(
        self, dense: List[float], sparse: Dict, top_k: int, stock_only: bool
    ) -> List[Dict]:
        try:
            kwargs: Dict = {
                "vector":       dense,
                "sparse_vector": sparse,
                "top_k":        top_k,
                "include_metadata": True,
            }
            if stock_only:
                kwargs["filter"] = {"stock": {"$eq": True}}

            resp = self._index.query(**kwargs)
            candidates = []
            for m in resp.get("matches", []):
                meta = m.get("metadata", {})
                # Pinecone returns a combined score for hybrid queries
                combined = float(m.get("score", 0.0))
                candidates.append({
                    "product_id":   m["id"],
                    "dense_sim":    combined,   # normalized later
                    "sparse_score": 0.0,        # Pinecone blends internally
                    "metadata":     meta,
                })
            return candidates
        except Exception as exc:
            logger.error("Pinecone query failed: %s — falling back to offline", exc)
            return self._offline_query(dense, sparse, top_k, stock_only)

    def _offline_query(
        self, dense: List[float], sparse: Dict, top_k: int, stock_only: bool
    ) -> List[Dict]:
        q_dense = np.array(dense, dtype=np.float32)
        q_norm  = float(np.linalg.norm(q_dense)) or 1.0
        q_sparse_map = dict(zip(sparse.get("indices", []), sparse.get("values", [])))

        candidates = []
        for pid, data in self._store.items():
            meta = data["metadata"]
            if stock_only and not meta.get("stock", True):
                continue

            # Dense cosine similarity
            p_dense = np.array(data["dense"], dtype=np.float32)
            p_norm  = float(np.linalg.norm(p_dense)) or 1.0
            cos_sim = float(np.dot(q_dense, p_dense) / (q_norm * p_norm))

            # Sparse dot product (keyword match)
            p_sparse_map = dict(
                zip(data["sparse"].get("indices", []), data["sparse"].get("values", []))
            )
            sparse_score = sum(
                q_sparse_map[i] * v for i, v in p_sparse_map.items() if i in q_sparse_map
            )

            candidates.append({
                "product_id":   pid,
                "dense_sim":    max(0.0, cos_sim),
                "sparse_score": max(0.0, sparse_score),
                "metadata":     meta,
            })

        # Sort by hybrid score and truncate
        candidates.sort(
            key=lambda c: W_DENSE * c["dense_sim"] + W_SPARSE * c["sparse_score"],
            reverse=True,
        )
        return candidates[:top_k]


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Search Intent Analyzer
# ═════════════════════════════════════════════════════════════════════════════

class IntentAnalyzer:
    """
    Rule-based intent detector with confidence scoring.
    Returns: {"label": str, "confidence": float, "slots": Dict}

    Intents: price_sensitive | premium | comparison | gift | urgent_buy | exploratory
    """

    def analyze(self, query: str) -> Dict:
        q = query.lower().strip()
        scores: Dict[str, float] = {}

        for intent, keywords in _INTENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in q)
            if hits:
                # Confidence: 0.5 base + 0.25 per extra keyword match, capped at 0.95
                scores[intent] = min(0.95, 0.50 + 0.25 * (hits - 1))

        slots = self._extract_slots(q)

        if not scores:
            return {"label": "exploratory", "confidence": 0.30, "slots": slots}

        best = max(scores, key=lambda k: scores[k])
        confidence = scores[best]

        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            return {"label": "exploratory", "confidence": confidence, "slots": slots}

        return {"label": best, "confidence": confidence, "slots": slots}

    @staticmethod
    def _extract_slots(q: str) -> Dict:
        slots: Dict[str, str] = {}

        # Price ceiling hint: "$50", "under 100", "less than 80", "max 60"
        price_pat = re.search(
            r"(?:under|below|less than|max|maximum|\$)\s*\$?(\d+(?:\.\d{1,2})?)", q
        )
        if price_pat:
            slots["max_price"] = price_pat.group(1)

        # Brand hint (longest match wins)
        for brand in sorted(_KNOWN_BRANDS, key=len, reverse=True):
            if brand in q:
                slots["brand"] = brand
                break

        # Category hint (longest phrase match wins)
        for phrase, cat in sorted(_CATEGORY_HINTS.items(), key=lambda x: len(x[0]), reverse=True):
            if phrase in q:
                slots["category"] = cat
                break

        return slots


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: BehaviorIQ Reranker
# ═════════════════════════════════════════════════════════════════════════════

class BehaviorIQReranker:
    """
    Produces the final ordered list by combining three signals:
      - vector_score  (dense cosine + sparse keyword, normalized)
      - intent_score  (query intent × product fit × confidence backoff)
      - pricing_score (business-aware price fitness per intent, churn-adjusted)

    Formula:  final = w_v * vector + w_i * intent + w_p * pricing
    Weights shift per intent via _INTENT_WEIGHTS.
    """

    def rerank(
        self,
        candidates: List[Dict],
        intent: Dict,
        churn_score: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        label      = intent.get("label", "exploratory")
        confidence = float(intent.get("confidence", 0.30))
        w          = weights or _INTENT_WEIGHTS.get(label, BASE_WEIGHTS)

        # Normalize dense + sparse scores across the candidate set
        dense_max  = max((c["dense_sim"]    for c in candidates), default=1.0) or 1.0
        sparse_max = max((c["sparse_score"] for c in candidates), default=1.0) or 1.0

        # Price stats for in-stock candidates
        prices = [
            c["metadata"]["price"]
            for c in candidates
            if c["metadata"].get("stock", True) and c["metadata"].get("price") is not None
        ]
        min_p      = min(prices) if prices else 0.0
        max_p      = max(prices) if prices else 1.0
        price_span = (max_p - min_p) or 1.0

        results = []
        for c in candidates:
            meta = c["metadata"]
            if not meta.get("stock", True):
                continue  # guardrail: never rank out-of-stock items

            price      = float(meta.get("price", 0))
            discount   = float(meta.get("discount", 0))
            rating     = float(meta.get("rating", 3.5))
            popularity = float(meta.get("popularity", 0.5))

            # ── vector_score ──────────────────────────────────────────────
            norm_dense  = c["dense_sim"]    / dense_max
            norm_sparse = c["sparse_score"] / sparse_max
            vector_score = W_DENSE * norm_dense + W_SPARSE * norm_sparse

            # Relevance floor: suppress semantically irrelevant candidates
            relevance_ok = vector_score >= 0.05

            # ── pricing_score ─────────────────────────────────────────────
            norm_price    = (price - min_p) / price_span  # 0 = cheapest, 1 = priciest
            discount_lift = discount * 0.30               # up to +0.30 for 100% discount

            if label == "price_sensitive":
                pricing_score = (1.0 - norm_price) + discount_lift
                # High-churn users get a stronger price boost (retention lever)
                pricing_score *= 1.0 + 0.40 * churn_score
            elif label == "premium":
                pricing_score = 0.50 * norm_price + 0.50 * (rating / 5.0)
            elif label == "urgent_buy":
                pricing_score = 0.60 * popularity + 0.40 * (1.0 - norm_price * 0.40)
            elif label == "gift":
                pricing_score = (
                    0.40 * (rating / 5.0)
                    + 0.30 * (1.0 - abs(norm_price - 0.50))
                    + 0.30 * popularity
                )
            else:
                # Exploratory / comparison: gentle popularity tilt, no price bias
                pricing_score = 0.50 + 0.20 * (popularity - 0.50)

            pricing_score = float(np.clip(pricing_score, 0.0, 1.0))

            # ── intent_score  ─────────────────────────────────────────────
            if label == "price_sensitive":
                raw_intent = (1.0 - norm_price) + discount_lift
            elif label == "premium":
                raw_intent = 0.60 * (rating / 5.0) + 0.40 * norm_price
            elif label == "comparison":
                raw_intent = vector_score
            elif label == "gift":
                raw_intent = 0.50 * (rating / 5.0) + 0.50 * popularity
            elif label == "urgent_buy":
                raw_intent = popularity
            else:
                raw_intent = vector_score

            intent_score = float(np.clip(raw_intent, 0.0, 1.0)) * confidence

            # Penalise low-relevance results to prevent cheap-but-irrelevant items rising
            if not relevance_ok:
                intent_score  *= 0.30
                pricing_score *= 0.30

            # ── final score ───────────────────────────────────────────────
            final_score = (
                w.get("vector", 0.45) * vector_score
                + w.get("intent",  0.30) * intent_score
                + w.get("pricing", 0.25) * pricing_score
            )

            results.append({
                "product_id":        c["product_id"],
                "name":              meta.get("name", ""),
                "category":          meta.get("category", ""),
                "brand":             meta.get("brand", ""),
                "price":             price,
                "discount":          discount,
                "rating":            rating,
                "final_score":       round(float(final_score), 4),
                "vector_score":      round(float(vector_score), 4),
                "dense_sim":         round(float(norm_dense), 4),
                "sparse_score":      round(float(norm_sparse), 4),
                "intent_score":      round(float(intent_score), 4),
                "pricing_score":     round(float(pricing_score), 4),
                "intent_label":      label,
                "intent_confidence": round(confidence, 4),
                "rank_explanation":  self._explain(label, price, discount, rating, final_score),
            })

        results.sort(key=lambda r: r["final_score"], reverse=True)
        return results

    @staticmethod
    def _explain(label: str, price: float, discount: float, rating: float, score: float) -> str:
        parts: List[str] = []
        if label == "price_sensitive":
            parts.append(f"budget pick at ${price:.2f}")
            if discount > 0:
                parts.append(f"{int(discount * 100)}% off")
        elif label == "premium":
            parts.append(f"premium quality — rated {rating:.1f}/5")
        elif label == "gift":
            parts.append(f"highly-rated gift option ({rating:.1f}/5)")
        elif label == "urgent_buy":
            parts.append("popular & available now")
        elif label == "comparison":
            parts.append("strong semantic match for comparison")
        else:
            parts.append("relevant match")
        parts.append(f"score {score:.3f}")
        return ", ".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI schemas
# ═════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str = Field(..., description="Raw natural language query, e.g. 'cheap running shoes'")
    churn_score: float = Field(0.5, ge=0.0, le=1.0, description="User churn probability [0,1] from churn model")
    recent_product_ids: Optional[List[str]] = Field(None, description="Recently viewed product IDs for user context")
    top_k: int = Field(20, ge=1, le=100, description="How many results to return")
    weights: Optional[Dict[str, float]] = Field(None, description="Override scoring weights {vector, intent, pricing}")


class ProductResult(BaseModel):
    rank: int
    product_id: str
    name: str
    category: str
    brand: str
    price: float
    discount: float
    rating: float
    final_score: float
    vector_score: float
    intent_score: float
    pricing_score: float
    intent_label: str
    intent_confidence: float
    rank_explanation: str


class IntentInfo(BaseModel):
    label: str
    confidence: float
    slots: Dict


class SearchResponse(BaseModel):
    query: str
    intent: IntentInfo
    results: List[ProductResult]
    search_backend: str
    total_candidates: int
    pipeline_ms: float


# ═════════════════════════════════════════════════════════════════════════════
# Application
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="BehaviorIQ Reranker",
    version="2.0.0",
    description="Full pipeline: semantic+keyword search → intent analysis → BehaviorIQ reranker",
)

# Singletons initialised at startup
_embedder:      Optional[CloudEmbedder]   = None
_sparse_enc:    Optional[SparseEncoder]   = None
_pinecone_mgr:  Optional[PineconeManager] = None
_intent_analyzer = IntentAnalyzer()
_reranker        = BehaviorIQReranker()


def _load_products() -> List[Dict]:
    """Load products from products_v2.json, then products.json, then SEED_PRODUCTS fallback."""
    for fname in ("data/products_v2.json", "data/products.json"):
        # data/ lives in ml-service/, one level above models/
        path = Path(__file__).parent.parent / fname
        if path.exists():
            products = json.loads(path.read_text())
            # Ensure required fields exist
            for p in products:
                p.setdefault("brand",      "Generic")
                p.setdefault("rating",     3.5)
                p.setdefault("popularity", 0.5)
                p.setdefault("stock",      True)
                p.setdefault("discount",   0.0)
            logger.info("Loaded %d products from %s", len(products), fname)
            return products

    logger.warning("No products file found — using embedded seed catalog")
    return _SEED_PRODUCTS


@app.on_event("startup")
def startup_event() -> None:
    global _embedder, _sparse_enc, _pinecone_mgr

    logger.info("=== BehaviorIQ Reranker v2 starting (offline=%s) ===", OFFLINE_MODE)

    products = _load_products()
    corpus   = [
        f"{p['name']} {p.get('desc', '')} {p.get('category', '')} {p.get('brand', '')}"
        for p in products
    ]

    # Sparse encoder (always local)
    _sparse_enc = SparseEncoder()
    _sparse_enc.fit(corpus)

    # Cloud embedder — always fit TF-IDF fallback so it's ready if HF API fails
    _embedder = CloudEmbedder(api_key=HF_API_KEY, offline=OFFLINE_MODE)
    _embedder.fit_fallback(corpus)

    # Pinecone (or offline in-memory)
    _pinecone_mgr = PineconeManager(
        api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        dim=EMBEDDING_DIM,
        offline=OFFLINE_MODE,
    )

    # Embed all products and upsert
    logger.info("Embedding %d products…", len(products))
    dense_vecs  = _embedder.embed(corpus)
    sparse_vecs = [_sparse_enc.encode(t) for t in corpus]
    _pinecone_mgr.upsert_products(products, dense_vecs, sparse_vecs)

    backend = "pinecone_hybrid" if not _pinecone_mgr.offline else "offline_tfidf"
    logger.info(
        "BehaviorIQ Reranker ready | products=%d | backend=%s",
        len(products), backend,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    Full BehaviorIQ pipeline:
      1. Embed query (dense + sparse)
      2. Hybrid retrieve candidates from Pinecone
      3. Detect search intent with confidence
      4. Rerank with vector + intent + pricing + churn signals
    """
    if _embedder is None or _sparse_enc is None or _pinecone_mgr is None:
        raise HTTPException(503, "Service not initialised yet — retry in a moment")

    t0 = time.perf_counter()

    # ── Step 1: Embed query ───────────────────────────────────────────────────
    query_dense  = _embedder.embed_one(req.query)
    query_sparse = _sparse_enc.encode(req.query)

    # ── Step 2: Intent analysis ───────────────────────────────────────────────
    intent = _intent_analyzer.analyze(req.query)

    # ── Step 3: Retrieve candidates ───────────────────────────────────────────
    candidates = _pinecone_mgr.query(query_dense, query_sparse, top_k=TOP_CANDIDATES)

    # Optional: narrow by category slot from intent
    cat_slot = intent.get("slots", {}).get("category")
    if cat_slot:
        filtered = [c for c in candidates if c["metadata"].get("category") == cat_slot]
        candidates = filtered if filtered else candidates  # don't empty the set

    # Optional: narrow by max_price slot
    max_price_slot = intent.get("slots", {}).get("max_price")
    if max_price_slot:
        try:
            ceiling = float(max_price_slot)
            filtered = [c for c in candidates if c["metadata"].get("price", 9999) <= ceiling]
            candidates = filtered if filtered else candidates
        except ValueError:
            pass

    # ── Step 4: BehaviorIQ rerank ─────────────────────────────────────────────
    ranked = _reranker.rerank(
        candidates,
        intent,
        churn_score=req.churn_score,
        weights=req.weights,
    )
    ranked = ranked[: req.top_k]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    backend    = "pinecone_hybrid" if not _pinecone_mgr.offline else "offline_tfidf_fallback"

    return SearchResponse(
        query=req.query,
        intent=IntentInfo(**intent),
        results=[ProductResult(rank=i + 1, **r) for i, r in enumerate(ranked)],
        search_backend=backend,
        total_candidates=len(candidates),
        pipeline_ms=elapsed_ms,
    )


@app.get("/health")
def health() -> Dict:
    indexed = len(_pinecone_mgr._store) if _pinecone_mgr else 0
    backend = "pinecone_hybrid" if (_pinecone_mgr and not _pinecone_mgr.offline) else "offline_tfidf_fallback"
    return {
        "status":          "ok",
        "service":         "BehaviorIQ Reranker v2",
        "products_indexed": indexed,
        "search_backend":  backend,
        "offline_mode":    OFFLINE_MODE,
    }


@app.get("/index-status")
def index_status() -> Dict:
    if _pinecone_mgr and not _pinecone_mgr.offline and _pinecone_mgr._index:
        try:
            stats = _pinecone_mgr._index.describe_index_stats()
            return {"status": "connected", "index": INDEX_NAME, "stats": stats}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
    return {
        "status":           "offline",
        "index":            INDEX_NAME,
        "indexed_products": len(_pinecone_mgr._store) if _pinecone_mgr else 0,
    }


# ── Minimal embedded seed catalog (used only if no products file found) ───────
_SEED_PRODUCTS: List[Dict] = [
    {"id": "rs-001", "name": "Skechers Go Run Pulse", "desc": "Budget running shoe memory foam", "category": "running_shoes", "brand": "Skechers", "price": 49.99, "rating": 4.1, "popularity": 0.72, "stock": True, "discount": 0.0},
    {"id": "rs-002", "name": "New Balance Fresh Foam 520", "desc": "Affordable everyday running shoe plush cushioning", "category": "running_shoes", "brand": "New Balance", "price": 59.99, "rating": 4.3, "popularity": 0.78, "stock": True, "discount": 0.10},
    {"id": "rs-003", "name": "Nike Air Zoom Pegasus 41", "desc": "Versatile running shoe Zoom Air cushioning all distances", "category": "running_shoes", "brand": "Nike", "price": 130.00, "rating": 4.5, "popularity": 0.91, "stock": True, "discount": 0.0},
    {"id": "rs-004", "name": "Nike Vaporfly 3", "desc": "Elite marathon carbon plate racing shoe", "category": "running_shoes", "brand": "Nike", "price": 249.99, "rating": 4.9, "popularity": 0.82, "stock": True, "discount": 0.0},
    {"id": "rs-005", "name": "Adidas Adizero Adios Pro 3", "desc": "Professional marathon racing shoe carbon energy rods", "category": "running_shoes", "brand": "Adidas", "price": 229.99, "rating": 4.8, "popularity": 0.78, "stock": True, "discount": 0.0},
    {"id": "cs-001", "name": "Nike Air Force 1 Low", "desc": "Classic casual sneaker versatile everyday street wear", "category": "casual_shoes", "brand": "Nike", "price": 109.99, "rating": 4.6, "popularity": 0.96, "stock": True, "discount": 0.0},
    {"id": "cs-002", "name": "Converse Chuck Taylor All Star", "desc": "Iconic canvas high-top casual sneaker timeless style", "category": "casual_shoes", "brand": "Converse", "price": 59.99, "rating": 4.4, "popularity": 0.89, "stock": True, "discount": 0.0},
    {"id": "ap-001", "name": "Nike Dri-FIT Running T-Shirt", "desc": "Moisture-wicking running shirt lightweight breathable", "category": "sports_shirt", "brand": "Nike", "price": 29.99, "rating": 4.4, "popularity": 0.82, "stock": True, "discount": 0.0},
    {"id": "ap-002", "name": "Lululemon Surge Jogger", "desc": "Premium athletic jogger pants for running casual wear", "category": "sports_pants", "brand": "Lululemon", "price": 118.00, "rating": 4.8, "popularity": 0.68, "stock": True, "discount": 0.0},
    {"id": "ac-001", "name": "Garmin Forerunner 965", "desc": "Premium flagship GPS running watch advanced metrics AMOLED", "category": "running_accessories", "brand": "Garmin", "price": 599.99, "rating": 4.8, "popularity": 0.65, "stock": True, "discount": 0.0},
]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("models.reranker_updated:app", host="0.0.0.0", port=8002, reload=True)
