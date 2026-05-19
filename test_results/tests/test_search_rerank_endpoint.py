"""Deterministic tests for the /ml/search-rerank endpoint."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class _FakeEmbedder:
    def embed_one(self, _query: str):
        return [0.4, 0.6]


class _FakeSparseEncoder:
    def encode(self, _query: str):
        return {"indices": [1], "values": [0.9]}


class _FakeIntentAnalyzer:
    def analyze(self, query: str):
        q = query.lower()
        if "under $" in q or "cheap" in q or "budget" in q:
            return {"label": "price_sensitive", "confidence": 0.9, "slots": {}}
        if "premium" in q or "luxury" in q:
            return {"label": "premium", "confidence": 0.9, "slots": {}}
        if "gift" in q or "birthday" in q:
            return {"label": "gift", "confidence": 0.85, "slots": {}}
        if "vs" in q or "compare" in q:
            return {"label": "comparison", "confidence": 0.8, "slots": {}}
        return {"label": "exploratory", "confidence": 0.6, "slots": {}}


class _FakePineconeManager:
    offline = True

    def query(self, _dense, _sparse, top_k=100):
        candidates = []
        for i in range(top_k):
            idx = i + 1
            candidates.append(
                {
                    "product_id": f"p{idx}",
                    "dense_sim": 1.0 - (i * 0.01),
                    "sparse_score": 0.5,
                    "metadata": {
                        "name": f"Product {idx}",
                        "category": "running_shoes",
                        "brand": "Generic",
                        "price": float(20 + i),
                        "discount": 0.0,
                        "rating": 4.2,
                        "stock": True,
                    },
                }
            )
        return candidates


class _FakeReranker:
    def rerank(self, candidates, intent, churn_score=0.5, weights=None):
        w = weights or {"vector": 0.45, "intent": 0.30, "pricing": 0.25}
        ranked = []
        for i, c in enumerate(candidates):
            meta = c["metadata"]
            vector_score = max(0.0, c["dense_sim"])
            intent_score = min(1.0, intent["confidence"])
            pricing_score = max(0.0, 1.0 - (meta["price"] / 300.0))
            final_score = (
                w["vector"] * vector_score
                + w["intent"] * intent_score
                + w["pricing"] * pricing_score
            )
            ranked.append(
                {
                    "product_id": c["product_id"],
                    "name": meta["name"],
                    "category": meta["category"],
                    "brand": meta["brand"],
                    "price": meta["price"],
                    "discount": meta["discount"],
                    "rating": meta["rating"],
                    "final_score": round(final_score - i * 0.001, 4),
                    "vector_score": round(vector_score, 4),
                    "intent_score": round(intent_score, 4),
                    "pricing_score": round(pricing_score, 4),
                    "intent_label": intent["label"],
                    "intent_confidence": intent["confidence"],
                    "rank_explanation": "mocked ranking",
                }
            )
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked


class SearchEndpointPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app)

    def _post(self, payload: dict):
        return self.client.post("/ml/search", json=payload)

    def test_unique_search_payloads_return_valid_structure(self):
        payloads = [
            {
                "query": "cheap running shoes with fast shipping",
                "churn_score": 0.72,
                "recent_product_ids": ["p2", "p6", "p9"],
                "top_k": 15,
                "weights": {"vector": 0.35, "intent": 0.40, "pricing": 0.25},
            },
            {
                "query": "premium nike running shoes best quality",
                "churn_score": 0.10,
                "recent_product_ids": ["p3", "p11", "p15"],
                "top_k": 10,
                "weights": {"vector": 0.50, "intent": 0.35, "pricing": 0.15},
            },
            {
                "query": "running shoes under $60",
                "churn_score": 0.33,
                "recent_product_ids": ["p1", "p4", "p8"],
                "top_k": 20,
                "weights": {"vector": 0.40, "intent": 0.25, "pricing": 0.35},
            },
            {
                "query": "brooks vs asics daily trainer",
                "churn_score": 0.21,
                "recent_product_ids": ["p5", "p7", "p14"],
                "top_k": 12,
                "weights": {"vector": 0.55, "intent": 0.30, "pricing": 0.15},
            },
            {
                "query": "gift running shoes for him birthday",
                "churn_score": 0.47,
                "recent_product_ids": ["p10", "p12", "p16"],
                "top_k": 8,
                "weights": {"vector": 0.45, "intent": 0.40, "pricing": 0.15},
            },
            {
                "query": "good shoes for daily jogging",
                "churn_score": 0.28,
                "recent_product_ids": ["p1", "p13", "p17"],
                "top_k": 25,
                "weights": {"vector": 0.60, "intent": 0.20, "pricing": 0.20},
            },
            {
                "query": "budget sneakers on sale",
                "churn_score": 0.95,
                "recent_product_ids": ["p2", "p2", "p3"],
                "top_k": 20,
                "weights": {"vector": 0.30, "intent": 0.25, "pricing": 0.45},
            },
            {
                "query": "trail running shoes",
                "churn_score": 0.14,
                "recent_product_ids": [],
                "top_k": 10,
                "weights": {"vector": 0.50, "intent": 0.30, "pricing": 0.20},
            },
        ]

        with patch.object(main, "_biq_embedder", _FakeEmbedder()), patch.object(
            main, "_biq_sparse_enc", _FakeSparseEncoder()
        ), patch.object(main, "_biq_intent", _FakeIntentAnalyzer()), patch.object(
            main, "_biq_pinecone_mgr", _FakePineconeManager()
        ), patch.object(main, "_biq_reranker", _FakeReranker()):
            for payload in payloads:
                with self.subTest(query=payload["query"]):
                    response = self._post(payload)
                    self.assertEqual(response.status_code, 200)

                    body = response.json()
                    self.assertEqual(body["query"], payload["query"])
                    self.assertEqual(body["search_backend"], "offline_tfidf_fallback")
                    self.assertLessEqual(len(body["results"]), payload["top_k"])
                    self.assertGreater(len(body["results"]), 0)
                    self.assertIn("label", body["intent"])
                    self.assertIn("confidence", body["intent"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
