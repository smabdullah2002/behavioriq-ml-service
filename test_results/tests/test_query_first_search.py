"""Tests for query-first product matching in the BIQ search pipeline."""

from __future__ import annotations

import unittest

from models.reranker_updated import (
    IntentAnalyzer,
    apply_query_first_filter,
    compute_query_match_score,
)


class QueryFirstSearchTests(unittest.TestCase):
    def test_extract_product_terms_strips_intent_modifiers(self):
        self.assertEqual(IntentAnalyzer.extract_product_terms("premium shoes"), ["shoes"])
        self.assertEqual(
            IntentAnalyzer.extract_product_terms("cheap nike running shoes"),
            ["nike", "running", "shoes"],
        )
        self.assertEqual(IntentAnalyzer.extract_product_terms("gift boxers for him"), ["boxers"])

    def test_shoe_query_matches_footwear_not_shirts(self):
        shoe_meta = {
            "name": "Nike Air Zoom Pegasus Running Shoes",
            "desc": "Versatile daily trainer",
            "category": "road running",
            "brand": "Nike",
        }
        shirt_meta = {
            "name": "Premium Cotton Polo Shirt",
            "desc": "Soft breathable polo",
            "category": "polos",
            "brand": "Polo",
        }

        terms = ["shoes"]
        self.assertGreater(compute_query_match_score(shoe_meta, terms), 0.5)
        self.assertEqual(compute_query_match_score(shirt_meta, terms), 0.0)

    def test_query_first_filter_removes_irrelevant_candidates(self):
        candidates = [
            {
                "product_id": "ec-001",
                "dense_sim": 0.9,
                "sparse_score": 0.2,
                "metadata": {
                    "name": "Premium Polo Shirt",
                    "desc": "Soft cotton polo",
                    "category": "polos",
                    "brand": "Polo",
                    "price": 39.99,
                    "stock": True,
                },
            },
            {
                "product_id": "ec-500",
                "dense_sim": 0.7,
                "sparse_score": 0.8,
                "metadata": {
                    "name": "Adidas Ultraboost Running Shoes",
                    "desc": "Responsive cushioned running shoe",
                    "category": "road running",
                    "brand": "Adidas",
                    "price": 129.99,
                    "stock": True,
                },
            },
        ]

        filtered = apply_query_first_filter(candidates, ["shoes"])
        product_ids = {item["product_id"] for item in filtered}

        self.assertIn("ec-500", product_ids)
        self.assertNotIn("ec-001", product_ids)

    def test_premium_shoes_intent_label_does_not_block_category_slot(self):
        intent = IntentAnalyzer().analyze("premium shoes")
        self.assertEqual(intent["label"], "premium")
        self.assertEqual(intent["slots"].get("category"), "shoes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
