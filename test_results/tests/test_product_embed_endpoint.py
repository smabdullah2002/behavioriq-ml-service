"""Smoke tests for POST /ml/product-embed."""

import requests

BASE = "http://127.0.0.1:8001"


def test_product_embed_returns_vector():
    payload = {
        "product_id": "test-embed-001",
        "name": "Trail Runner Pro",
        "description": "Lightweight trail shoe for long runs",
        "category": "trail_shoes",
        "brand": "Stride",
    }
    r = requests.post(f"{BASE}/ml/product-embed", json=payload, timeout=15)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["product_id"] == payload["product_id"]
    assert len(data["product_vector"]) == data["vector_dim"]
    assert data["vector_dim"] > 0


def test_product_embed_user_vector_lookup():
    """After embed, user-vector should find the new product id."""
    pid = "test-embed-002"
    requests.post(
        f"{BASE}/ml/product-embed",
        json={
            "product_id": pid,
            "name": "City Sprint",
            "description": "Urban daily trainer",
            "category": "running_shoes",
            "brand": "Pace",
        },
        timeout=15,
    )
    r = requests.post(
        f"{BASE}/ml/user-vector",
        json={"recent_product_ids": [pid]},
        timeout=15,
    )
    assert r.status_code == 200
    assert len(r.json()["user_vector"]) > 0
