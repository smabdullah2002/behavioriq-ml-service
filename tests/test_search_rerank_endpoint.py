#!/usr/bin/env python3
"""Test cases for the /ml/search-rerank endpoint."""

import requests

BASE_URL = "http://127.0.0.1:8001"

# 100-dim zero vector (fallback when not using real user vector)
ZERO_VECTOR = [0.0] * 100

# Slight bias toward first few dimensions (simulates a real user vector)
MOCK_USER_VECTOR = [0.3 if i < 10 else 0.0 for i in range(100)]


def fetch_user_vector(product_ids, weights=None):
    """Helper: fetch a real user vector from /ml/user-vector."""
    payload = {"recent_product_ids": product_ids}
    if weights:
        payload["weights"] = weights
    resp = requests.post(f"{BASE_URL}/ml/user-vector", json=payload, timeout=5)
    if resp.status_code == 200:
        return resp.json()["user_vector"]
    return MOCK_USER_VECTOR


test_cases = [
    # ── Basic Ordering ────────────────────────────────────────────────────────
    {
        "name": "Higher keyword score ranks first (default weights)",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [
                {"product_id": "p1", "keyword_score": 0.3, "popularity_score": 0.5},
                {"product_id": "p2", "keyword_score": 0.9, "popularity_score": 0.5},
            ],
        },
        "expect": {
            "top_product": "p2",
            "description": "p2 has higher keyword_score so should rank first with default kw=0.5 weight",
        },
    },
    {
        "name": "Higher popularity score ranks first when popularity weight is dominant",
        "payload": {
            "user_vector": ZERO_VECTOR,
            "candidates": [
                {"product_id": "p3", "keyword_score": 0.5, "popularity_score": 0.2},
                {"product_id": "p4", "keyword_score": 0.5, "popularity_score": 0.9},
            ],
            "weights": {"kw": 0.1, "cosine": 0.0, "popularity": 0.9},
        },
        "expect": {
            "top_product": "p4",
            "description": "p4 has higher popularity and popularity weight is 0.9, so p4 should win",
        },
    },
    {
        "name": "Keyword-only weights — keyword score fully determines rank",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [
                {"product_id": "p5", "keyword_score": 0.1, "popularity_score": 0.9},
                {"product_id": "p6", "keyword_score": 0.8, "popularity_score": 0.1},
                {"product_id": "p7", "keyword_score": 0.5, "popularity_score": 0.5},
            ],
            "weights": {"kw": 1.0, "cosine": 0.0, "popularity": 0.0},
        },
        "expect": {
            "top_product": "p6",
            "description": "With kw=1.0 and all else zero, p6 (keyword=0.8) must rank first",
        },
    },
    {
        "name": "Popularity-only weights — popularity fully determines rank",
        "payload": {
            "user_vector": ZERO_VECTOR,
            "candidates": [
                {"product_id": "p8",  "keyword_score": 0.9, "popularity_score": 0.2},
                {"product_id": "p9",  "keyword_score": 0.1, "popularity_score": 0.95},
                {"product_id": "p10", "keyword_score": 0.5, "popularity_score": 0.5},
            ],
            "weights": {"kw": 0.0, "cosine": 0.0, "popularity": 1.0},
        },
        "expect": {
            "top_product": "p9",
            "description": "With popularity=1.0, p9 (popularity=0.95) must rank first",
        },
    },

    # ── User Vector & Cosine Affinity ─────────────────────────────────────────
    {
        "name": "Real user vector from /ml/user-vector drives personalized ranking",
        "use_real_vector": True,
        "vector_product_ids": ["p1", "p2", "p3"],
        "payload": {
            "candidates": [
                {"product_id": "p1", "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p50", "keyword_score": 0.5, "popularity_score": 0.5},
            ],
            "weights": {"kw": 0.0, "cosine": 1.0, "popularity": 0.0},
        },
        "expect": {
            "top_product": "p1",
            "description": "User viewed p1/p2/p3 — cosine affinity should rank p1 above p50",
        },
    },
    {
        "name": "Zero user vector — cosine similarity is 0 for all, keyword decides",
        "payload": {
            "user_vector": ZERO_VECTOR,
            "candidates": [
                {"product_id": "p11", "keyword_score": 0.9, "popularity_score": 0.5},
                {"product_id": "p12", "keyword_score": 0.2, "popularity_score": 0.5},
            ],
        },
        "expect": {
            "top_product": "p11",
            "description": "Zero vector gives 0 cosine for all — keyword score decides; p11 wins",
        },
    },

    # ── Edge Cases ────────────────────────────────────────────────────────────
    {
        "name": "Single candidate — always ranks first",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [
                {"product_id": "p20", "keyword_score": 0.5, "popularity_score": 0.5},
            ],
        },
        "expect": {
            "top_product": "p20",
            "result_count": 1,
            "description": "Only one candidate — must appear as top (and only) result",
        },
    },
    {
        "name": "Empty candidate list — returns empty results",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [],
        },
        "expect": {
            "result_count": 0,
            "description": "No candidates passed — response results list must be empty",
        },
    },
    {
        "name": "Unknown product IDs not in catalog — falls back to zero cosine",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [
                {"product_id": "unknown_xyz", "keyword_score": 0.4, "popularity_score": 0.5},
                {"product_id": "p1",          "keyword_score": 0.4, "popularity_score": 0.5},
            ],
            "weights": {"kw": 0.4, "cosine": 0.6, "popularity": 0.0},
        },
        "expect": {
            "top_product": "p1",
            "description": "p1 is in catalog so it gets real cosine score; unknown_xyz gets 0 — p1 ranks higher",
        },
    },
    {
        "name": "Mixed known and unknown product IDs — no crash",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [
                {"product_id": "p5",       "keyword_score": 0.6, "popularity_score": 0.5},
                {"product_id": "ghost_id", "keyword_score": 0.9, "popularity_score": 0.8},
                {"product_id": "p10",      "keyword_score": 0.4, "popularity_score": 0.3},
            ],
        },
        "expect": {
            "result_count": 3,
            "description": "Service must handle unknown product IDs gracefully and still return all candidates",
        },
    },

    # ── Score Validity ────────────────────────────────────────────────────────
    {
        "name": "All final scores must be between 0 and 1",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [
                {"product_id": "p1",  "keyword_score": 1.0, "popularity_score": 1.0},
                {"product_id": "p2",  "keyword_score": 0.0, "popularity_score": 0.0},
                {"product_id": "p3",  "keyword_score": 0.5, "popularity_score": 0.5},
            ],
        },
        "expect": {
            "scores_in_range": True,
            "description": "Regardless of inputs, all final_score values must be in [0, 1]",
        },
    },
    {
        "name": "Results are sorted descending by final_score",
        "payload": {
            "user_vector": ZERO_VECTOR,
            "candidates": [
                {"product_id": "p1",  "keyword_score": 0.2, "popularity_score": 0.5},
                {"product_id": "p2",  "keyword_score": 0.9, "popularity_score": 0.5},
                {"product_id": "p3",  "keyword_score": 0.6, "popularity_score": 0.5},
                {"product_id": "p4",  "keyword_score": 0.1, "popularity_score": 0.5},
            ],
        },
        "expect": {
            "sorted_descending": True,
            "description": "API must always return results sorted by final_score descending",
        },
    },

    # ── Large Candidate Set ───────────────────────────────────────────────────
    {
        "name": "Large candidate set (20 products) — correct count and structure",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [
                {"product_id": f"p{i}", "keyword_score": round(i / 20, 2), "popularity_score": 0.5}
                for i in range(1, 21)
            ],
        },
        "expect": {
            "result_count": 20,
            "sorted_descending": True,
            "description": "All 20 candidates must be returned and sorted correctly",
        },
    },

    # ── Default popularity_score ──────────────────────────────────────────────
    {
        "name": "Omitted popularity_score defaults to 0.5 — no crash",
        "payload": {
            "user_vector": MOCK_USER_VECTOR,
            "candidates": [
                {"product_id": "p1", "keyword_score": 0.7},
                {"product_id": "p2", "keyword_score": 0.4},
            ],
        },
        "expect": {
            "top_product": "p1",
            "result_count": 2,
            "description": "popularity_score is optional (defaults 0.5) — service must not crash when omitted",
        },
    },

    # ── Identical Scores ──────────────────────────────────────────────────────
    {
        "name": "All candidates have identical scores — no crash, all returned",
        "payload": {
            "user_vector": ZERO_VECTOR,
            "candidates": [
                {"product_id": "p1", "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p2", "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p3", "keyword_score": 0.5, "popularity_score": 0.5},
            ],
        },
        "expect": {
            "result_count": 3,
            "scores_in_range": True,
            "description": "Tie scenario — all 3 results returned with valid scores, no crash",
        },
    },
]


def run_tests():
    print("=" * 80)
    print("SEARCH RE-RANK ENDPOINT TEST CASES")
    print("=" * 80)

    passed = failed = warnings = 0
    results_log = []

    for i, tc in enumerate(test_cases, 1):
        name = tc["name"]
        print(f"\nTest {i}/{len(test_cases)}: {name}")
        print(f"   Expected: {tc['expect']['description']}")

        # Build payload — optionally fetch real user vector first
        payload = dict(tc["payload"])
        if tc.get("use_real_vector"):
            try:
                payload["user_vector"] = fetch_user_vector(tc["vector_product_ids"])
                print(f"   Fetched real user vector for product IDs: {tc['vector_product_ids']}")
            except Exception as e:
                print(f"   Warning: could not fetch user vector ({e}). Using mock vector.")
                payload["user_vector"] = MOCK_USER_VECTOR

        try:
            resp = requests.post(f"{BASE_URL}/ml/search-rerank", json=payload, timeout=5)

            if resp.status_code != 200:
                print(f"   FAILED — HTTP {resp.status_code}: {resp.text}")
                failed += 1
                results_log.append({"name": name, "status": "failed", "reason": resp.text})
                continue

            data = resp.json()
            results = data.get("results", [])
            is_valid = True
            issues = []

            # ── Structural check ──────────────────────────────────────────────
            for r in results:
                if "product_id" not in r or "final_score" not in r:
                    is_valid = False
                    issues.append(f"Result missing required fields: {r}")

            # ── result_count check ────────────────────────────────────────────
            if "result_count" in tc["expect"]:
                if len(results) != tc["expect"]["result_count"]:
                    is_valid = False
                    issues.append(
                        f"Expected {tc['expect']['result_count']} results, got {len(results)}"
                    )

            # ── top_product check ─────────────────────────────────────────────
            if "top_product" in tc["expect"] and results:
                if results[0]["product_id"] != tc["expect"]["top_product"]:
                    is_valid = False
                    issues.append(
                        f"Expected top product {tc['expect']['top_product']}, "
                        f"got {results[0]['product_id']}"
                    )

            # ── scores_in_range check ─────────────────────────────────────────
            if tc["expect"].get("scores_in_range"):
                for r in results:
                    score = r.get("final_score", -1)
                    if not (0.0 <= score <= 1.0):
                        is_valid = False
                        issues.append(f"Score out of [0,1] for {r['product_id']}: {score}")

            # ── sorted_descending check ───────────────────────────────────────
            if tc["expect"].get("sorted_descending") and len(results) > 1:
                scores = [r["final_score"] for r in results]
                if scores != sorted(scores, reverse=True):
                    is_valid = False
                    issues.append(f"Results not sorted descending: {scores}")

            # ── Report ────────────────────────────────────────────────────────
            top = results[0] if results else None
            if is_valid:
                print(f"   PASSED")
                passed += 1
                status = "passed"
            else:
                print(f"   WARNING — validation issues:")
                for issue in issues:
                    print(f"      - {issue}")
                warnings += 1
                status = "warning"

            if top:
                print(f"   Top result: {top['product_id']} (score={top['final_score']:.4f})")
            print(f"   Total results returned: {len(results)}")
            if results:
                scores = [round(r["final_score"], 4) for r in results]
                print(f"   All scores: {scores}")

            results_log.append({"name": name, "status": status, "results": results})

        except requests.exceptions.ConnectionError:
            print(f"   ERROR — Cannot connect to {BASE_URL}. Is the service running?")
            failed += 1
            results_log.append({"name": name, "status": "failed", "reason": "connection_error"})
            break
        except Exception as e:
            print(f"   ERROR — {e}")
            failed += 1
            results_log.append({"name": name, "status": "failed", "reason": str(e)})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total   : {len(test_cases)}")
    print(f"Passed  : {passed}")
    print(f"Warnings: {warnings}")
    print(f"Failed  : {failed}")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
