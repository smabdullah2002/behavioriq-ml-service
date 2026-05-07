#!/usr/bin/env python3
"""
Thorough test suite for /ml/search-rerank endpoint.

Covers:
  - Schema / contract validation (422 errors)
  - Exact score math verification
  - Weight edge cases (non-summing, extra keys, all-zero)
  - Score boundary enforcement
  - Ordering stability
  - Large/stress inputs
  - Real user-vector integration
  - Identified bug in original Test 9
"""

import time
import requests

BASE_URL = "http://127.0.0.1:8001"

ZERO_VECTOR = [0.0] * 98   # actual TF-IDF dimension is 98
MOCK_VECTOR = [0.3 if i < 10 else 0.0 for i in range(98)]


def post(payload, timeout=5):
    return requests.post(f"{BASE_URL}/ml/search-rerank", json=payload, timeout=timeout)


def user_vector_for(product_ids):
    r = requests.post(f"{BASE_URL}/ml/user-vector",
                      json={"recent_product_ids": product_ids}, timeout=5)
    r.raise_for_status()
    return r.json()["user_vector"]


# --- helpers ------------------------------------------------------------------

def assert_status(resp, expected=200):
    if resp.status_code != expected:
        raise AssertionError(
            f"Expected HTTP {expected}, got {resp.status_code}: {resp.text[:300]}"
        )


def assert_sorted_desc(results):
    scores = [r["final_score"] for r in results]
    if scores != sorted(scores, reverse=True):
        raise AssertionError(f"Results not sorted descending: {scores}")


def assert_scores_in_range(results, lo=0.0, hi=1.0):
    for r in results:
        s = r["final_score"]
        if not (lo <= s <= hi):
            raise AssertionError(
                f"{r['product_id']} score {s:.6f} outside [{lo}, {hi}]"
            )


def assert_top(results, product_id):
    if not results:
        raise AssertionError("Empty results list")
    got = results[0]["product_id"]
    if got != product_id:
        raise AssertionError(f"Expected top={product_id}, got={got}")


def approx_equal(a, b, tol=1e-4):
    return abs(a - b) <= tol


# --- test registry ------------------------------------------------------------

TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA / CONTRACT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

@test("Missing user_vector -> 422 Unprocessable Entity")
def _():
    resp = post({"candidates": [{"product_id": "p1", "keyword_score": 0.5}]})
    assert_status(resp, 422)


@test("Missing candidates field -> 422 Unprocessable Entity")
def _():
    resp = post({"user_vector": ZERO_VECTOR})
    assert_status(resp, 422)


@test("Missing keyword_score in candidate -> 422 Unprocessable Entity")
def _():
    resp = post({"user_vector": ZERO_VECTOR,
                 "candidates": [{"product_id": "p1"}]})
    assert_status(resp, 422)


@test("user_vector as string -> 422 Unprocessable Entity")
def _():
    resp = post({"user_vector": "not_a_vector",
                 "candidates": [{"product_id": "p1", "keyword_score": 0.5}]})
    assert_status(resp, 422)


@test("product_id as number -> 422 Unprocessable Entity")
def _():
    resp = post({"user_vector": ZERO_VECTOR,
                 "candidates": [{"product_id": 42, "keyword_score": 0.5}]})
    assert_status(resp, 422)


@test("keyword_score as string -> 422 Unprocessable Entity")
def _():
    resp = post({"user_vector": ZERO_VECTOR,
                 "candidates": [{"product_id": "p1", "keyword_score": "high"}]})
    assert_status(resp, 422)


# ══════════════════════════════════════════════════════════════════════════════
# EXACT MATH VERIFICATION
# Use ZERO_VECTOR so cosine=0 and the formula is fully deterministic.
# ══════════════════════════════════════════════════════════════════════════════

@test("Exact math: kw=0.6, pop=0.4, cosine=0 — single candidate")
def _():
    kw, pop = 0.8, 0.3
    expected = 0.6 * kw + 0.0 * 0 + 0.4 * pop   # = 0.48 + 0.12 = 0.60
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": kw, "popularity_score": pop}],
        "weights": {"kw": 0.6, "cosine": 0.0, "popularity": 0.4},
    })
    assert_status(resp)
    score = resp.json()["results"][0]["final_score"]
    if not approx_equal(score, expected):
        raise AssertionError(f"Expected {expected:.4f}, got {score:.4f}")


@test("Exact math: kw=0.5, pop=0.5, cosine=0 — ordering of two candidates")
def _():
    # p_high: 0.5*0.9 + 0.5*0.1 = 0.45+0.05 = 0.50
    # p_low:  0.5*0.1 + 0.5*0.9 = 0.05+0.45 = 0.50 — tie, both equal
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [
            {"product_id": "p1", "keyword_score": 0.9, "popularity_score": 0.1},
            {"product_id": "p2", "keyword_score": 0.1, "popularity_score": 0.9},
        ],
        "weights": {"kw": 0.5, "cosine": 0.0, "popularity": 0.5},
    })
    assert_status(resp)
    results = resp.json()["results"]
    scores = {r["product_id"]: r["final_score"] for r in results}
    if not approx_equal(scores["p1"], 0.50) or not approx_equal(scores["p2"], 0.50):
        raise AssertionError(f"Expected both 0.50, got {scores}")


@test("Exact math: popularity=1.0, three candidates — scores match formula")
def _():
    candidates = [
        {"product_id": "p1", "keyword_score": 0.0, "popularity_score": 0.2},
        {"product_id": "p2", "keyword_score": 0.0, "popularity_score": 0.7},
        {"product_id": "p3", "keyword_score": 0.0, "popularity_score": 1.0},
    ]
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": candidates,
        "weights": {"kw": 0.0, "cosine": 0.0, "popularity": 1.0},
    })
    assert_status(resp)
    results = resp.json()["results"]
    scores = {r["product_id"]: r["final_score"] for r in results}
    for pid, expected in [("p1", 0.2), ("p2", 0.7), ("p3", 1.0)]:
        if not approx_equal(scores[pid], expected):
            raise AssertionError(f"{pid}: expected {expected}, got {scores[pid]:.4f}")
    assert_top(results, "p3")


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

@test("Weights sum > 1.0 (over-weighted) — endpoint handles gracefully")
def _():
    # Service should not crash; scores may exceed [0,1] but shouldn't error
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [
            {"product_id": "p1", "keyword_score": 0.5, "popularity_score": 0.5},
        ],
        "weights": {"kw": 1.0, "cosine": 1.0, "popularity": 1.0},
    })
    assert_status(resp)
    results = resp.json()["results"]
    assert len(results) == 1, "Should return 1 result"


@test("Weights sum < 1.0 — endpoint handles gracefully")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [
            {"product_id": "p1", "keyword_score": 0.5, "popularity_score": 0.5},
        ],
        "weights": {"kw": 0.1, "cosine": 0.0, "popularity": 0.1},
    })
    assert_status(resp)
    results = resp.json()["results"]
    score = results[0]["final_score"]
    expected = 0.1 * 0.5 + 0.1 * 0.5   # = 0.10
    if not approx_equal(score, expected):
        raise AssertionError(f"Expected {expected:.4f}, got {score:.4f}")


@test("All-zero weights — all scores are 0.0")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [
            {"product_id": "p1", "keyword_score": 0.9, "popularity_score": 0.9},
            {"product_id": "p2", "keyword_score": 0.1, "popularity_score": 0.1},
        ],
        "weights": {"kw": 0.0, "cosine": 0.0, "popularity": 0.0},
    })
    assert_status(resp)
    for r in resp.json()["results"]:
        if not approx_equal(r["final_score"], 0.0):
            raise AssertionError(f"Expected 0.0 for {r['product_id']}, got {r['final_score']}")


@test("Extra unknown weight keys are ignored — no crash")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": 0.5, "popularity_score": 0.5}],
        "weights": {"kw": 0.5, "cosine": 0.3, "popularity": 0.2, "bogus_key": 99.9},
    })
    assert_status(resp)


@test("Weights dict is null — defaults (kw=0.5, cosine=0.3, popularity=0.2) apply")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": 0.8, "popularity_score": 0.4}],
        "weights": None,
    })
    assert_status(resp)
    score = resp.json()["results"][0]["final_score"]
    expected = 0.5 * 0.8 + 0.0 * 0 + 0.2 * 0.4   # = 0.40 + 0.08 = 0.48
    if not approx_equal(score, expected):
        raise AssertionError(f"Expected {expected:.4f} with defaults, got {score:.4f}")


@test("Partial weights dict — missing keys fall back to per-key defaults (cosine=0.3, pop=0.2)")
def _():
    # reranker uses weights.get("cosine", 0.3) and weights.get("popularity", 0.2)
    # so partial dict {"kw": 0.5} still applies cosine=0.3 and popularity=0.2 defaults
    kw_score, pop_score = 0.6, 0.8
    # ZERO_VECTOR -> cosine_sim=0, so cosine term is 0 regardless
    expected = 0.5 * kw_score + 0.3 * 0.0 + 0.2 * pop_score  # = 0.30 + 0 + 0.16 = 0.46
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": kw_score, "popularity_score": pop_score}],
        "weights": {"kw": 0.5},
    })
    assert_status(resp)
    score = resp.json()["results"][0]["final_score"]
    if not approx_equal(score, expected):
        raise AssertionError(f"Expected {expected:.4f} (partial weights use per-key defaults), got {score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SCORE BOUNDARY / OUT-OF-RANGE INPUT SCORES
# ══════════════════════════════════════════════════════════════════════════════

@test("keyword_score=0.0, popularity_score=0.0 — final score is 0")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": 0.0, "popularity_score": 0.0}],
        "weights": {"kw": 0.5, "cosine": 0.0, "popularity": 0.5},
    })
    assert_status(resp)
    score = resp.json()["results"][0]["final_score"]
    if not approx_equal(score, 0.0):
        raise AssertionError(f"Expected 0.0, got {score}")


@test("keyword_score=1.0, popularity_score=1.0, cosine_weight=0 — final score is 1.0")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": 1.0, "popularity_score": 1.0}],
        "weights": {"kw": 0.6, "cosine": 0.0, "popularity": 0.4},
    })
    assert_status(resp)
    score = resp.json()["results"][0]["final_score"]
    if not approx_equal(score, 1.0):
        raise AssertionError(f"Expected 1.0, got {score}")


@test("Negative keyword_score — service accepts or returns 422 (no 500)")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": -0.5, "popularity_score": 0.5}],
    })
    # Accept either 200 (no clamping) or 422 (validation), but never 500
    if resp.status_code == 500:
        raise AssertionError(f"Server crashed with 500 on negative keyword_score")


@test("keyword_score > 1.0 — service accepts or returns 422 (no 500)")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": 1.5, "popularity_score": 0.5}],
    })
    if resp.status_code == 500:
        raise AssertionError(f"Server crashed with 500 on keyword_score > 1.0")


# ══════════════════════════════════════════════════════════════════════════════
# USER VECTOR DIMENSION HANDLING
# ══════════════════════════════════════════════════════════════════════════════

@test("1-dimensional user vector — no crash, cosine handled")
def _():
    resp = post({
        "user_vector": [1.0],
        "candidates": [{"product_id": "p1", "keyword_score": 0.5, "popularity_score": 0.5}],
    })
    assert_status(resp)


@test("user_vector longer than product vector (200-dim) — zero-padding handles mismatch")
def _():
    long_vector = [0.1] * 200
    resp = post({
        "user_vector": long_vector,
        "candidates": [
            {"product_id": "p1", "keyword_score": 0.5, "popularity_score": 0.5},
            {"product_id": "p2", "keyword_score": 0.3, "popularity_score": 0.5},
        ],
    })
    assert_status(resp)
    results = resp.json()["results"]
    assert len(results) == 2
    assert_sorted_desc(results)


@test("Empty user vector (length 0) — graceful response or 422")
def _():
    resp = post({
        "user_vector": [],
        "candidates": [{"product_id": "p1", "keyword_score": 0.5, "popularity_score": 0.5}],
    })
    # Should not 500
    if resp.status_code == 500:
        raise AssertionError("Server crashed with empty user_vector")


# ══════════════════════════════════════════════════════════════════════════════
# ORDERING AND STABILITY
# ══════════════════════════════════════════════════════════════════════════════

@test("5-candidate strictly-ordered keyword — rank matches keyword order")
def _():
    candidates = [
        {"product_id": f"p{i}", "keyword_score": round(i * 0.1, 1), "popularity_score": 0.5}
        for i in range(1, 6)  # kw: 0.1, 0.2, 0.3, 0.4, 0.5
    ]
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": candidates,
        "weights": {"kw": 1.0, "cosine": 0.0, "popularity": 0.0},
    })
    assert_status(resp)
    results = resp.json()["results"]
    assert_top(results, "p5")   # highest keyword
    assert_sorted_desc(results)
    assert results[-1]["product_id"] == "p1"   # lowest keyword last


@test("Reverse-ordered input — output still sorted correctly")
def _():
    # Feed candidates in ascending keyword order; verify descending output
    candidates = [
        {"product_id": f"p{i}", "keyword_score": round(i * 0.1, 1), "popularity_score": 0.0}
        for i in range(1, 8)
    ]
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": list(reversed(candidates)),
        "weights": {"kw": 1.0, "cosine": 0.0, "popularity": 0.0},
    })
    assert_status(resp)
    assert_sorted_desc(resp.json()["results"])


@test("All products unknown — all get cosine=0, keyword decides")
def _():
    resp = post({
        "user_vector": MOCK_VECTOR,
        "candidates": [
            {"product_id": "ghost_a", "keyword_score": 0.9, "popularity_score": 0.5},
            {"product_id": "ghost_b", "keyword_score": 0.3, "popularity_score": 0.5},
        ],
        "weights": {"kw": 0.6, "cosine": 0.4, "popularity": 0.0},
    })
    assert_status(resp)
    results = resp.json()["results"]
    # Both have cosine=0 so keyword decides
    assert_top(results, "ghost_a")


# ══════════════════════════════════════════════════════════════════════════════
# BUG REPRODUCTION: Original Test 9
# The original test used MOCK_USER_VECTOR (100-dim) with weights kw=0.4, cosine=0.6.
# With equal kw and pop scores, ranking depends on cosine similarity.
# MOCK_VECTOR (dims 0-9 = 0.3) may have near-zero cosine with real product TF-IDF
# vectors whose significant terms map to other dimensions. The fix: use a real
# user vector built from viewing p1.
# ══════════════════════════════════════════════════════════════════════════════

@test("BUG REPRO (original test 9): known vs unknown product — use real user vector")
def _():
    """Original test 9 assumed MOCK_USER_VECTOR has positive cosine with p1.
    This test correctly fetches a real user vector from /ml/user-vector for p1."""
    real_vector = user_vector_for(["p1"])
    resp = post({
        "user_vector": real_vector,
        "candidates": [
            {"product_id": "unknown_xyz", "keyword_score": 0.4, "popularity_score": 0.5},
            {"product_id": "p1",          "keyword_score": 0.4, "popularity_score": 0.5},
        ],
        "weights": {"kw": 0.4, "cosine": 0.6, "popularity": 0.0},
    })
    assert_status(resp)
    results = resp.json()["results"]
    assert_top(results, "p1")


@test("Known product has higher cosine than unknown when user vector reflects it")
def _():
    """Build user from p5, p6, p7 — those products should score higher via cosine."""
    real_vector = user_vector_for(["p5", "p6", "p7"])
    resp = post({
        "user_vector": real_vector,
        "candidates": [
            {"product_id": "p5",      "keyword_score": 0.5, "popularity_score": 0.5},
            {"product_id": "ghost99", "keyword_score": 0.5, "popularity_score": 0.5},
        ],
        "weights": {"kw": 0.0, "cosine": 1.0, "popularity": 0.0},
    })
    assert_status(resp)
    results = resp.json()["results"]
    scores = {r["product_id"]: r["final_score"] for r in results}
    if scores["p5"] <= scores["ghost99"]:
        raise AssertionError(
            f"Expected p5 (catalog) to score higher than ghost99 via cosine. "
            f"p5={scores['p5']:.4f}, ghost99={scores['ghost99']:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# STRESS / PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

@test("50-candidate set — all returned, sorted, response < 500ms")
def _():
    candidates = [
        {"product_id": f"p{i}", "keyword_score": round((i % 10) / 10, 1), "popularity_score": 0.5}
        for i in range(1, 51)
    ]
    start = time.perf_counter()
    resp = post({"user_vector": MOCK_VECTOR, "candidates": candidates})
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert_status(resp)
    results = resp.json()["results"]
    assert len(results) == 50, f"Expected 50 results, got {len(results)}"
    assert_sorted_desc(results)
    if elapsed_ms > 500:
        raise AssertionError(f"Response took {elapsed_ms:.0f}ms (threshold: 500ms)")


@test("100-candidate set with mix of known/unknown — no crash, all returned")
def _():
    candidates = [
        {"product_id": f"p{i}" if i <= 50 else f"ghost{i}",
         "keyword_score": round((i % 10) / 10, 1),
         "popularity_score": 0.5}
        for i in range(1, 101)
    ]
    resp = post({
        "user_vector": MOCK_VECTOR,
        "candidates": candidates,
    })
    assert_status(resp)
    results = resp.json()["results"]
    assert len(results) == 100, f"Expected 100, got {len(results)}"
    assert_sorted_desc(results)


@test("Rapid successive calls — service handles 10 back-to-back requests")
def _():
    payload = {
        "user_vector": MOCK_VECTOR,
        "candidates": [
            {"product_id": "p1", "keyword_score": 0.7, "popularity_score": 0.5},
            {"product_id": "p2", "keyword_score": 0.4, "popularity_score": 0.5},
        ],
    }
    for i in range(10):
        resp = post(payload)
        assert_status(resp, 200)


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

@test("Response contains only 'results' key at top level")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": 0.5}],
    })
    assert_status(resp)
    data = resp.json()
    assert "results" in data, "Missing 'results' key"
    unexpected = set(data.keys()) - {"results"}
    if unexpected:
        raise AssertionError(f"Unexpected top-level keys: {unexpected}")


@test("Each result has exactly 'product_id' and 'final_score' fields")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [
            {"product_id": "p1", "keyword_score": 0.5},
            {"product_id": "p2", "keyword_score": 0.7},
        ],
    })
    assert_status(resp)
    for r in resp.json()["results"]:
        keys = set(r.keys())
        if keys != {"product_id", "final_score"}:
            raise AssertionError(f"Unexpected result keys: {keys}")


@test("final_score is a float (not int or string)")
def _():
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": "p1", "keyword_score": 0.5}],
    })
    assert_status(resp)
    score = resp.json()["results"][0]["final_score"]
    if not isinstance(score, float):
        raise AssertionError(f"final_score type is {type(score).__name__}, expected float")


@test("product_id preserved exactly in response (case-sensitive)")
def _():
    pid = "MyProduct_XYZ-123"
    resp = post({
        "user_vector": ZERO_VECTOR,
        "candidates": [{"product_id": pid, "keyword_score": 0.5}],
    })
    assert_status(resp)
    returned_pid = resp.json()["results"][0]["product_id"]
    if returned_pid != pid:
        raise AssertionError(f"product_id mutated: sent '{pid}', got '{returned_pid}'")


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 80)
    print("THOROUGH RERANK ENDPOINT TEST SUITE")
    print(f"Target: {BASE_URL}/ml/search-rerank")
    print("=" * 80)

    passed = failed = 0

    for name, fn in TESTS:
        print(f"\n  {'-'*70}")
        print(f"  TEST: {name}")
        try:
            fn()
            print(f"  PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED — {e}")
            failed += 1
        except requests.exceptions.ConnectionError:
            print(f"  ERROR — Cannot connect to {BASE_URL}. Is the server running?")
            failed += 1
            break
        except Exception as e:
            print(f"  ERROR — Unexpected: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*80}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
    print(f"{'='*80}")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run()
    sys.exit(0 if success else 1)
