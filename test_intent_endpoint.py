#!/usr/bin/env python3
"""Test the intent-score endpoint with all test cases from test-cases.md."""

import requests

BASE_URL = "http://127.0.0.1:8000"

test_cases = [
    {
        "name": "Empty session",
        "data": {
            "product_visit_count": 0,
            "time_on_product_page": 0,
            "cart_add_events": 0,
            "scroll_depth": 0,
            "avg_spend_score": 0,
            "session_recency": 0,
        },
        "expected": "Score ~0, all contributions near 0",
        "checks": lambda r: r["intent_score"] < 5,
    },
    {
        "name": "Light browsing",
        "data": {
            "product_visit_count": 2,
            "time_on_product_page": 18,
            "cart_add_events": 0,
            "scroll_depth": 0.2,
            "avg_spend_score": 0.3,
            "session_recency": 24,
        },
        "expected": "Low score, small contributions from views and page time",
        "checks": lambda r: r["intent_score"] < 40,
    },
    {
        "name": "Strong purchase intent",
        "data": {
            "product_visit_count": 12,
            "time_on_product_page": 95,
            "cart_add_events": 3,
            "scroll_depth": 0.8,
            "avg_spend_score": 0.7,
            "session_recency": 2,
        },
        "expected": "High score, cart adds and page time drive it up",
        "checks": lambda r: r["intent_score"] >= 55,
    },
    {
        "name": "Cart-heavy user",
        "data": {
            "product_visit_count": 5,
            "time_on_product_page": 20,
            "cart_add_events": 5,
            "scroll_depth": 0.4,
            "avg_spend_score": 0.5,
            "session_recency": 1,
        },
        "expected": "Very high score, cart_add_events should be dominant signal",
        "checks": lambda r: r["intent_score"] >= 55 and r["dominant_signal"] == "cart_add_events",
    },
    {
        "name": "Out-of-range input",
        "data": {
            "product_visit_count": 50,
            "time_on_product_page": 1000,
            "cart_add_events": 20,
            "scroll_depth": 2.5,
            "avg_spend_score": 120,
            "session_recency": -5,
        },
        "expected": "Score still in 0-100 range after clamping/normalization",
        "checks": lambda r: 0 <= r["intent_score"] <= 100,
    },
]

def run():
    print("=" * 80)
    print("INTENT SCORE ENDPOINT TEST CASES")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, tc in enumerate(test_cases, 1):
        print(f"\nTest {i}: {tc['name']}")
        print(f"   Expected: {tc['expected']}")
        try:
            resp = requests.post(f"{BASE_URL}/ml/intent-score", json=tc["data"], timeout=5)
            if resp.status_code == 200:
                r = resp.json()
                ok = tc["checks"](r)
                status = "PASS" if ok else "FAIL"
                if ok:
                    passed += 1
                else:
                    failed += 1
                print(f"   [{status}] Status: 200 OK")
                print(f"   Intent Score:     {r['intent_score']}")
                print(f"   Score Bucket:     {r['score_bucket']}")
                print(f"   Dominant Signal:  {r['dominant_signal']}")
                print(f"   Contributions:")
                for k, v in (r.get("contributions") or {}).items():
                    print(f"      {k}: {v}")
            else:
                failed += 1
                print(f"   [FAIL] HTTP {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"   [ERROR] Cannot connect to {BASE_URL}. Is the service running?")
            break
        except Exception as e:
            failed += 1
            print(f"   [ERROR] {e}")

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

if __name__ == "__main__":
    run()
