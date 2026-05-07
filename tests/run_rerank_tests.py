import requests

BASE = "http://127.0.0.1:8001"

UV_P1_P2_P3    = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.19172639159486762,0.31663876532702656,0.19172639159486762,0.32430503406852135,0.32097595472603524]
UV_P10_P20_P30 = [0.30286280808902166,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.29421188357005934,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.29421188357005934,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.09054158914872663,0.13225765671534118,0.09054158914872663,0.0,0.301754483658479]
UV_P50_P60_P70 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2831129408326866,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.29067406534524537,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.30286280808902166,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.15772085843621106,0.0,0.08905470980560885,0.13225765671534118,0.08905470980560885,0.1711837800559948,0.0]
UV_P1_ONLY     = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.16345672806638586,0.0,0.16345672806638586,0.9729151022055641,0.0]
UV_WEIGHTED    = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.17825043936411542,0.11175485835071526,0.17825043936411542,0.5723030012973906,0.28321407769944285]

tests = [
    {
        "name": "Test 1 — Casual browser (p1,p2,p3) default weights",
        "user_label": "viewed p1, p2, p3",
        "payload": {
            "user_vector": UV_P1_P2_P3,
            "candidates": [
                {"product_id": "p1",  "keyword_score": 0.7, "popularity_score": 0.6},
                {"product_id": "p2",  "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p50", "keyword_score": 0.8, "popularity_score": 0.9},
            ],
            "weights": {"kw": 0.5, "cosine": 0.3, "popularity": 0.2},
        },
        "check": lambda r: r[0]["product_id"] in ("p1", "p2"),
        "check_desc": "p1 or p2 ranks first (cosine affinity from viewing history)",
    },
    {
        "name": "Test 2 — Cosine-only: viewed products must dominate",
        "user_label": "viewed p1,p2,p3 | kw=0.0, cosine=1.0",
        "payload": {
            "user_vector": UV_P1_P2_P3,
            "candidates": [
                {"product_id": "p1",  "keyword_score": 0.1, "popularity_score": 0.1},
                {"product_id": "p2",  "keyword_score": 0.1, "popularity_score": 0.1},
                {"product_id": "p80", "keyword_score": 0.9, "popularity_score": 0.9},
            ],
            "weights": {"kw": 0.0, "cosine": 1.0, "popularity": 0.0},
        },
        "check": lambda r: r[0]["product_id"] in ("p1", "p2"),
        "check_desc": "p1 or p2 ranks above p80 despite p80 having 0.9 keyword+popularity",
    },
    {
        "name": "Test 3 — Mid-catalog user (p10,p20,p30) vs unrelated products",
        "user_label": "viewed p10,p20,p30 | kw=0.2, cosine=0.8",
        "payload": {
            "user_vector": UV_P10_P20_P30,
            "candidates": [
                {"product_id": "p10", "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p20", "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p1",  "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p2",  "keyword_score": 0.5, "popularity_score": 0.5},
            ],
            "weights": {"kw": 0.2, "cosine": 0.8, "popularity": 0.0},
        },
        "check": lambda r: r[0]["product_id"] in ("p10", "p20"),
        "check_desc": "p10 or p20 ranks first — cosine affinity beats p1/p2",
    },
    {
        "name": "Test 4 — Recency-weighted history (p1 most recent)",
        "user_label": "viewed p1(w=1.0), p2(w=0.5), p3(w=0.2)",
        "payload": {
            "user_vector": UV_WEIGHTED,
            "candidates": [
                {"product_id": "p1",  "keyword_score": 0.6, "popularity_score": 0.7},
                {"product_id": "p2",  "keyword_score": 0.6, "popularity_score": 0.7},
                {"product_id": "p3",  "keyword_score": 0.6, "popularity_score": 0.7},
                {"product_id": "p40", "keyword_score": 0.6, "popularity_score": 0.7},
            ],
            "weights": {"kw": 0.3, "cosine": 0.6, "popularity": 0.1},
        },
        "check": lambda r: r[-1]["product_id"] == "p40",
        "check_desc": "p40 (never viewed) ranks last",
    },
    {
        "name": "Test 5 — Single product viewed (p1), narrow affinity",
        "user_label": "viewed p1 only | cosine=1.0",
        "payload": {
            "user_vector": UV_P1_ONLY,
            "candidates": [
                {"product_id": "p1",  "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p30", "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "p60", "keyword_score": 0.5, "popularity_score": 0.5},
            ],
            "weights": {"kw": 0.0, "cosine": 1.0, "popularity": 0.0},
        },
        "check": lambda r: r[0]["product_id"] == "p1",
        "check_desc": "p1 ranks first with perfect cosine match to itself",
    },
    {
        "name": "Test 6 — Keyword override (kw=0.95 ignores user affinity)",
        "user_label": "viewed p1,p2,p3 | kw=0.95",
        "payload": {
            "user_vector": UV_P1_P2_P3,
            "candidates": [
                {"product_id": "p1",  "keyword_score": 0.2, "popularity_score": 0.5},
                {"product_id": "p2",  "keyword_score": 0.3, "popularity_score": 0.5},
                {"product_id": "p99", "keyword_score": 0.95, "popularity_score": 0.5},
            ],
            "weights": {"kw": 0.95, "cosine": 0.05, "popularity": 0.0},
        },
        "check": lambda r: r[0]["product_id"] == "p99",
        "check_desc": "p99 ranks first — high keyword score wins despite user never viewing it",
    },
    {
        "name": "Test 7 — Tail user (p50,p60,p70) balanced weights",
        "user_label": "viewed p50,p60,p70 | kw=0.4, cosine=0.4, pop=0.2",
        "payload": {
            "user_vector": UV_P50_P60_P70,
            "candidates": [
                {"product_id": "p50", "keyword_score": 0.6, "popularity_score": 0.4},
                {"product_id": "p60", "keyword_score": 0.4, "popularity_score": 0.6},
                {"product_id": "p1",  "keyword_score": 0.7, "popularity_score": 0.8},
                {"product_id": "p10", "keyword_score": 0.7, "popularity_score": 0.8},
            ],
            "weights": {"kw": 0.4, "cosine": 0.4, "popularity": 0.2},
        },
        "check": lambda r: (
            all(0.0 <= x["final_score"] <= 1.0 for x in r)
            and [x["final_score"] for x in r] == sorted([x["final_score"] for x in r], reverse=True)
        ),
        "check_desc": "all scores in [0,1] and results sorted descending",
    },
    {
        "name": "Test 8 — Unknown product ID mixed with known",
        "user_label": "viewed p1,p2,p3 | cosine=0.7",
        "payload": {
            "user_vector": UV_P1_P2_P3,
            "candidates": [
                {"product_id": "p1",             "keyword_score": 0.5, "popularity_score": 0.5},
                {"product_id": "not_in_catalog",  "keyword_score": 0.9, "popularity_score": 0.9},
                {"product_id": "p2",             "keyword_score": 0.5, "popularity_score": 0.5},
            ],
            "weights": {"kw": 0.3, "cosine": 0.7, "popularity": 0.0},
        },
        "check": lambda r: r[0]["product_id"] in ("p1", "p2") and len(r) == 3,
        "check_desc": "p1 or p2 ranks first; all 3 candidates returned (no crash on unknown ID)",
    },
]


def run():
    print("=" * 72)
    print("  SEARCH RE-RANK ENDPOINT — TEST RESULTS")
    print("=" * 72)

    passed = failed = 0

    for tc in tests:
        print(f"\n{tc['name']}")
        print(f"  User  : {tc['user_label']}")
        try:
            resp = requests.post(f"{BASE}/ml/search-rerank", json=tc["payload"], timeout=5)

            if resp.status_code != 200:
                print(f"  FAIL   HTTP {resp.status_code}: {resp.text}")
                failed += 1
                continue

            results = resp.json()["results"]
            scores  = [round(r["final_score"], 4) for r in results]
            order   = [r["product_id"] for r in results]
            sorted_ok = scores == sorted(scores, reverse=True)
            check_ok  = tc["check"](results)

            status = "PASS" if check_ok else "FAIL"
            if check_ok:
                passed += 1
            else:
                failed += 1

            print(f"  {status}   {tc['check_desc']}")
            print(f"  Rank  : {order}")
            print(f"  Scores: {scores}")
            print(f"  Sorted descending: {sorted_ok}")

        except requests.exceptions.ConnectionError:
            print(f"  ERROR  Cannot connect to {BASE} — is the service running?")
            failed += 1
            break
        except Exception as e:
            print(f"  ERROR  {e}")
            failed += 1

    print()
    print("=" * 72)
    print(f"  TOTAL: {len(tests)}    PASSED: {passed}    FAILED: {failed}")
    print("=" * 72)


if __name__ == "__main__":
    run()
