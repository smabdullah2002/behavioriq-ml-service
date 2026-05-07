# Testcases Guide — Intent + Churn

This guide explains how to run the two test suites in this repository:
- `test_intent_endpoint.py` — intent score endpoint tests
- `test_churn_endpoint.py` — churn prediction endpoint tests

Both tests assume the ML service is running locally (FastAPI app). Default host/port used by test scripts:
- `http://127.0.0.1:8000`

Files location (workspace-relative):
- [test_intent_endpoint.py](test_intent_endpoint.py)
- [test_churn_endpoint.py](test_churn_endpoint.py)
- [test_churn_comprehensive.py](test_churn_comprehensive.py) — optional comprehensive churn suite
- [test_churn_datagen.py](test_churn_datagen.py) — optional data generator

Quick Start

1) Start the ML service (from `ml-service` directory):

```powershell
python main.py
# or
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

2) Run the intent tests:

```powershell
python test_intent_endpoint.py
```

3) Run the churn tests (basic):

```powershell
python test_churn_endpoint.py
```

Optional: run the comprehensive churn suite (saves JSON results):

```powershell
python test_churn_comprehensive.py
# results saved to test_results/ by timestamp
```

What each file does

- `test_intent_endpoint.py` — sends 5 predefined sessions to `/ml/intent-score`, verifies response structure, prints contributions and bucket, and reports pass/fail per case.

- `test_churn_endpoint.py` — sends ~15 predefined RFM profiles to `/ml/churn-predict`, validates response fields (`churn_probability` in [0,1], `rfm_breakdown` scores, required keys), prints per-test details and a summary (pass/warn/fail, min/max/mean probabilities, risk distribution).

- `test_churn_comprehensive.py` — category-organized tests covering loyalty, risk, edge cases, and RFM variations; measures response times and writes a JSON report to `test_results/`.

- `test_churn_datagen.py` — CLI tool to generate realistic test cases by customer segment (useful to feed into bulk tests).

Notes & Troubleshooting

- If you get a Windows socket permission error starting on port `8001`, use `--port 8000` or run with elevated permissions. Test scripts default to port `8000` in this workspace.
- If a test reports a 4xx error with a missing field, verify the test case payload keys match the schema (`avg_order_value` vs `avg_order_count` typo can cause 422).
- The comprehensive test writes JSON to `test_results/` — check that folder for saved results.

Want CSV export or a filtered report for specific tests? I can add a small script to extract selected cases from the JSON output.
