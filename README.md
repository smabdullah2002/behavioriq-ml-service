# BehaviorIQ ML Service

A lightweight, hackathon-friendly FastAPI service for behavioral commerce ML.

## Quick Start

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the service

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Test an endpoint

```bash
curl -X POST http://localhost:8001/ml/intent-score \
  -H 'Content-Type: application/json' \
  -d '{
    "product_visit_count": 5,
    "time_on_product_page": 120,
    "cart_add_events": 1,
    "scroll_depth": 0.2,
    "avg_spend_score": 0.7,
    "session_recency": 0.6
  }'
```

## Project Structure

```
ml-service/
  main.py                   # FastAPI app, all routes
  models/
    __init__.py
    intent.py               # Intent score logic
    churn.py                # RFM churn model
    embedder.py             # TF-IDF product embedder
    reranker.py             # Search re-ranking
  schemas/
    __init__.py
    requests.py             # All Pydantic models
  data/
    __init__.py
    synthetic_churn.py      # Synthetic data generator
  saved_models/
    __init__.py
    churn_model.pkl
    embedder.pkl
  requirements.txt
  README.md
```

## API Endpoints

### POST /ml/intent-score
Compute user intent score (0-100) from session features.

**Request:**
```json
{
  "product_visit_count": 5,
  "time_on_product_page": 120,
  "cart_add_events": 1,
  "scroll_depth": 0.25,
  "avg_spend_score": 0.7,
  "session_recency": 0.8
}
```

**Response:**
```json
{
  "intent_score": 62.3
}
```

### POST /ml/churn-predict
Predict churn probability (0-1) using RFM features.

**Request:**
```json
{
  "days_since_last_purchase": 42,
  "total_order_count": 3,
  "avg_order_value": 78.5
}
```

**Response:**
```json
{
  "churn_probability": 0.72
}
```

### POST /ml/user-vector
Build user behavioral vector from recent product views.

**Request:**
```json
{
  "recent_product_ids": ["p1", "p5", "p8"],
  "weights": [1.0, 0.5, 0.8]
}
```

**Response:**
```json
{
  "user_vector": [0.04, 0.0, 0.3, ...]
}
```

### POST /ml/search-rerank
Re-rank search candidates using user vector.

**Request:**
```json
{
  "user_vector": [0.04, 0.0, 0.3, ...],
  "candidates": [
    { "product_id": "p1", "keyword_score": 0.76, "popularity_score": 0.5 },
    { "product_id": "p2", "keyword_score": 0.64, "popularity_score": 0.3 }
  ],
  "weights": { "kw": 0.5, "cosine": 0.3, "popularity": 0.2 }
}
```

**Response:**
```json
{
  "results": [
    { "product_id": "p1", "final_score": 0.82 },
    { "product_id": "p2", "final_score": 0.76 }
  ]
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Implementation Notes

- Models are kept in memory and loaded at startup.
- Synthetic data is generated on first run if saved models don't exist.
- All scoring functions are deterministic for reproducible demos.
- Schemas and requests are defined in Pydantic for automatic validation.

## Demo Tips

- Seed with ~100 synthetic products for fast demo iteration.
- Use deterministic coefficients for intent/churn so behavior is explainable.
- Keep LLM calls async and non-blocking in production.
- Add rate-limiting and API key auth for deployment.
"# behavioriq-ml-service" 


