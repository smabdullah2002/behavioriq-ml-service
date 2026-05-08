# BehaviorIQ ML Integration Contract

This document is the shared contract for backend and frontend integration with the ML service.

It covers the three user-facing ML features:

- intent scoring
- churn prediction
- search reranking

The current search reranking flow is the full BehaviorIQ search pipeline:

`query -> semantic retrieval -> search intent analysis -> behavior-aware reranking`

## 1. Intent Score

### Endpoint

`POST /ml/intent-score`

### Purpose

Compute a live purchase intent score for the current session.

### Input schema

| Field | Type | Required | Notes |
|---|---|---:|---|
| `product_visit_count` | number | yes | Normalized session visit count |
| `time_on_product_page` | number | yes | Normalized time spent on product pages |
| `cart_add_events` | number | yes | Normalized cart add count |
| `scroll_depth` | number | yes | Normalized scroll depth or engagement ratio |
| `avg_spend_score` | number | yes | Normalized price affinity score |
| `session_recency` | number | yes | Normalized freshness score |

### Example request

```json
{
  "product_visit_count": 0.75,
  "time_on_product_page": 0.80,
  "cart_add_events": 0.20,
  "scroll_depth": 0.60,
  "avg_spend_score": 0.55,
  "session_recency": 0.90
}
```

### Output schema

| Field | Type | Notes |
|---|---|---|
| `intent_score` | number | Final score from 0 to 100 |
| `score_bucket` | string | Example: `hot_buyer`, `interested_hesitant`, `cold_browser`, `churn_risk` |
| `dominant_signal` | string | Strongest contributing input |
| `contributions` | object | Optional breakdown by input feature |

### Example response

```json
{
  "intent_score": 72.4,
  "score_bucket": "interested_hesitant",
  "dominant_signal": "cart_add_events",
  "contributions": {
    "product_visit_count": 0.12,
    "time_on_product_page": 0.16,
    "cart_add_events": 0.25,
    "scroll_depth": 0.06,
    "avg_spend_score": 0.08,
    "session_recency": 0.14
  }
}
```

### Frontend note

Use this endpoint when the UI or backend wants a quick live purchase-intent signal for the current session.

---

## 2. Churn Prediction

### Endpoint

`POST /ml/churn-predict`

### Purpose

Predict how likely a returning user is to churn based on RFM-style behavior.

### Input schema

| Field | Type | Required | Notes |
|---|---|---:|---|
| `days_since_last_purchase` | integer | yes | Recency in days |
| `total_order_count` | integer | yes | Total orders placed |
| `avg_order_value` | number | yes | Average order value |

### Example request

```json
{
  "days_since_last_purchase": 42,
  "total_order_count": 3,
  "avg_order_value": 78.5
}
```

### Output schema

| Field | Type | Notes |
|---|---|---|
| `churn_probability` | number | Final churn probability from 0 to 1 |
| `churn_risk_level` | string | Example: `low`, `medium`, `high`, `critical`, `error` |
| `rfm_breakdown` | object | Normalized RFM component scores |
| `recommended_action` | string | Example: `none`, `retention_offer`, `win_back_pricing` |
| `model_type` | string or null | Indicates `real_trained` or fallback mode |

### Example response

```json
{
  "churn_probability": 0.72,
  "churn_risk_level": "high",
  "rfm_breakdown": {
    "recency_score": 0.31,
    "frequency_score": 0.24,
    "monetary_score": 0.45
  },
  "recommended_action": "win_back_pricing",
  "model_type": "real_trained"
}
```

### Frontend note

Use this endpoint for returning customers only. The UI can show a churn badge, win-back offer, or owner alert using the returned risk level and action.

---

## 3. Search Rerank

### Endpoint

`POST /ml/search`

### Purpose

Run the full BehaviorIQ search pipeline:

1. semantic retrieval
2. search intent analysis
3. behavior-aware reranking

### Input schema

| Field | Type | Required | Notes |
|---|---|---:|---|
| `query` | string | yes | Raw search query from the user |
| `churn_score` | number | yes | User churn score used as a ranking signal |
| `recent_product_ids` | array of strings | no | Optional recent product history |
| `top_k` | integer | no | Number of final results to return |
| `weights` | object | no | Optional weighting override |

### Search intent object

| Field | Type | Required | Notes |
|---|---|---:|---|
| `label` | string | yes | Example: `price_sensitive`, `premium`, `comparison`, `exploratory`, `urgent_buy`, `gift` |
| `confidence` | number | yes | Confidence from 0 to 1 |
| `slots` | object | no | Optional extracted hints such as `max_price`, `brand`, `category` |

### Example request

```json
{
  "query": "cheap running shoes",
  "churn_score": 0.18,
  "recent_product_ids": ["p1", "p8", "p14"],
  "top_k": 20,
  "weights": {
    "vector": 0.45,
    "intent": 0.30,
    "pricing": 0.25
  }
}
```

### Output schema

| Field | Type | Notes |
|---|---|---|
| `query` | string | Original user query |
| `intent` | object | Detected search intent and confidence |
| `results` | array | Ranked product results |
| `search_backend` | string | Example: `pinecone_hybrid`, `offline_tfidf_fallback` |
| `total_candidates` | integer | Number of candidates before final trim |
| `pipeline_ms` | number | Total search pipeline latency in ms |

### Result item schema

| Field | Type | Notes |
|---|---|---|
| `rank` | integer | Final rank position |
| `product_id` | string | Product identifier |
| `name` | string | Product name |
| `category` | string | Product category |
| `brand` | string | Product brand |
| `price` | number | Product price |
| `discount` | number | Product discount |
| `rating` | number | Product rating |
| `final_score` | number | Final rerank score |
| `vector_score` | number | Dense + sparse vector score |
| `intent_score` | number | Query intent fit score |
| `pricing_score` | number | Pricing-aware score |
| `intent_label` | string | Detected intent label |
| `intent_confidence` | number | Intent confidence |
| `rank_explanation` | string | Short human-readable reason |

### Example response

```json
{
  "query": "cheap running shoes",
  "intent": {
    "label": "price_sensitive",
    "confidence": 0.92,
    "slots": {
      "max_price": "100",
      "category": "running_shoes"
    }
  },
  "results": [
    {
      "rank": 1,
      "product_id": "p42",
      "name": "Runner Lite",
      "category": "running_shoes",
      "brand": "Nike",
      "price": 89.99,
      "discount": 0.15,
      "rating": 4.6,
      "final_score": 0.9123,
      "vector_score": 0.8421,
      "intent_score": 0.9100,
      "pricing_score": 0.9600,
      "intent_label": "price_sensitive",
      "intent_confidence": 0.92,
      "rank_explanation": "budget pick at $89.99, 15% off, score 0.912"
    }
  ],
  "search_backend": "pinecone_hybrid",
  "total_candidates": 100,
  "pipeline_ms": 42.7
}
```

### Frontend note

The frontend should use the `rank_explanation` field to show a short personalization message, and it should read `intent_label` and `search_backend` for debugging or analytics.

---

## 4. Integration Notes

### Backend responsibilities

- Normalize input features before calling the ML service
- Cache intent scores when needed
- Call churn prediction for returning users
- Send search query, churn score, and recent product history to `/ml/search`

### Frontend responsibilities

- Send clean numeric values for intent and churn inputs
- Send the raw search query for search reranking
- Render the returned score fields directly in the UI when needed
- Use the explanation fields to show "why this result" or "why this price"

### Shared rule

The backend and frontend should treat the response schemas as stable contracts. If a field changes, update this document and the API client together.

## 5. Summary

- Use `POST /ml/intent-score` for live session intent.
- Use `POST /ml/churn-predict` for returning-user churn risk.
- Use `POST /ml/search` for the full rerank pipeline.
- The old rerank-only route is retired.
