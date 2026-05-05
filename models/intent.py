"""Intent score computation module."""

import numpy as np

# Intent score weights (from design doc)
INTENT_WEIGHTS = {
    "time_on_product_page": 0.20,
    "product_visit_count": 0.15,
    "cart_add_events": 0.25,
    "search_to_view_ratio": 0.10,
    "price_range_affinity": 0.15,
    "session_recency": 0.15,
}

# Score bucket thresholds
SCORE_BUCKETS = [
    (80, "hot_buyer"),
    (55, "interested_hesitant"),
    (30, "cold_browser"),
    (0,  "churn_risk"),
]


def intent_score(features: dict) -> dict:

    s = 0.0
    contributions = {}

    for k, w in INTENT_WEIGHTS.items():
        val = features.get(k, 0)
        # clamp feature value to [0,1]
        try:
            val = float(val)
        except Exception:
            val = 0.0
        val = max(0.0, min(1.0, val))
        contrib = val * w
        contributions[k] = round(contrib, 4)
        s += contrib

    score = max(0.0, min(100.0, s * 100))

    # Determine score bucket
    bucket = "churn_risk"
    for threshold, label in SCORE_BUCKETS:
        if score >= threshold:
            bucket = label
            break

    # Find dominant signal (feature with highest contribution)
    dominant = max(contributions, key=contributions.get) if contributions else "none"

    return {
        "intent_score": round(score, 2),
        "score_bucket": bucket,
        "dominant_signal": dominant,
        "contributions": contributions,
    }
