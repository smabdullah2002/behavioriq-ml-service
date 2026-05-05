"""Model-based churn prediction helper."""

from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


MODEL_PATH = Path("saved_models/churn_model.pkl")
MAX_DAYS = 365.0
MAX_ORDERS = 50.0
MAX_VALUE = 500.0

RISK_LEVELS = [
    (0.8, "critical", "win_back_pricing"),
    (0.6, "high", "win_back_pricing"),
    (0.3, "medium", "retention_offer"),
    (0.0, "low", "none"),
]


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def normalize_rfm(days_since_last_purchase: Any, total_order_count: Any, avg_order_value: Any) -> Tuple[float, float, float]:
    """Turn raw churn inputs into 0-1 scores."""
    try:
        days = float(days_since_last_purchase)
    except Exception:
        days = 0.0
    try:
        orders = float(total_order_count)
    except Exception:
        orders = 0.0
    try:
        value = float(avg_order_value)
    except Exception:
        value = 0.0

    recency_score = _clamp(1.0 - (days / MAX_DAYS))
    frequency_score = _clamp(orders / MAX_ORDERS)
    monetary_score = _clamp(value / MAX_VALUE)
    return recency_score, frequency_score, monetary_score


def _build_synthetic_training_data(seed: int = 42, size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Create a small fallback dataset so the service can run without a saved model."""
    rng = np.random.default_rng(seed)
    days = rng.integers(0, int(MAX_DAYS) + 1, size=size)
    orders = rng.integers(0, int(MAX_ORDERS) + 1, size=size)
    values = rng.uniform(0, MAX_VALUE, size=size)

    recency = np.clip(1.0 - (days / MAX_DAYS), 0.0, 1.0)
    frequency = np.clip(orders / MAX_ORDERS, 0.0, 1.0)
    monetary = np.clip(values / MAX_VALUE, 0.0, 1.0)
    X = np.column_stack([recency, frequency, monetary])

    churn_pressure = (1.8 * (1.0 - recency)) + (1.5 * (1.0 - frequency)) + (1.0 * (1.0 - monetary))
    y = (churn_pressure > 1.9).astype(int)
    return X, y


def load_or_train_model() -> LogisticRegression:
    """Load the churn model if it exists, otherwise train a small fallback model."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    X, y = _build_synthetic_training_data()
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def predict_churn(model: LogisticRegression, days_since_last_purchase: Any, total_order_count: Any, avg_order_value: Any) -> Dict[str, Any]:
    """Return churn score, risk level, breakdown, and action."""
    recency_score, frequency_score, monetary_score = normalize_rfm(
        days_since_last_purchase,
        total_order_count,
        avg_order_value,
    )

    features = np.array([[recency_score, frequency_score, monetary_score]])
    probability = float(model.predict_proba(features)[0][1])

    risk_level = "low"
    recommended_action = "none"
    for threshold, level, action in RISK_LEVELS:
        if probability >= threshold:
            risk_level = level
            recommended_action = action
            break

    return {
        "churn_probability": round(probability, 4),
        "churn_risk_level": risk_level,
        "rfm_breakdown": {
            "recency_score": round(recency_score, 4),
            "frequency_score": round(frequency_score, 4),
            "monetary_score": round(monetary_score, 4),
        },
        "recommended_action": recommended_action,
    }