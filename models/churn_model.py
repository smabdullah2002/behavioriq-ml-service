"""Model-based churn prediction helper."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from logger import get_logger

logger = get_logger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "saved_models" / "churn_scaler.pkl"
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


def _calibrated_risk_score(days: float, orders: float, value: float) -> float:
    """Formula-based risk score calibrated for SME e-commerce demo ranges.

    Caps recency at 90 days (beyond that = fully churned), orders at 20,
    and monetary at 200. These ranges give smooth graduated probabilities
    across the expected input space and match the demo personas.
    """
    r = max(1.0, 5.0 * (1.0 - min(days, 90.0) / 90.0))
    f = max(1.0, min(5.0, 1.0 + (orders / 20.0) * 4.0))
    m = max(1.0, min(5.0, 1.0 + (value / 200.0) * 4.0))
    composite = (r * 0.4) + (f * 0.35) + (m * 0.25)
    return _clamp(1.0 - composite / 5.0)


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


def load_or_train_model() -> Tuple[LogisticRegression, Optional[StandardScaler]]:
    """Load the churn model if it exists, otherwise train a small fallback model.
    Returns (model, scaler) where scaler is None for fallback models."""
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = None
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
        logger.info(f"Loaded REAL churn model from {MODEL_PATH}")
        logger.info(f"Loaded scaler from {SCALER_PATH}" if scaler else "No scaler found (using raw 0-1 normalization)")
        return model, scaler
    # Fallback: train on synthetic data
    logger.warning("No saved model found. Training FALLBACK model on synthetic data...")
    X, y = _build_synthetic_training_data()
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.warning("FALLBACK model saved. Run 'python -m data.train' to train on real Kaggle data.")
    return model, None


def predict_churn(model: LogisticRegression, days_since_last_purchase: Any, total_order_count: Any, avg_order_value: Any, scaler: Optional[StandardScaler] = None) -> Dict[str, Any]:
    """Return churn score, risk level, breakdown, and action.
    If scaler is provided, uses raw values scaled (real trained model).
    Otherwise uses manual 0-1 normalization (fallback model)."""
    
    # If scaler exists (real model), use raw values directly
    if scaler is not None:
        try:
            days = float(days_since_last_purchase)
        except Exception:
            days = 0.0
        # Match training transform: use log1p(days) so model is sensitive in mid-range
        days_log = float(np.log1p(days))
        try:
            orders = float(total_order_count)
        except Exception:
            orders = 0.0
        try:
            value = float(avg_order_value)
        except Exception:
            value = 0.0

        # Build features to match training pipeline: [recency_log, frequency, monetary, interaction]
        interaction = days_log * orders
        features = np.array([[days_log, orders, value, interaction]])
        features = scaler.transform(features)
        ml_probability = float(model.predict_proba(features)[0][1])
        model_type = "real_trained"
    else:
        # Fallback: use manual 0-1 normalization
        recency_score, frequency_score, monetary_score = normalize_rfm(
            days_since_last_purchase,
            total_order_count,
            avg_order_value,
        )
        features = np.array([[recency_score, frequency_score, monetary_score]])
        ml_probability = float(model.predict_proba(features)[0][1])
        model_type = "fallback_synthetic"

    # The Kaggle-trained model only fires confidently above ~90 days because
    # its training distribution spans 0-738 days. Blend with a formula calibrated
    # to 90-day churn logic so intermediate cases (20-60 days) get graduated risk.
    calibrated = _calibrated_risk_score(
        float(days_since_last_purchase),
        float(total_order_count),
        float(avg_order_value),
    )
    probability = max(ml_probability, calibrated)

    # Calculate 0-1 breakdown for response (always use normalize_rfm for consistency)
    recency_score, frequency_score, monetary_score = normalize_rfm(
        days_since_last_purchase,
        total_order_count,
        avg_order_value,
    )

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
        "model_type": model_type,  # DEBUG: shows which model is being used
    }