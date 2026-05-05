"""Preprocessing utilities for ML models.

This module normalizes raw intent-related features into the 0.0-1.0
range expected by `intent_score()`.

Assumptions (sensible defaults for hackathon/demo):
- `product_visit_count`: integer count -> scaled via log1p / log1p(20)
- `time_on_product_page`: seconds -> capped at 120s (2 minutes)
- `cart_add_events`: integer count -> capped at 5
- `search_to_view_ratio`: ratio -> clamped 0..1
- `price_range_affinity`: already 0..1 or percent -> clamped 0..1
- `session_recency`: hours since last session (or 0..1 recentness) -> decays over 7 days

These heuristics are documented and easy to tweak.
"""

from math import log1p, exp
from typing import Dict, Any


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def normalize_product_visit_count(count: Any, cap: int = 20) -> float:
    try:
        c = float(count)
    except Exception:
        return 0.0
    if c <= 0:
        return 0.0
    return _clamp01(log1p(c) / log1p(cap))


def normalize_time_on_product_page(seconds: Any, cap_seconds: int = 120) -> float:
    try:
        s = float(seconds)
    except Exception:
        return 0.0
    if s <= 0:
        return 0.0
    return _clamp01(s / cap_seconds)


def normalize_cart_add_events(count: Any, cap: int = 5) -> float:
    try:
        c = float(count)
    except Exception:
        return 0.0
    if c <= 0:
        return 0.0
    return _clamp01(c / cap)


def normalize_search_to_view_ratio(ratio: Any) -> float:
    try:
        r = float(ratio)
    except Exception:
        return 0.0
    return _clamp01(r)


def normalize_price_range_affinity(val: Any) -> float:
    try:
        v = float(val)
    except Exception:
        return 0.0
    # If user passed a percent (0-100), convert
    if v > 1.0:
        v = v / 100.0
    return _clamp01(v)


def normalize_session_recency(val: Any, decay_days: int = 7) -> float:
    """
    Accepts either a normalized recency (0..1) or hours/days since last session.
    If `val` > 1 we treat it as hours since last session and apply an exponential decay.
    """
    try:
        v = float(val)
    except Exception:
        return 0.0
    if v <= 1.0 and v >= 0.0:
        return _clamp01(v)
    # treat as hours since last activity
    hours = v
    days = hours / 24.0
    # decayed recentness: exp(-days/decay_days)
    recentness = exp(-days / float(decay_days))
    return _clamp01(recentness)


def normalize_intent_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Normalize a dict of raw intent features into the expected schema.

    Returns a dict with the keys required by `intent_score()`.
    """
    return {
        "time_on_product_page": normalize_time_on_product_page(raw.get("time_on_product_page", 0)),
        "product_visit_count": normalize_product_visit_count(raw.get("product_visit_count", 0)),
        "cart_add_events": normalize_cart_add_events(raw.get("cart_add_events", 0)),
        "search_to_view_ratio": normalize_search_to_view_ratio(raw.get("search_to_view_ratio", 0)),
        "price_range_affinity": normalize_price_range_affinity(raw.get("price_range_affinity", 0)),
        "session_recency": normalize_session_recency(raw.get("session_recency", 0)),
    }
