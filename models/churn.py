"""Churn probability prediction module using RFM model."""

import numpy as np


def normalize_r(value, max_days=365):
    """Normalize recency (days since last purchase)."""
    return max(1.0, 5.0 * (1 - min(value, max_days) / max_days))


def normalize_f(value, max_orders=50):
    """Normalize frequency (total order count)."""
    return max(1.0, min(5.0, 1 + (value / max_orders) * 4))


def normalize_m(value, max_val=500.0):
    """Normalize monetary (average order value)."""
    return max(1.0, min(5.0, 1 + (value / max_val) * 4))


def churn_probability(days_since_last_purchase: int, 
                      total_order_count: int, 
                      avg_order_value: float) -> float:
    """
    Compute churn probability (0-1) from RFM features.
    
    Args:
        days_since_last_purchase: recency input
        total_order_count: frequency input
        avg_order_value: monetary input
    
    Returns:
        float: churn probability from 0 to 1
    """
    r = normalize_r(days_since_last_purchase)
    f = normalize_f(total_order_count)
    m = normalize_m(avg_order_value)
    
    composite = (r * 0.4) + (f * 0.35) + (m * 0.25)
    prob = 1 - (composite / 5.0)
    
    return float(max(0.0, min(1.0, prob)))
