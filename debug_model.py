#!/usr/bin/env python3
"""Debug script to check model and scaler loading."""

from pathlib import Path
import joblib

MODEL_PATH = Path("saved_models/churn_model.pkl")
SCALER_PATH = Path("saved_models/churn_scaler.pkl")

print("=" * 80)
print("🔍 MODEL & SCALER DEBUG")
print("=" * 80)

# Check if files exist
print(f"\n📁 File Check:")
print(f"   Model exists: {MODEL_PATH.exists()} ({MODEL_PATH})")
print(f"   Scaler exists: {SCALER_PATH.exists()} ({SCALER_PATH})")

# Check file sizes
if MODEL_PATH.exists():
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"   Model size: {size_kb:.2f} KB")

if SCALER_PATH.exists():
    size_kb = SCALER_PATH.stat().st_size / 1024
    print(f"   Scaler size: {size_kb:.2f} KB")

# Try loading
print(f"\n📦 Loading Check:")
try:
    model = joblib.load(MODEL_PATH)
    print(f"   ✅ Model loaded: {type(model).__name__}")
    print(f"   Coefficients: {model.coef_[0]}")
    print(f"   Intercept: {model.intercept_[0]}")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"   ✅ Scaler loaded: {type(scaler).__name__}")
    print(f"   Scale: {scaler.scale_}")
    print(f"   Mean: {scaler.mean_}")
except Exception as e:
    print(f"   ❌ Scaler loading failed: {e}")

# Test prediction with raw values
print(f"\n🧪 Test Prediction:")
try:
    import numpy as np
    
    # Test case: 60 days, 2 orders, $40 avg
    recency_score = 0.8356  # Normalized 0-1
    frequency_score = 0.04
    monetary_score = 0.08
    
    features_raw = np.array([[recency_score, frequency_score, monetary_score]])
    print(f"   Raw features (0-1): {features_raw[0]}")
    
    # Without scaler
    prob_raw = model.predict_proba(features_raw)[0][1]
    print(f"   Probability (raw): {prob_raw:.4f}")
    
    # With scaler (if available)
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        features_scaled = scaler.transform(features_raw)
        print(f"   Scaled features: {features_scaled[0]}")
        prob_scaled = model.predict_proba(features_scaled)[0][1]
        print(f"   Probability (scaled): {prob_scaled:.4f}")
    
except Exception as e:
    print(f"   ❌ Prediction test failed: {e}")

print("\n" + "=" * 80)
