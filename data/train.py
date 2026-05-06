"""
Train churn model on real Kaggle Online Retail dataset.
Reads churn_data.csv, engineers RFM features, labels churn, trains LogisticRegression.

Dataset schema:
  Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import joblib
from typing import Tuple, Any


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load transaction data and clean it for RFM analysis."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"\nData Cleaning:")
    print(f"  Initial rows: {len(df)}")
    
    # Step 1: Remove rows with missing critical fields
    df = df[df['Customer ID'].notna()]
    df = df[df['InvoiceDate'].notna()]
    df = df[df['Quantity'].notna()]
    df = df[df['Price'].notna()]
    print(f"  After removing nulls: {len(df)}")
    
    # Step 2: Remove cancelled orders (negative quantities)
    initial_rows = len(df)
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    print(f"  After removing negatives/zeros: {len(df)} (removed {initial_rows - len(df)})")
    
    # Step 3: Remove duplicate invoices (same customer, same timestamp, same items)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['Invoice', 'StockCode', 'Customer ID'], keep='first')
    print(f"  After removing duplicates: {len(df)} (removed {initial_rows - len(df)})")
    
    # Step 4: Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Step 5: Calculate transaction amount
    df['Amount'] = df['Quantity'] * df['Price']
    
    # Step 6: Remove outliers (very high prices per unit)
    price_q99 = df['Price'].quantile(0.99)
    initial_rows = len(df)
    df = df[df['Price'] <= price_q99]
    print(f"  After removing price outliers (>99%ile): {len(df)} (removed {initial_rows - len(df)})")
    
    print(f"\n✅ Final dataset: {len(df)} transactions from {df['Customer ID'].nunique()} unique customers")
    print(f"   Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    return df


def engineer_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer RFM (Recency, Frequency, Monetary) features for each customer.
    
    - Recency: Days since last purchase (relative to snapshot date)
    - Frequency: Number of unique purchase occasions
    - Monetary: Total spend (sum of all transaction amounts)
    
    Features are kept in RAW UNITS to match the ML service's normalize_rfm() function:
      - recency: days (int, 0-365+)
      - frequency: count (int, 0-50+)
      - monetary: GBP (float, 0-500+)
    """
    print(f"\n📈 Engineering RFM Features:")
    
    # Define snapshot date = latest date in dataset
    snapshot_date = df['InvoiceDate'].max()
    print(f"  Snapshot date: {snapshot_date}")
    
    # Group by customer and calculate RFM
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,      # Recency: days since last purchase
        'Invoice': 'nunique',                                          # Frequency: unique invoices
        'Amount': 'sum'                                                # Monetary: total spend
    }).reset_index()
    
    # Rename columns to match ml_service expectations
    rfm.columns = ['customer_id', 'days_since_last_purchase', 'total_order_count', 'total_spend']
    
    # Calculate avg_order_value (total_spend / total_orders)
    rfm['avg_order_value'] = rfm['total_spend'] / rfm['total_order_count']
    rfm = rfm[['customer_id', 'days_since_last_purchase', 'total_order_count', 'avg_order_value']]
    
    print(f"\n  📊 RFM Statistics:")
    print(f"    Recency (days):")
    print(f"      Min: {rfm['days_since_last_purchase'].min()}, Max: {rfm['days_since_last_purchase'].max()}, Median: {rfm['days_since_last_purchase'].median():.0f}")
    print(f"    Frequency (purchases):")
    print(f"      Min: {rfm['total_order_count'].min()}, Max: {rfm['total_order_count'].max()}, Median: {rfm['total_order_count'].median():.0f}")
    print(f"    Monetary (£):")
    print(f"      Min: £{rfm['avg_order_value'].min():.2f}, Max: £{rfm['avg_order_value'].max():.2f}, Median: £{rfm['avg_order_value'].median():.2f}")
    
    return rfm


def label_churn(rfm: pd.DataFrame, churn_threshold_days: int = 90) -> pd.DataFrame:
    """
    Label customers as churned if inactive for churn_threshold_days (default 6 months).
    
    Churn = 1 if recency > threshold, else 0
    """
    rfm['churn'] = (rfm['days_since_last_purchase'] > churn_threshold_days).astype(int)
    
    churn_count = rfm['churn'].sum()
    retention_count = len(rfm) - churn_count
    churn_rate = (churn_count / len(rfm)) * 100
    
    print(f"\n🎯 Churn Labeling (threshold: {churn_threshold_days} days inactivity):")
    print(f"   Churned: {churn_count} ({churn_rate:.2f}%)")
    print(f"   Retained: {retention_count} ({100-churn_rate:.2f}%)")
    
    return rfm


def train_churn_model(rfm: pd.DataFrame, model_output_path: str) -> Tuple[CalibratedClassifierCV, Any]:
    """
    Train LogisticRegression model on RFM features.
    
    Input features (raw units):
    - days_since_last_purchase (recency, 0-365+)
    - total_order_count (frequency, 0-50+)
    - avg_order_value (monetary, 0-500+ GBP)
    
    The StandardScaler normalizes these before training.
    At inference time, the same scaler will normalize incoming predictions.
    
    Returns trained model and scaler.
    """
    # Feature matrix: [log(recency+1), frequency, monetary, interaction]
    # Apply log1p to recency to compress long-tail (improves mid-range sensitivity)
    recency_log = np.log1p(rfm['days_since_last_purchase'].values)
    frequency = rfm['total_order_count'].values
    monetary = rfm['avg_order_value'].values
    interaction = recency_log * frequency
    X = np.column_stack([
        recency_log,
        frequency,
        monetary,
        interaction,
    ])
    y = rfm['churn'].values
    
    # Reduce outlier influence and map marginals to uniform distribution
    # QuantileTransformer improves mid-range separation for heavily-skewed features
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n🧠 Training LogisticRegression Model:")
    print(f"   Training samples: {len(X)}")
    print(f"   Feature scaling applied (StandardScaler)")
    
    # Train base LogisticRegression (we will calibrate probabilities)
    # Use stronger regularization to reduce overconfident extremes (smaller C)
    base_clf = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight='balanced',
        C=0.1
    )

    # Calibrate probabilities with sigmoid (Platt scaling) to avoid isotonic overfitting
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, cv=5, method='sigmoid')
    calibrated_clf.fit(X_scaled, y)

    # Evaluate on training set (calibrated estimator delegates predict to base estimator)
    train_score = calibrated_clf.score(X_scaled, y)
    train_preds = calibrated_clf.predict(X_scaled)
    train_precision = (train_preds == y).sum() / len(y)
    print(f"   Training accuracy (calibrated): {train_score:.4f}")
    print(f"   Precision (calibrated): {train_precision:.4f}")

    # Attempt to print base estimator coefficients for interpretability
    feature_names = ['days_since_last_purchase', 'total_order_count', 'avg_order_value']
    try:
        coefs = calibrated_clf.base_estimator.coef_[0]
        intercept = calibrated_clf.base_estimator.intercept_[0]
        print(f"\n   Base Estimator Coefficients (higher = stronger churn predictor):")
        for name, coef in zip(feature_names, coefs):
            print(f"     {name}: {coef:.4f}")
        print(f"     intercept: {intercept:.4f}")
    except Exception:
        print("   Could not read base estimator coefficients (unexpected estimator type)")
    
    # Save model and scaler
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_clf, model_path)
    print(f"\n✅ Calibrated model saved to {model_path}")
    
    # Save scaler (CRITICAL: must use same scaler for inference!)
    scaler_path = model_path.parent / 'churn_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")
    
    print(f"\n💡 Note: Scaler will be automatically loaded at inference time.")
    print(f"   Raw predictions will be normalized using this scaler.")
    
    return calibrated_clf, scaler


def main():
    """Main training pipeline."""
    # Paths
    csv_path = Path(__file__).parent / 'kaggle_raw' / 'churn_data.csv'
    model_output_path = Path(__file__).parent.parent / 'saved_models' / 'churn_model.pkl'
    
    if not csv_path.exists():
        print(f"[ERROR] CSV not found at {csv_path}")
        return
    
    print("=" * 70)
    print("🚀 CHURN MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load and clean data
    df = load_and_prepare_data(str(csv_path))
    
    # Step 2: Engineer RFM features
    rfm = engineer_rfm_features(df)
    
    # Step 3: Label churn
    rfm = label_churn(rfm, churn_threshold_days=45)
    
    # Step 4: Train model
    model, scaler = train_churn_model(rfm, str(model_output_path))
    
    print("\n" + "=" * 70)
    print("✨ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Artifacts saved:")
    print(f"   Model: {model_output_path}")
    print(f"   Scaler: {model_output_path.parent / 'churn_scaler.pkl'}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Start the ML service: python -m uvicorn main:app --reload")
    print(f"   2. Test churn endpoint: curl -X POST http://localhost:8001/ml/churn-predict ...")
    print(f"   3. Service will use your trained model instead of synthetic fallback")


if __name__ == '__main__':
    main()
