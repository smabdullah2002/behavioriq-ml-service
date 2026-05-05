#!/usr/bin/env python3
"""Test the churn-predict endpoint with all test cases."""

import json
import requests
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8001"

test_cases = [
    {
        "name": "Very recent buyer",
        "data": {
            "days_since_last_purchase": 2,
            "total_order_count": 12,
            "avg_order_value": 120
        },
        "expected": "Low churn probability. Frequent and recent buyers should look healthy."
    },
    {
        "name": "Moderate customer",
        "data": {
            "days_since_last_purchase": 20,
            "total_order_count": 5,
            "avg_order_value": 75
        },
        "expected": "Medium churn probability. Balanced profile."
    },
    {
        "name": "At-risk customer",
        "data": {
            "days_since_last_purchase": 60,
            "total_order_count": 2,
            "avg_order_value": 40
        },
        "expected": "Higher churn probability because recency and frequency are weak."
    },
    {
        "name": "Very high risk",
        "data": {
            "days_since_last_purchase": 180,
            "total_order_count": 1,
            "avg_order_value": 20
        },
        "expected": "Churn probability should be high."
    },
    {
        "name": "Strong loyal customer",
        "data": {
            "days_since_last_purchase": 1,
            "total_order_count": 30,
            "avg_order_value": 150
        },
        "expected": "Very low churn probability."
    }
]

def test_churn_endpoint():
    """Test all churn prediction cases."""
    print("=" * 80)
    print("🧪 CHURN ENDPOINT TEST CASES")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"   Expected: {test_case['expected']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/ml/churn-predict",
                json=test_case["data"],
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Status: 200 OK")
                print(f"   📊 Result:")
                print(f"      - Churn Probability: {result['churn_probability']:.4f}")
                print(f"      - Risk Level: {result['churn_risk_level']}")
                print(f"      - Action: {result['recommended_action']}")
                print(f"      - Model Type: {result.get('_model_type', 'N/A')}")
                print(f"   RFM Breakdown:")
                for key, val in result['rfm_breakdown'].items():
                    print(f"      - {key}: {val:.4f}")
            else:
                print(f"   ❌ Status: {response.status_code}")
                print(f"   Error: {response.text}")
        
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Error: Cannot connect to {BASE_URL}")
            print(f"   Make sure the service is running on port 8001")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_churn_endpoint()
    print("\n" + "=" * 80)
    print("✨ Test complete!")
    print("=" * 80)
