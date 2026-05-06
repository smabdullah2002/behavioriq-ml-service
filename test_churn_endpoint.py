#!/usr/bin/env python3
"""Test the churn-predict endpoint with all test cases."""

import json
import requests
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"

test_cases = [
    # Original test cases
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
    },
    # Additional edge cases
    {
        "name": "Brand new customer (day 0)",
        "data": {
            "days_since_last_purchase": 0,
            "total_order_count": 1,
            "avg_order_value": 50
        },
        "expected": "Very low churn probability. Just made first purchase."
    },
    {
        "name": "High frequency, low value customer",
        "data": {
            "days_since_last_purchase": 5,
            "total_order_count": 50,
            "avg_order_value": 10
        },
        "expected": "Low churn probability despite low order value. High frequency is protective."
    },
    {
        "name": "Low frequency, high value customer",
        "data": {
            "days_since_last_purchase": 45,
            "total_order_count": 2,
            "avg_order_value": 500
        },
        "expected": "Medium-high churn probability. High value can't offset low frequency."
    },
    {
        "name": "Dormant whale customer",
        "data": {
            "days_since_last_purchase": 365,
            "total_order_count": 100,
            "avg_order_value": 200
        },
        "expected": "High churn probability. Despite high historical value, long inactivity is risky."
    },
    {
        "name": "Recently churned-back customer",
        "data": {
            "days_since_last_purchase": 3,
            "total_order_count": 1,
            "avg_order_value": 200
        },
        "expected": "Low churn probability. Recent high-value purchase suggests re-engagement."
    },
    # Boundary conditions
    {
        "name": "Zero monetary value (edge case)",
        "data": {
            "days_since_last_purchase": 10,
            "total_order_count": 5,
            "avg_order_value": 0
        },
        "expected": "Should handle gracefully. Zero value orders shouldn't crash the model."
    },
    {
        "name": "Very large order values",
        "data": {
            "days_since_last_purchase": 30,
            "total_order_count": 3,
            "avg_order_value": 10000
        },
        "expected": "Low-medium churn probability. High values reduce risk."
    },
    {
        "name": "Extreme recency (1 day)",
        "data": {
            "days_since_last_purchase": 1,
            "total_order_count": 1,
            "avg_order_value": 25
        },
        "expected": "Very low churn probability. Recent activity is strongest signal."
    },
    {
        "name": "One year of no purchases",
        "data": {
            "days_since_last_purchase": 365,
            "total_order_count": 5,
            "avg_order_value": 75
        },
        "expected": "Very high churn probability. A year of inactivity is critical risk."
    },
    {
        "name": "Steady, consistent customer",
        "data": {
            "days_since_last_purchase": 14,
            "total_order_count": 10,
            "avg_order_value": 100
        },
        "expected": "Low churn probability. Consistent patterns are positive."
    },
]

def test_churn_endpoint():
    """Test all churn prediction cases."""
    print("=" * 80)
    print("🧪 CHURN ENDPOINT TEST CASES")
    print("=" * 80)
    
    results = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"   Input: {test_case['data']}")
        print(f"   Expected: {test_case['expected']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/ml/churn-predict",
                json=test_case["data"],
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure and values
                is_valid = True
                validation_errors = []
                
                # Check required fields
                required_fields = ['churn_probability', 'churn_risk_level', 'rfm_breakdown', 'recommended_action']
                for field in required_fields:
                    if field not in result:
                        is_valid = False
                        validation_errors.append(f"Missing field: {field}")
                
                # Validate churn probability is between 0 and 1
                churn_prob = result.get('churn_probability', -1)
                if not (0 <= churn_prob <= 1):
                    is_valid = False
                    validation_errors.append(f"Churn probability out of range: {churn_prob}")
                
                # Validate risk level
                valid_risk_levels = ['low', 'medium', 'high', 'very_high', 'error']
                risk_level = result.get('churn_risk_level', '')
                if risk_level not in valid_risk_levels:
                    is_valid = False
                    validation_errors.append(f"Invalid risk level: {risk_level}")
                
                # Validate RFM breakdown
                rfm = result.get('rfm_breakdown', {})
                for key, val in rfm.items():
                    if not (0 <= val <= 1):
                        is_valid = False
                        validation_errors.append(f"RFM score out of range: {key}={val}")
                
                if is_valid:
                    print(f"   ✅ Status: 200 OK")
                    passed += 1
                else:
                    print(f"   ⚠️  Status: 200 OK (Validation warnings)")
                    for error in validation_errors:
                        print(f"      - {error}")
                
                print(f"   📊 Result:")
                print(f"      - Churn Probability: {churn_prob:.4f}")
                print(f"      - Risk Level: {risk_level.upper()}")
                print(f"      - Action: {result['recommended_action']}")
                print(f"      - Model Type: {result.get('model_type', 'N/A')}")
                print(f"   RFM Breakdown:")
                for key, val in rfm.items():
                    print(f"      - {key}: {val:.4f}")
                
                results.append({
                    "test_name": test_case['name'],
                    "status": "passed" if is_valid else "warning",
                    "churn_probability": churn_prob,
                    "risk_level": risk_level,
                    "rfm_breakdown": rfm
                })
            else:
                print(f"   ❌ Status: {response.status_code}")
                print(f"   Error: {response.text}")
                failed += 1
                results.append({
                    "test_name": test_case['name'],
                    "status": "failed",
                    "error": response.text
                })
        
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Error: Cannot connect to {BASE_URL}")
            print(f"   Make sure the service is running on port 8001")
            failed += 1
            results.append({
                "test_name": test_case['name'],
                "status": "failed",
                "error": f"Connection refused to {BASE_URL}"
            })
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
            results.append({
                "test_name": test_case['name'],
                "status": "failed",
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Warnings: {len([r for r in results if r['status'] == 'warning'])}")
    print(f"❌ Failed: {failed}")
    
    # Group by risk level
    print("\n📈 RISK LEVEL DISTRIBUTION:")
    risk_levels = {}
    for result in results:
        if result['status'] != 'failed':
            risk = result.get('risk_level', 'unknown')
            if risk not in risk_levels:
                risk_levels[risk] = []
            risk_levels[risk].append(result['test_name'])
    
    for risk, tests in sorted(risk_levels.items()):
        print(f"   {risk.upper()}: {len(tests)} tests")
        for test_name in tests:
            print(f"      - {test_name}")
    
    # Show probability statistics
    print("\n📉 PROBABILITY STATISTICS:")
    probabilities = [r['churn_probability'] for r in results if 'churn_probability' in r]
    if probabilities:
        print(f"   Min: {min(probabilities):.4f}")
        print(f"   Max: {max(probabilities):.4f}")
        print(f"   Mean: {sum(probabilities) / len(probabilities):.4f}")
    
    print("\n✨ Test complete!")
    print("=" * 80)
