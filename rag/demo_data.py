"""Demo data seeder for RAG namespaces."""

from typing import List, Dict, Any
import random
from datetime import datetime, timedelta


def generate_demo_orders(count: int = 20) -> List[Dict[str, Any]]:
    """Generate demo order data."""
    orders = []
    customers = [f"cust_{i}" for i in range(1, 11)]
    statuses = ["completed", "shipped", "pending", "delivered"]

    for i in range(count):
        order_date = datetime.now() - timedelta(days=random.randint(1, 90))
        orders.append(
            {
                "id": f"order_{i+1}",
                "order_id": f"ORD-{1000+i}",
                "customer_id": random.choice(customers),
                "total": round(random.uniform(50, 500), 2),
                "items_count": random.randint(1, 5),
                "date": order_date.strftime("%Y-%m-%d"),
                "status": random.choice(statuses),
            }
        )
    return orders


def generate_demo_churn_predictions(count: int = 15) -> List[Dict[str, Any]]:
    """Generate demo churn prediction data."""
    predictions = []
    actions = ["monitor", "engage", "retention_offer", "win_back"]

    for i in range(count):
        predictions.append(
            {
                "id": f"churn_{i+1}",
                "customer_id": f"cust_{i+1}",
                "churn_probability": round(random.uniform(0.1, 0.95), 3),
                "days_since_purchase": random.randint(5, 120),
                "rfm_score": round(random.uniform(1, 10), 2),
                "predicted_action": random.choice(actions),
            }
        )
    return predictions


def generate_demo_pricing_decisions(count: int = 12) -> List[Dict[str, Any]]:
    """Generate demo pricing decision data."""
    from catalog import load_product_catalog

    try:
        products = load_product_catalog()
    except:
        products = []

    decisions = []
    decision_types = ["seasonal", "inventory_clear", "competitor_match", "loyalty_reward"]

    for i in range(count):
        product = products[i % len(products)] if products else {"id": f"p{i+1}", "name": f"Product {i+1}"}

        decisions.append(
            {
                "id": f"pricing_{i+1}",
                "product_id": product.get("id", f"p{i+1}"),
                "product_name": product.get("name", f"Product {i+1}"),
                "discount_percent": round(random.uniform(5, 40), 1),
                "decision_type": random.choice(decision_types),
                "effective_date": (datetime.now() + timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
                "reason": random.choice(["High inventory", "Competitor promotion", "Seasonal sale", "Customer acquisition"]),
            }
        )
    return decisions


def generate_demo_search_logs(count: int = 25) -> List[Dict[str, Any]]:
    """Generate demo search log data."""
    queries = [
        "running shoes",
        "winter jacket",
        "blue jeans",
        "sports equipment",
        "outdoor gear",
        "casual apparel",
        "accessories",
        "shoes",
        "athletic wear",
        "winter clothing",
    ]

    logs = []
    for i in range(count):
        logs.append(
            {
                "id": f"search_{i+1}",
                "search_query": random.choice(queries),
                "results_count": random.randint(5, 50),
                "conversions": random.randint(0, 5),
                "conversion_rate": round(random.uniform(0.0, 0.2), 3),
                "avg_position": round(random.uniform(1, 10), 1),
                "date": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
            }
        )
    return logs


def get_all_demo_data() -> Dict[str, List[Dict[str, Any]]]:
    """Get all demo data organized by namespace."""
    return {
        "orders": generate_demo_orders(20),
        "churn_predictions": generate_demo_churn_predictions(15),
        "pricing_decisions": generate_demo_pricing_decisions(12),
        "search_logs": generate_demo_search_logs(25),
    }
