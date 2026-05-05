"""Synthetic data generator for churn model training."""

import json
import random
from typing import List, Dict


def generate_synthetic_products(count: int = 100) -> List[Dict]:
    """
    Generate synthetic product catalog.
    
    Args:
        count: number of products to generate
    
    Returns:
        list of product dicts
    """
    categories = ["shoes", "apparel", "accessories", "sports", "outdoor"]
    products = []
    
    for i in range(count):
        category = random.choice(categories)
        products.append({
            "id": f"p{i+1}",
            "name": f"Product {i+1}",
            "desc": f"{category.capitalize()} item {i+1}",
            "category": category,
            "price": round(random.uniform(20, 500), 2),
        })
    
    return products


def generate_synthetic_users(count: int = 50) -> List[Dict]:
    """
    Generate synthetic user/order history data.
    
    Args:
        count: number of users to generate
    
    Returns:
        list of user dicts with RFM features
    """
    users = []
    
    for i in range(count):
        users.append({
            "user_id": f"u{i+1}",
            "days_since_last_purchase": random.randint(1, 365),
            "total_order_count": random.randint(1, 50),
            "avg_order_value": round(random.uniform(20, 500), 2),
        })
    
    return users


def save_products_json(products: List[Dict], filepath: str) -> None:
    """Save products to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(products, f, indent=2)


def load_products_json(filepath: str) -> List[Dict]:
    """Load products from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
