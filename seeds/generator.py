"""Seed data generator for BehaviorIQ ML service."""

import json
import joblib
from pathlib import Path
from data.synthetic_churn import generate_synthetic_products


def initialize_seed_data(embedder, products_count=100):
    """Generate and load seed data at service startup.
    
    Args:
        embedder: ProductEmbedder instance
        products_count: Number of synthetic products to generate
        
    Returns:
        dict: product_vectors from embedder
    """
    print("\n" + "="*60)
    print("🚀 BehaviorIQ ML Service Startup")
    print("="*60)
    
    # Ensure saved_models directory exists
    saved_models_dir = Path("saved_models")
    saved_models_dir.mkdir(exist_ok=True)
    
    # Step 1: Check/create products seed data
    products_file = Path("data/products.json")
    if not products_file.exists():
        print("\n📦 Generating seed products...")
        products = generate_synthetic_products(products_count)
        products_file.parent.mkdir(exist_ok=True)
        with open(products_file, 'w') as f:
            json.dump(products, f, indent=2)
        print(f"   ✓ Generated {len(products)} synthetic products → data/products.json")
    else:
        print(f"\n✓ Found existing products file: data/products.json")
        with open(products_file, 'r') as f:
            products = json.load(f)
        print(f"   ✓ Loaded {len(products)} products")
    
    # Step 2: Try to load saved embedder, otherwise fit new one
    embedder_path = saved_models_dir / "embedder.pkl"
    if embedder_path.exists():
        print(f"\n🔄 Loading saved embedder from saved_models/embedder.pkl...")
        try:
            embedder = joblib.load(embedder_path)
            print(f"   ✓ Embedder loaded with {len(embedder.product_vectors)} products")
        except Exception as e:
            print(f"   ⚠ Error loading embedder: {e}")
            print(f"   🔧 Fitting new embedder from products...")
            embedder.fit(products)
            joblib.dump(embedder, embedder_path)
            print(f"   ✓ Embedder fitted and saved")
    else:
        print(f"\n🔧 No saved embedder found. Fitting TF-IDF embedder...")
        embedder.fit(products)
        try:
            joblib.dump(embedder, embedder_path)
            print(f"   ✓ Embedder fitted and saved to saved_models/embedder.pkl")
        except Exception as e:
            print(f"   ⚠ Could not save embedder: {e}")
            print(f"   ✓ Embedder in memory (will regenerate on next startup)")
    
    # Step 3: Update and return product vectors
    product_vectors = embedder.product_vectors
    
    print("\n" + "="*60)
    print("✅ ML Service Ready!")
    print(f"   • Products: {len(products)}")
    print(f"   • Embedder: TF-IDF (100 features)")
    print(f"   • Endpoints: intent-score, churn-predict, user-vector, search-rerank")
    print("="*60 + "\n")
    
    return product_vectors
