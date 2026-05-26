"""Seed data loader for BehaviorIQ ML service."""

import joblib
from pathlib import Path

from catalog import load_product_catalog


def initialize_seed_data(embedder, products=None):
    """Load the shared catalog and fit or restore the TF-IDF embedder.

    Args:
        embedder: ProductEmbedder instance
        products: Optional pre-loaded catalog (defaults to shared products.json)

    Returns:
        dict: product_vectors from embedder
    """
    print("\n" + "=" * 60)
    print("BehaviorIQ ML Service Startup")
    print("=" * 60)

    saved_models_dir = Path("saved_models")
    saved_models_dir.mkdir(exist_ok=True)

    if products is None:
        products = load_product_catalog()

    print(f"\nLoaded shared catalog: {len(products)} products")

    embedder_path = saved_models_dir / "embedder.pkl"
    should_fit = True

    if embedder_path.exists():
        print("\nLoading saved embedder from saved_models/embedder.pkl...")
        try:
            loaded = joblib.load(embedder_path)
            if len(loaded.product_vectors) == len(products):
                embedder.vectorizer = loaded.vectorizer
                embedder.product_vectors = loaded.product_vectors
                embedder.corpus = loaded.corpus
                should_fit = False
                print(
                    f"   Embedder loaded with {len(embedder.product_vectors)} products"
                )
            else:
                print(
                    "   Catalog size changed — refitting embedder from shared catalog..."
                )
        except Exception as e:
            print(f"   Error loading embedder: {e}")
            print("   Fitting new embedder from shared catalog...")

    if should_fit:
        print("\nFitting TF-IDF embedder from shared catalog...")
        embedder.fit(products)
        try:
            joblib.dump(embedder, embedder_path)
            print("   Embedder fitted and saved to saved_models/embedder.pkl")
        except Exception as e:
            print(f"   Could not save embedder: {e}")
            print("   Embedder in memory (will regenerate on next startup)")

    product_vectors = embedder.product_vectors

    print("\n" + "=" * 60)
    print("ML Service Ready!")
    print(f"   • Products: {len(products)}")
    print(f"   • Embedder: TF-IDF (100 features)")
    print(
        "   • Endpoints: intent-score, churn-predict, user-vector, "
        "product-embed, search-rerank"
    )
    print("=" * 60 + "\n")

    return product_vectors
