"""Shared product catalog loader — single source of truth for the ML service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CatalogProduct = dict[str, Any]

_SERVICE_ROOT = Path(__file__).resolve().parent


def resolve_catalog_path() -> Path:
    env_path = os.environ.get("PRODUCT_CATALOG_PATH", "").strip()
    if env_path:
        return Path(env_path)

    service_catalog = _SERVICE_ROOT / "data" / "products.json"
    if service_catalog.exists():
        return service_catalog

    repo_catalog = _SERVICE_ROOT.parent / "data" / "products.json"
    if repo_catalog.exists():
        return repo_catalog

    return service_catalog


def load_product_catalog() -> list[CatalogProduct]:
    path = resolve_catalog_path()
    if not path.exists():
        raise FileNotFoundError(
            "Product catalog not found. Expected a single shared file at "
            f"{path}. Set PRODUCT_CATALOG_PATH or create data/products.json "
            "at the repository root."
        )

    products = json.loads(path.read_text(encoding="utf-8"))
    for product in products:
        product.setdefault("brand", "Generic")
        product.setdefault("rating", 3.5)
        product.setdefault("popularity", 0.5)
        product.setdefault("stock", True)
        product.setdefault("discount", 0.0)

    return products
