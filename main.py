"""FastAPI application for BehaviorIQ ML service."""

import time
import logging

from fastapi import FastAPI, Request
from fastapi.responses import Response
import numpy as np
from pathlib import Path

# ── Centralized logging — must be imported before any other local module ─────
from logger import setup_logging, get_logger
from metrics import (
    BUSINESS_ERRORS,
    MALFORMED_OUTPUTS,
    VECTOR_LOOKUP_FAILURES,
    metrics_payload,
    observe_http_request,
    observe_step,
    refresh_resource_gauges,
)

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

_EP_INTENT = "/ml/intent-score"
_EP_CHURN = "/ml/churn-predict"
_EP_USER_VECTOR = "/ml/user-vector"
_EP_PRODUCT_EMBED = "/ml/product-embed"
_EP_SEARCH_RERANK = "/ml/search-rerank"

# ── Domain imports ────────────────────────────────────────────────────────────
from models.intent import intent_score
from models.churn_model import load_or_train_model, predict_churn
from models.embedder import ProductEmbedder
from models.reranker import rerank_candidates, cosine_similarity
from models.reranker_updated import (
    CloudEmbedder,
    SparseEncoder,
    PineconeManager,
    IntentAnalyzer,
    BehaviorIQReranker,
    PINECONE_API_KEY,
    HF_API_KEY,
    OFFLINE_MODE,
    INDEX_NAME,
    EMBEDDING_DIM,
    TOP_CANDIDATES,
    _load_products,
)
from schemas.requests import (
    IntentRequest,
    IntentResponse,
    ChurnRequest,
    ChurnResponse,
    ProductEmbedRequest,
    ProductEmbedResponse,
    UserVectorRequest,
    UserVectorResponse,
    SearchRerankRequest,
    SearchRerankResponse,
    SearchRerankResult,
    BIQSearchRequest,
    BIQSearchResponse,
    BIQProductResult,
    BIQIntentInfo,
)
from seeds.generator import initialize_seed_data
from models.preprocessor import normalize_intent_features

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(title="BehaviorIQ ML Service", version="0.1.0")

# Global state
embedder = ProductEmbedder()
product_vectors = {}
churn_model = None
churn_scaler = None

# BehaviorIQ full-pipeline singletons
_biq_embedder: CloudEmbedder | None = None
_biq_sparse_enc: SparseEncoder | None = None
_biq_pinecone_mgr: PineconeManager | None = None
_biq_intent = IntentAnalyzer()
_biq_reranker = BehaviorIQReranker()


# ── Middleware: per-request latency logging + Prometheus ────────────────────
@app.middleware("http")
async def log_and_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_s = time.perf_counter() - start
    elapsed_ms = round(elapsed_s * 1000, 2)
    path = request.url.path
    logger.info(
        "HTTP %s %s → %s  (%sms)",
        request.method,
        path,
        response.status_code,
        elapsed_ms,
    )
    refresh_resource_gauges()
    if path != "/metrics":
        observe_http_request(request.method, path, response.status_code, elapsed_s)
    return response


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus scrape endpoint (configure Grafana via Prometheus datasource)."""
    body, ctype = metrics_payload()
    return Response(content=body, media_type=ctype)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    """Initialize seed data and load embedder at startup."""
    global embedder, product_vectors, churn_model, churn_scaler

    logger.info("=== BehaviorIQ ML Service starting up ===")

    product_vectors = initialize_seed_data(embedder)
    logger.info("Products loaded: %d", len(product_vectors))

    churn_model, churn_scaler = load_or_train_model()
    scaler_status = "with scaler" if churn_scaler else "no scaler (fallback)"
    logger.info("Churn model ready (%s)", scaler_status)

    # ── BehaviorIQ full-pipeline init ─────────────────────────────────────────
    global _biq_embedder, _biq_sparse_enc, _biq_pinecone_mgr

    biq_products = _load_products()
    biq_corpus = [
        f"{p['name']} {p.get('desc', '')} {p.get('category', '')} {p.get('brand', '')}"
        for p in biq_products
    ]

    _biq_sparse_enc = SparseEncoder()
    _biq_sparse_enc.fit(biq_corpus)

    _biq_embedder = CloudEmbedder(api_key=HF_API_KEY, offline=OFFLINE_MODE)
    _biq_embedder.fit_fallback(biq_corpus)

    _biq_pinecone_mgr = PineconeManager(
        api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        dim=EMBEDDING_DIM,
        offline=OFFLINE_MODE,
    )

    dense_vecs = _biq_embedder.embed(biq_corpus)
    sparse_vecs = [_biq_sparse_enc.encode(t) for t in biq_corpus]
    _biq_pinecone_mgr.upsert_products(biq_products, dense_vecs, sparse_vecs)

    biq_backend = (
        "pinecone_hybrid" if not _biq_pinecone_mgr.offline else "offline_tfidf"
    )
    embedder_type = "HF" if not _biq_embedder.offline else "TF-IDF"

    logger.info(
        "ML Service ready | products=%d | embedder=%s | biq_backend=%s | endpoints=intent-score,churn-predict,user-vector,search-rerank,search",
        len(product_vectors),
        embedder_type,
        biq_backend,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/ml/intent-score", response_model=IntentResponse)
def ml_intent(req: IntentRequest):
    """Compute user intent score."""
    raw = req.dict()
    with observe_step(_EP_INTENT, "normalize_features"):
        features = normalize_intent_features(raw)
    with observe_step(_EP_INTENT, "intent_score"):
        res = intent_score(features)

    if isinstance(res, dict):
        score = res.get("intent_score", 0.0)
        bucket = res.get("score_bucket", "churn_risk")
        dominant = res.get("dominant_signal", "none")

        logger.info(
            "intent-score | score=%.2f | bucket=%s | dominant_signal=%s",
            score,
            bucket,
            dominant,
        )

        return IntentResponse(
            intent_score=score,
            score_bucket=bucket,
            dominant_signal=dominant,
            contributions=res.get("contributions"),
        )

    logger.warning("intent-score fallback triggered — res was not a dict")
    MALFORMED_OUTPUTS.labels(reason="intent_score_shape").inc()
    return IntentResponse(
        intent_score=0.0, score_bucket="churn_risk", dominant_signal="none"
    )


@app.post("/ml/churn-predict", response_model=ChurnResponse)
def ml_churn(req: ChurnRequest):
    """Predict churn probability."""
    try:
        with observe_step(_EP_CHURN, "predict_churn"):
            result = predict_churn(
                churn_model,
                req.days_since_last_purchase,
                req.total_order_count,
                req.avg_order_value,
                scaler=churn_scaler,
            )

        logger.info(
            "churn-predict | days=%s | orders=%s | avg_value=%s | churn_prob=%.4f | risk=%s | model=%s",
            req.days_since_last_purchase,
            req.total_order_count,
            req.avg_order_value,
            result.get("churn_probability", 0.0),
            result.get("churn_risk_level", "unknown"),
            result.get("model_type", "unknown"),
        )

        if result.get("churn_risk_level") == "error":
            BUSINESS_ERRORS.labels(endpoint=_EP_CHURN).inc()

        return ChurnResponse(**result)

    except Exception:
        logger.exception(
            "churn-predict FAILED | days=%s | orders=%s | avg_value=%s",
            req.days_since_last_purchase,
            req.total_order_count,
            req.avg_order_value,
        )
        BUSINESS_ERRORS.labels(endpoint=_EP_CHURN).inc()
        return ChurnResponse(
            churn_probability=0.0,
            churn_risk_level="error",
            rfm_breakdown={
                "recency_score": 0.0,
                "frequency_score": 0.0,
                "monetary_score": 0.0,
            },
            recommended_action="none",
            model_type=None,
        )


@app.post("/ml/product-embed", response_model=ProductEmbedResponse)
def ml_product_embed(req: ProductEmbedRequest):
    """Embed a single product and register it for user-vector / rerank lookup."""
    from fastapi import HTTPException

    if embedder.vectorizer is None:
        raise HTTPException(503, "Embedder not initialised yet")

    product = {
        "product_id": req.product_id,
        "name": req.name,
        "description": req.description or "",
        "category": req.category or "",
        "brand": req.brand or "",
    }
    text = embedder.product_text(product)

    with observe_step(_EP_PRODUCT_EMBED, "embed_product"):
        vec = embedder.embed_product(product)

    global product_vectors
    product_vectors = embedder.product_vectors

    # Keep BIQ hybrid index in sync when the full pipeline is loaded
    if _biq_embedder is not None and _biq_sparse_enc is not None and _biq_pinecone_mgr is not None:
        with observe_step(_EP_PRODUCT_EMBED, "biq_index_upsert"):
            dense = _biq_embedder.embed_one(text)
            sparse = _biq_sparse_enc.encode(text)
            biq_row = {
                "id": req.product_id,
                "name": req.name,
                "desc": req.description or "",
                "category": req.category or "",
                "brand": req.brand or "",
                "price": 0.0,
                "rating": 3.5,
                "popularity": 0.5,
                "stock": True,
                "discount": 0.0,
            }
            _biq_pinecone_mgr.upsert_products([biq_row], [dense], [sparse])

    logger.info(
        "product-embed | id=%s | dim=%d",
        req.product_id,
        len(vec),
    )

    return ProductEmbedResponse(
        product_id=req.product_id,
        product_vector=vec.tolist(),
        vector_dim=len(vec),
    )


@app.post("/ml/user-vector", response_model=UserVectorResponse)
def ml_user_vector(req: UserVectorRequest):
    """Build user behavioral vector."""
    missing = sum(
        1 for pid in req.recent_product_ids if pid not in embedder.product_vectors
    )
    if missing:
        VECTOR_LOOKUP_FAILURES.labels(endpoint=_EP_USER_VECTOR).inc(missing)

    with observe_step(_EP_USER_VECTOR, "build_user_vector"):
        user_vector = embedder.build_user_vector(
            req.recent_product_ids,
            req.weights,
        )

    logger.info(
        "user-vector | product_count=%d | vector_dim=%d",
        len(req.recent_product_ids),
        len(user_vector),
    )

    return UserVectorResponse(user_vector=user_vector.tolist())


@app.post("/ml/search-rerank", response_model=SearchRerankResponse)
def ml_search_rerank(req: SearchRerankRequest):
    """Re-rank search candidates using user vector."""
    with observe_step(_EP_SEARCH_RERANK, "prepare_arrays"):
        user_vector = np.array(req.user_vector)
        candidates = [c.dict() for c in req.candidates]
        weights = req.weights or {"vector": 0.45, "intent": 0.30, "pricing": 0.25}
        search_intent = req.search_intent.dict() if req.search_intent else None

    with observe_step(_EP_SEARCH_RERANK, "rerank_candidates"):
        results = rerank_candidates(
            user_vector, candidates, product_vectors, weights, search_intent
        )

    top_score = results[0]["final_score"] if results else 0.0
    logger.info(
        "search-rerank | candidates=%d | top_score=%.4f | weights=%s",
        len(candidates),
        top_score,
        weights,
    )

    return SearchRerankResponse(results=[SearchRerankResult(**r) for r in results])


_EP_SEARCH = "/ml/search"


@app.post("/ml/search", response_model=BIQSearchResponse)
def ml_search(req: BIQSearchRequest):
    """Full BehaviorIQ pipeline: semantic+keyword search → intent → rerank."""
    from fastapi import HTTPException

    if _biq_embedder is None or _biq_sparse_enc is None or _biq_pinecone_mgr is None:
        raise HTTPException(503, "BIQ pipeline not initialised yet")

    t0 = time.perf_counter()

    with observe_step(_EP_SEARCH, "embed_query"):
        query_dense = _biq_embedder.embed_one(req.query)
        query_sparse = _biq_sparse_enc.encode(req.query)

    with observe_step(_EP_SEARCH, "intent_analysis"):
        intent = _biq_intent.analyze(req.query)

    with observe_step(_EP_SEARCH, "retrieve_candidates"):
        candidates = _biq_pinecone_mgr.query(
            query_dense, query_sparse, top_k=TOP_CANDIDATES
        )

    # Narrow by category slot
    cat_slot = intent.get("slots", {}).get("category")
    if cat_slot:
        filtered = [c for c in candidates if c["metadata"].get("category") == cat_slot]
        if filtered:
            candidates = filtered

    # Narrow by max_price slot
    max_price_slot = intent.get("slots", {}).get("max_price")
    if max_price_slot:
        try:
            ceiling = float(max_price_slot)
            filtered = [
                c for c in candidates if c["metadata"].get("price", 9999) <= ceiling
            ]
            if filtered:
                candidates = filtered
        except ValueError:
            pass

    with observe_step(_EP_SEARCH, "rerank"):
        ranked = _biq_reranker.rerank(
            candidates,
            intent,
            churn_score=req.churn_score,
            weights=req.weights,
        )
    ranked = ranked[: req.top_k]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    backend = (
        "pinecone_hybrid" if not _biq_pinecone_mgr.offline else "offline_tfidf_fallback"
    )

    logger.info(
        "search | query=%r | intent=%s | candidates=%d | results=%d | backend=%s | %.1fms",
        req.query,
        intent.get("label"),
        len(candidates),
        len(ranked),
        backend,
        elapsed_ms,
    )

    return BIQSearchResponse(
        query=req.query,
        intent=BIQIntentInfo(**intent),
        results=[BIQProductResult(rank=i + 1, **r) for i, r in enumerate(ranked)],
        search_backend=backend,
        total_candidates=len(candidates),
        pipeline_ms=elapsed_ms,
    )


@app.get("/ml/search/index-status")
def ml_search_index_status():
    """Status of the Pinecone hybrid index."""
    if _biq_pinecone_mgr and not _biq_pinecone_mgr.offline and _biq_pinecone_mgr._index:
        try:
            stats = _biq_pinecone_mgr._index.describe_index_stats()
            return {"status": "connected", "index": INDEX_NAME, "stats": stats}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
    return {
        "status": "offline",
        "index": INDEX_NAME,
        "indexed_products": len(_biq_pinecone_mgr._store) if _biq_pinecone_mgr else 0,
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    products_file = Path("data/products.json")
    embedder_file = Path("saved_models/embedder.pkl")

    status = {
        "status": "ok",
        "service": "BehaviorIQ ML Service",
        "products_loaded": len(product_vectors),
        "products_file_exists": products_file.exists(),
        "embedder_cached": embedder_file.exists(),
    }

    logger.debug("health-check | products_loaded=%d", len(product_vectors))

    return status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
