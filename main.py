"""FastAPI application for BehaviorIQ ML service."""

import time
import logging

from fastapi import FastAPI, Request
import numpy as np
from pathlib import Path

# ── Centralized logging — must be imported before any other local module ─────
from logger import setup_logging, get_logger

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

# ── Domain imports ────────────────────────────────────────────────────────────
from models.intent import intent_score
from models.churn import churn_probability as formula_churn_probability
from models.churn_model import load_or_train_model, predict_churn
from models.embedder import ProductEmbedder
from models.reranker import rerank_candidates, cosine_similarity
from schemas.requests import (
    IntentRequest, IntentResponse,
    ChurnRequest, ChurnResponse, FormulaChurnResponse,
    UserVectorRequest, UserVectorResponse,
    SearchRerankRequest, SearchRerankResponse, SearchRerankResult
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


# ── Middleware: per-request latency logging ───────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "HTTP %s %s → %s  (%sms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


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

    logger.info(
        "ML Service ready | products=%d | embedder=TF-IDF | endpoints=intent-score,churn-predict,user-vector,search-rerank",
        len(product_vectors),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/ml/intent-score", response_model=IntentResponse)
def ml_intent(req: IntentRequest):
    """Compute user intent score."""
    raw = req.dict()
    features = normalize_intent_features(raw)
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
    return IntentResponse(intent_score=0.0, score_bucket="churn_risk", dominant_signal="none")


@app.post("/ml/churn-predict", response_model=ChurnResponse)
def ml_churn(req: ChurnRequest):
    """Predict churn probability."""
    try:
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

        return ChurnResponse(**result)

    except Exception:
        logger.exception(
            "churn-predict FAILED | days=%s | orders=%s | avg_value=%s",
            req.days_since_last_purchase,
            req.total_order_count,
            req.avg_order_value,
        )
        return ChurnResponse(
            churn_probability=0.0,
            churn_risk_level="error",
            rfm_breakdown={"recency_score": 0.0, "frequency_score": 0.0, "monetary_score": 0.0},
            recommended_action="none",
            model_type=None,
        )


@app.post("/ml/churn-predict-formula", response_model=FormulaChurnResponse)
def ml_churn_formula(req: ChurnRequest):
    """Predict churn probability using the formula-based RFM helper."""
    probability = formula_churn_probability(
        req.days_since_last_purchase,
        req.total_order_count,
        req.avg_order_value,
    )

    logger.info(
        "churn-predict-formula | days=%s | orders=%s | avg_value=%s | churn_prob=%.4f",
        req.days_since_last_purchase,
        req.total_order_count,
        req.avg_order_value,
        probability,
    )

    return FormulaChurnResponse(churn_probability=probability)


@app.post("/ml/user-vector", response_model=UserVectorResponse)
def ml_user_vector(req: UserVectorRequest):
    """Build user behavioral vector."""
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
    user_vector = np.array(req.user_vector)
    candidates = [c.dict() for c in req.candidates]
    weights = req.weights or {"kw": 0.5, "cosine": 0.3, "popularity": 0.2}

    results = rerank_candidates(user_vector, candidates, product_vectors, weights)

    top_score = results[0]["final_score"] if results else 0.0
    logger.info(
        "search-rerank | candidates=%d | top_score=%.4f | weights=%s",
        len(candidates),
        top_score,
        weights,
    )

    return SearchRerankResponse(
        results=[SearchRerankResult(**r) for r in results]
    )


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
