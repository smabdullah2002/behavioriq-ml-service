"""FastAPI application for BehaviorIQ ML service."""

from fastapi import FastAPI
import numpy as np
from pathlib import Path

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

# Initialize FastAPI app
app = FastAPI(title="BehaviorIQ ML Service", version="0.1.0")

# Global state
embedder = ProductEmbedder()
product_vectors = {}
churn_model = None
churn_scaler = None


@app.on_event("startup")
def startup_event():
    """Initialize seed data and load embedder at startup."""
    global embedder, product_vectors, churn_model, churn_scaler
    product_vectors = initialize_seed_data(embedder)
    churn_model, churn_scaler = load_or_train_model()


@app.post("/ml/intent-score", response_model=IntentResponse)
def ml_intent(req: IntentRequest):
    """Compute user intent score."""
    raw = req.dict()
    features = normalize_intent_features(raw)
    res = intent_score(features)
    # intent_score now returns a dict with score, bucket, signal, and contributions
    if isinstance(res, dict):
        return IntentResponse(
            intent_score=res.get("intent_score", 0.0),
            score_bucket=res.get("score_bucket", "churn_risk"),
            dominant_signal=res.get("dominant_signal", "none"),
            contributions=res.get("contributions")
        )
    # fallback
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
            scaler=churn_scaler
        )
        return ChurnResponse(**result)
    except Exception as e:
        import traceback, logging
        logging.getLogger(__name__).exception("Churn prediction failed")
        # Return a safe error response for debugging
        return ChurnResponse(
            churn_probability=0.0,
            churn_risk_level="error",
            rfm_breakdown={"recency_score": 0.0, "frequency_score": 0.0, "monetary_score": 0.0},
            recommended_action="none",
            model_type=None
        )


@app.post("/ml/churn-predict-formula", response_model=FormulaChurnResponse)
def ml_churn_formula(req: ChurnRequest):
    """Predict churn probability using the formula-based RFM helper."""
    probability = formula_churn_probability(
        req.days_since_last_purchase,
        req.total_order_count,
        req.avg_order_value,
    )
    return FormulaChurnResponse(churn_probability=probability)


@app.post("/ml/user-vector", response_model=UserVectorResponse)
def ml_user_vector(req: UserVectorRequest):
    """Build user behavioral vector."""
    user_vector = embedder.build_user_vector(
        req.recent_product_ids,
        req.weights
    )
    return UserVectorResponse(user_vector=user_vector.tolist())


@app.post("/ml/search-rerank", response_model=SearchRerankResponse)
def ml_search_rerank(req: SearchRerankRequest):
    """Re-rank search candidates using user vector."""
    user_vector = np.array(req.user_vector)
    candidates = [c.dict() for c in req.candidates]
    weights = req.weights or {"kw": 0.5, "cosine": 0.3, "popularity": 0.2}
    
    results = rerank_candidates(user_vector, candidates, product_vectors, weights)
    
    return SearchRerankResponse(
        results=[SearchRerankResult(**r) for r in results]
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    products_file = Path("data/products.json")
    embedder_file = Path("saved_models/embedder.pkl")
    
    return {
        "status": "ok",
        "service": "BehaviorIQ ML Service",
        "products_loaded": len(product_vectors),
        "products_file_exists": products_file.exists(),
        "embedder_cached": embedder_file.exists()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
