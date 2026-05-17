"""Pydantic request and response schemas."""

from pydantic import BaseModel
from typing import List, Dict, Optional


class IntentRequest(BaseModel):
    """Request for intent score endpoint."""
    product_visit_count: float = 0
    time_on_product_page: float = 0
    cart_add_events: float = 0
    scroll_depth: float = 0
    avg_spend_score: float = 0
    session_recency: float = 0


class IntentResponse(BaseModel):
    """Response for intent score endpoint."""
    intent_score: float
    score_bucket: str
    dominant_signal: str
    contributions: Optional[Dict[str, float]] = None


class ChurnRequest(BaseModel):
    """Request for churn prediction endpoint."""
    days_since_last_purchase: int
    total_order_count: int
    avg_order_value: float


class ChurnResponse(BaseModel):
    """Response for churn prediction endpoint."""
    churn_probability: float
    churn_risk_level: str
    rfm_breakdown: Dict[str, float]
    recommended_action: str
    model_type: Optional[str] = None  # DEBUG: indicates real_trained or fallback_synthetic


class ProductEmbedRequest(BaseModel):
    """Request to embed (or re-embed) a single catalog product."""
    product_id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None


class ProductEmbedResponse(BaseModel):
    """TF-IDF embedding for API persistence and user-vector lookup."""
    product_id: str
    product_vector: List[float]
    vector_dim: int


class UserVectorRequest(BaseModel):
    """Request for user vector builder endpoint."""
    recent_product_ids: List[str]
    weights: Optional[List[float]] = None


class UserVectorResponse(BaseModel):
    """Response for user vector builder endpoint."""
    user_vector: List[float]


class CandidateItem(BaseModel):
    """A search candidate product."""
    product_id: str
    keyword_score: float
    popularity_score: Optional[float] = 0.5
    semantic_score: Optional[float] = None
    price: Optional[float] = None
    category: Optional[str] = None


class SearchIntent(BaseModel):
    """Optional search intent provided by intent service."""
    label: str
    confidence: float
    slots: Optional[Dict[str, str]] = None


class SearchRerankRequest(BaseModel):
    """Request for search re-ranking endpoint."""
    user_vector: List[float]
    candidates: List[CandidateItem]
    weights: Optional[Dict[str, float]] = None
    search_intent: Optional[SearchIntent] = None


class SearchRerankResult(BaseModel):
    """Result item in re-ranked search results."""
    product_id: str
    final_score: float
    vector_score: Optional[float] = None
    cosine_score: Optional[float] = None
    keyword_score: Optional[float] = None
    intent_score: Optional[float] = None
    pricing_score: Optional[float] = None
    intent_label: Optional[str] = None
    intent_confidence: Optional[float] = None


class SearchRerankResponse(BaseModel):
    """Response for search re-ranking endpoint."""
    results: List[SearchRerankResult]


# ── BehaviorIQ full-pipeline search (/ml/search) ─────────────────────────────

class BIQSearchRequest(BaseModel):
    """Request for the full BehaviorIQ search pipeline."""
    query: str
    churn_score: float = 0.5
    recent_product_ids: Optional[List[str]] = None
    top_k: int = 20
    weights: Optional[Dict[str, float]] = None


class BIQIntentInfo(BaseModel):
    label: str
    confidence: float
    slots: Dict


class BIQProductResult(BaseModel):
    rank: int
    product_id: str
    name: str
    category: str
    brand: str
    price: float
    discount: float
    rating: float
    final_score: float
    vector_score: float
    intent_score: float
    pricing_score: float
    intent_label: str
    intent_confidence: float
    rank_explanation: str


class BIQSearchResponse(BaseModel):
    query: str
    intent: BIQIntentInfo
    results: List[BIQProductResult]
    search_backend: str
    total_candidates: int
    pipeline_ms: float


