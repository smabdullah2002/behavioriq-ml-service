"""Pydantic request and response schemas."""

from pydantic import BaseModel
from typing import List, Dict, Optional


class IntentRequest(BaseModel):
    """Request for intent score endpoint."""
    product_visit_count: float = 0
    time_on_product_page: float = 0
    cart_add_events: float = 0
    search_to_view_ratio: float = 0
    price_range_affinity: float = 0
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


class SearchRerankRequest(BaseModel):
    """Request for search re-ranking endpoint."""
    user_vector: List[float]
    candidates: List[CandidateItem]
    weights: Optional[Dict[str, float]] = None


class SearchRerankResult(BaseModel):
    """Result item in re-ranked search results."""
    product_id: str
    final_score: float


class SearchRerankResponse(BaseModel):
    """Response for search re-ranking endpoint."""
    results: List[SearchRerankResult]
