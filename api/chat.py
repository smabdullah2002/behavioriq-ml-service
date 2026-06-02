"""Chat endpoint for the AI Commerce Assistant."""

from fastapi import APIRouter, HTTPException
from schemas.requests import ChatRequest, ChatResponse
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ml", tags=["chat"])

# Global assistant instance (initialized at startup)
_assistant = None


def set_assistant(assistant):
    """Set the global assistant instance."""
    global _assistant
    _assistant = assistant


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat endpoint for the AI Commerce Assistant."""
    if _assistant is None:
        raise HTTPException(
            status_code=503, detail="Assistant not initialized yet"
        )

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Chat request | question={request.question[:50]}...")

    result = _assistant.answer(request.question)

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources"),
        pipeline_ms=result.get("pipeline_ms"),
    )
