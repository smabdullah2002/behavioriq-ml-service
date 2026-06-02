"""Data ingestion endpoints for RAG namespaces."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from logger import get_logger
from rag.document_builder import DocumentBuilder
from rag.demo_data import get_all_demo_data

logger = get_logger(__name__)

router = APIRouter(prefix="/ml", tags=["ingest"])

# Global references (set from main.py)
_embedder = None
_sparse_encoder = None
_vector_store = None


def set_ingest_dependencies(embedder, sparse_encoder, vector_store):
    """Set the global dependencies for ingest operations."""
    global _embedder, _sparse_encoder, _vector_store
    _embedder = embedder
    _sparse_encoder = sparse_encoder
    _vector_store = vector_store


class IngestRequest(BaseModel):
    """Request to ingest data into a namespace."""
    namespace: str
    data: List[Dict[str, Any]]


class IngestResponse(BaseModel):
    """Response from data ingestion."""
    success: bool
    namespace: str
    documents_ingested: int
    message: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest) -> IngestResponse:
    """
    Ingest data into a specific namespace.

    Supported namespaces:
    - products
    - orders
    - churn_predictions
    - pricing_decisions
    - search_logs
    """
    if _vector_store is None or _embedder is None or _sparse_encoder is None:
        raise HTTPException(
            status_code=503, detail="Vector store not initialized"
        )

    if not request.namespace:
        raise HTTPException(status_code=400, detail="Namespace is required")

    if not request.data:
        raise HTTPException(status_code=400, detail="Data cannot be empty")

    try:
        # Build documents from raw data
        documents = DocumentBuilder.build_documents(request.data, request.namespace)

        if not documents:
            raise ValueError(f"No documents built from data for namespace: {request.namespace}")

        # Generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings_dense = _embedder.embed(texts)

        # Generate sparse embeddings with fallback
        embeddings_sparse = []
        for text in texts:
            sparse = _sparse_encoder.encode(text)
            # Ensure sparse vector has at least one value (Pinecone requirement)
            indices = sparse.get('indices', [])
            values = sparse.get('values', [])
            if not indices or len(indices) == 0:
                # Fallback: create a minimal valid sparse vector
                sparse = {"indices": [0], "values": [1.0]}
            embeddings_sparse.append(sparse)

        # Upsert to Pinecone
        _vector_store.upsert_documents(
            documents, embeddings_dense, embeddings_sparse, namespace=request.namespace
        )

        logger.info(
            f"Ingested {len(documents)} documents to namespace: {request.namespace}"
        )

        return IngestResponse(
            success=True,
            namespace=request.namespace,
            documents_ingested=len(documents),
            message=f"Successfully ingested {len(documents)} documents",
        )

    except Exception as e:
        logger.error(f"Error ingesting data to {request.namespace}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/demo", response_model=Dict[str, Any])
async def ingest_demo_data() -> Dict[str, Any]:
    """
    Ingest pre-generated demo data into all namespaces.
    Useful for testing and development.
    """
    if _vector_store is None or _embedder is None or _sparse_encoder is None:
        raise HTTPException(
            status_code=503, detail="Vector store not initialized"
        )

    demo_data = get_all_demo_data()
    results = {}

    for namespace, data in demo_data.items():
        try:
            # Build documents from demo data
            documents = DocumentBuilder.build_documents(data, namespace)

            if documents:
                # Generate embeddings
                texts = [doc["text"] for doc in documents]
                embeddings_dense = _embedder.embed(texts)

                # Generate sparse embeddings with fallback
                embeddings_sparse = []
                for text in texts:
                    sparse = _sparse_encoder.encode(text)
                    # Ensure sparse vector has at least one value (Pinecone requirement)
                    indices = sparse.get('indices', [])
                    values = sparse.get('values', [])
                    if not indices or len(indices) == 0:
                        # Fallback: create a minimal valid sparse vector
                        sparse = {"indices": [0], "values": [1.0]}
                    embeddings_sparse.append(sparse)

                # Upsert to Pinecone
                _vector_store.upsert_documents(
                    documents, embeddings_dense, embeddings_sparse, namespace=namespace
                )

                results[namespace] = {
                    "success": True,
                    "documents_ingested": len(documents),
                    "message": f"Ingested {len(documents)} demo documents",
                }
                logger.info(f"Ingested {len(documents)} demo documents to {namespace}")

            else:
                results[namespace] = {
                    "success": False,
                    "documents_ingested": 0,
                    "message": "No documents built",
                }

        except Exception as e:
            results[namespace] = {
                "success": False,
                "documents_ingested": 0,
                "message": f"Error: {str(e)}",
            }
            logger.error(f"Error ingesting demo data to {namespace}: {str(e)}")

    return {
        "message": "Demo data ingestion completed",
        "results": results,
    }
