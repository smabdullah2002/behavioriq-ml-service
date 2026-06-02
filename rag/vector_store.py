"""Vector store wrapper for Pinecone hybrid search with multiple namespaces."""

from typing import List, Dict, Any
from logger import get_logger

logger = get_logger(__name__)

# Available data source namespaces
NAMESPACES = {
    "products": "products",
    "orders": "orders",
    "churn_predictions": "churn_predictions",
    "pricing_decisions": "pricing_decisions",
    "search_logs": "search_logs",
}


class PineconeVectorStore:
    """Wrapper around Pinecone for RAG retrieval with namespace support."""

    def __init__(self, pinecone_manager, embedder, sparse_encoder):
        """
        Initialize the vector store with Pinecone manager and embedders.

        Args:
            pinecone_manager: PineconeManager instance
            embedder: CloudEmbedder for dense vectors
            sparse_encoder: SparseEncoder for sparse vectors
        """
        self.pinecone_manager = pinecone_manager
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder

    def retrieve(
        self, query: str, top_k: int = 3, namespaces: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k results PER namespace so every data source contributes context.

        Args:
            query: Natural language query
            top_k: Number of results per namespace (not total)
            namespaces: List of namespaces to query (default: all)

        Returns:
            List of documents with metadata and source
        """
        if namespaces is None:
            namespaces = list(NAMESPACES.values())

        try:
            # Generate embeddings for the query
            query_dense = self.embedder.embed_one(query)
            query_sparse = self.sparse_encoder.encode(query)

            # Sparse encoder is fitted on product corpus - may not know query words
            if not query_sparse.get("indices"):
                query_sparse = {"indices": [0], "values": [1.0]}

            all_candidates = []

            if self.pinecone_manager.offline:
                for namespace in namespaces:
                    candidates = self._offline_retrieve_from_namespace(
                        query_dense, query_sparse, namespace, top_k
                    )
                    all_candidates.extend(candidates)
            else:
                for namespace in namespaces:
                    try:
                        response = self.pinecone_manager._index.query(
                            vector=query_dense,
                            sparse_vector=query_sparse,
                            top_k=top_k,
                            include_metadata=True,
                            namespace=namespace,
                        )
                        candidates = [
                            {
                                "id": match.get("id"),
                                "metadata": match.get("metadata", {}),
                            }
                            for match in response.get("matches", [])
                        ]
                        all_candidates.extend(candidates)
                        logger.debug(
                            f"Namespace '{namespace}': {len(candidates)} docs"
                        )
                    except Exception as ns_error:
                        logger.warning(
                            f"Error querying namespace '{namespace}': {str(ns_error)}"
                        )

            logger.info(
                f"RAG retrieve | query={query[:40]!r} | namespaces={len(namespaces)} | total_docs={len(all_candidates)}"
            )

            return all_candidates

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def _offline_retrieve_from_namespace(
        self,
        query_dense: List[float],
        query_sparse: Dict,
        namespace: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve from in-memory store for a specific namespace."""
        results = []

        for doc_id, doc_data in self.pinecone_manager._store.items():
            # Filter by namespace
            if not doc_id.startswith(f"{namespace}#"):
                continue

            metadata = doc_data.get("metadata", {})
            if metadata.get("source") != namespace:
                continue

            results.append({"id": doc_id, "metadata": metadata})

        return results[:top_k]

    def format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for use in prompts.

        Args:
            documents: List of documents from Pinecone

        Returns:
            Formatted string for prompt context
        """
        if not documents:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")

            # Format based on source type
            if source == "products":
                text = f"{i}. [Product] {metadata.get('name', 'Unknown')}"
                text += f" - {metadata.get('desc', '')}"
                text += f" (Category: {metadata.get('category', 'N/A')}, Price: ${metadata.get('price', 'N/A')})"

            elif source == "orders":
                text = f"{i}. [Order] Order ID: {metadata.get('order_id', 'N/A')}"
                text += f" - Total: ${metadata.get('total', 'N/A')}"
                text += f" (Customer: {metadata.get('customer_id', 'N/A')}, Date: {metadata.get('date', 'N/A')})"

            elif source == "churn_predictions":
                text = f"{i}. [Churn Risk] Customer: {metadata.get('customer_id', 'N/A')}"
                text += f" - Risk: {metadata.get('churn_probability', 'N/A')}"
                text += f" (Last Purchase: {metadata.get('days_since_purchase', 'N/A')} days ago)"

            elif source == "pricing_decisions":
                text = f"{i}. [Pricing] Product: {metadata.get('product_id', 'N/A')}"
                text += f" - Discount: {metadata.get('discount_percent', 'N/A')}%"
                text += f" (Decision: {metadata.get('decision_type', 'N/A')})"

            elif source == "search_logs":
                text = f"{i}. [Search] Query: {metadata.get('search_query', 'N/A')}"
                text += f" - Results: {metadata.get('results_count', 'N/A')}"
                text += f" (Conversions: {metadata.get('conversions', 'N/A')})"

            else:
                text = f"{i}. {metadata.get('name', 'Unknown')}"

            formatted.append(text)

        return "\n".join(formatted)

    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings_dense: List[List[float]],
        embeddings_sparse: List[Any],
        namespace: str,
    ) -> None:
        """
        Upsert documents to a specific namespace.

        Args:
            documents: List of document dictionaries
            embeddings_dense: List of dense embeddings
            embeddings_sparse: List of sparse embeddings
            namespace: Target namespace
        """
        try:
            if self.pinecone_manager.offline:
                # Offline mode: store in memory
                for doc, dense, sparse in zip(documents, embeddings_dense, embeddings_sparse):
                    doc_id = f"{namespace}#{doc.get('id', '')}"
                    metadata = doc.get("metadata", {})
                    metadata["source"] = namespace
                    self.pinecone_manager._store[doc_id] = {
                        "dense": dense,
                        "sparse": sparse,
                        "metadata": metadata,
                    }
                logger.info(
                    f"Upserted {len(documents)} documents to {namespace} (offline mode)"
                )
            else:
                # Online mode: upsert to Pinecone with namespace
                pinecone_vectors = []
                for doc, dense, sparse in zip(documents, embeddings_dense, embeddings_sparse):
                    doc_id = f"{namespace}#{doc.get('id', '')}"
                    metadata = doc.get("metadata", {})
                    metadata["source"] = namespace
                    pinecone_vectors.append({
                        "id": doc_id,
                        "values": dense,
                        "sparse_values": sparse,
                        "metadata": metadata,
                    })

                if pinecone_vectors and self.pinecone_manager._index:
                    # Upsert with namespace
                    self.pinecone_manager._index.upsert(
                        vectors=pinecone_vectors, namespace=namespace
                    )
                    logger.info(
                        f"Upserted {len(documents)} documents to namespace: {namespace}"
                    )

        except Exception as e:
            logger.error(f"Error upserting documents to {namespace}: {str(e)}")
            raise
