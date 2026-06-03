"""Document builder for converting raw data into RAG documents."""

from typing import List, Dict, Any
from logger import get_logger

logger = get_logger(__name__)


class DocumentBuilder:
    """Convert raw data from different sources into RAG documents."""

    @staticmethod
    def build_product_documents(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert product catalog to documents.

        Args:
            products: List of product dictionaries

        Returns:
            List of formatted documents
        """
        documents = []
        for product in products:
            doc = {
                "id": product.get("id", f"prod_{len(documents)}"),
                "text": f"{product.get('name', '')} {product.get('desc', '')} {product.get('category', '')}",
                "metadata": {
                    "source": "products",
                    "name": product.get("name", ""),
                    "desc": product.get("desc", ""),
                    "category": product.get("category", ""),
                    "brand": product.get("brand", ""),
                    "price": product.get("price", 0),
                    "rating": product.get("rating", 3.5),
                    "popularity": product.get("popularity", 0.5),
                },
            }
            documents.append(doc)
        return documents

    @staticmethod
    def build_order_documents(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert order data to documents.

        Args:
            orders: List of order dictionaries

        Returns:
            List of formatted documents
        """
        documents = []
        for order in orders:
            doc = {
                "id": order.get("id", f"order_{len(documents)}"),
                "text": f"Order {order.get('order_id', '')} from customer {order.get('customer_id', '')} total {order.get('total', '')} items {order.get('items_count', '')}",
                "metadata": {
                    "source": "orders",
                    "name": f"Order {order.get('order_id', '')}",
                    "category": "Order",
                    "order_id": order.get("order_id", ""),
                    "customer_id": order.get("customer_id", ""),
                    "total": order.get("total", 0),
                    "items_count": order.get("items_count", 0),
                    "date": order.get("date", ""),
                    "status": order.get("status", "completed"),
                },
            }
            documents.append(doc)
        return documents

    @staticmethod
    def build_churn_prediction_documents(
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert churn prediction data to documents.

        Args:
            predictions: List of churn prediction dictionaries

        Returns:
            List of formatted documents
        """
        documents = []
        for pred in predictions:
            doc = {
                "id": pred.get("id", f"churn_{len(documents)}"),
                "text": f"Customer {pred.get('customer_id', '')} churn probability {pred.get('churn_probability', '')} days since purchase {pred.get('days_since_purchase', '')}",
                "metadata": {
                    "source": "churn_predictions",
                    "name": f"Churn Risk - {pred.get('customer_id', '')}",
                    "category": "Churn Risk",
                    "customer_id": pred.get("customer_id", ""),
                    "churn_probability": pred.get("churn_probability", 0),
                    "days_since_purchase": pred.get("days_since_purchase", 0),
                    "rfm_score": pred.get("rfm_score", 0),
                    "predicted_action": pred.get("predicted_action", "monitor"),
                },
            }
            documents.append(doc)
        return documents

    @staticmethod
    def build_pricing_decision_documents(
        decisions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert pricing decision data to documents.

        Args:
            decisions: List of pricing decision dictionaries

        Returns:
            List of formatted documents
        """
        documents = []
        for decision in decisions:
            doc = {
                "id": decision.get("id", f"pricing_{len(documents)}"),
                "text": f"Product {decision.get('product_id', '')} discount {decision.get('discount_percent', '')} percent decision type {decision.get('decision_type', '')}",
                "metadata": {
                    "source": "pricing_decisions",
                    "name": f"{decision.get('product_name', 'Product')} - {decision.get('discount_percent', 0)}% off",
                    "category": "Pricing",
                    "product_id": decision.get("product_id", ""),
                    "product_name": decision.get("product_name", ""),
                    "discount_percent": decision.get("discount_percent", 0),
                    "decision_type": decision.get("decision_type", "standard"),
                    "effective_date": decision.get("effective_date", ""),
                    "reason": decision.get("reason", ""),
                },
            }
            documents.append(doc)
        return documents

    @staticmethod
    def build_search_log_documents(
        logs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert search log data to documents.

        Args:
            logs: List of search log dictionaries

        Returns:
            List of formatted documents
        """
        documents = []
        for log in logs:
            doc = {
                "id": log.get("id", f"search_{len(documents)}"),
                "text": f"Search query {log.get('search_query', '')} results {log.get('results_count', '')} conversions {log.get('conversions', '')}",
                "metadata": {
                    "source": "search_logs",
                    "name": f"Search: {log.get('search_query', '')}",
                    "category": "Search",
                    "search_query": log.get("search_query", ""),
                    "results_count": log.get("results_count", 0),
                    "conversions": log.get("conversions", 0),
                    "conversion_rate": log.get("conversion_rate", 0),
                    "avg_position": log.get("avg_position", 0),
                    "date": log.get("date", ""),
                },
            }
            documents.append(doc)
        return documents

    @staticmethod
    def build_documents(
        data: Dict[str, Any], data_type: str
    ) -> List[Dict[str, Any]]:
        """
        Generic document builder that routes to specific builders.

        Args:
            data: Raw data
            data_type: Type of data ("products", "orders", "churn_predictions", etc.)

        Returns:
            List of formatted documents
        """
        builders = {
            "products": DocumentBuilder.build_product_documents,
            "orders": DocumentBuilder.build_order_documents,
            "churn_predictions": DocumentBuilder.build_churn_prediction_documents,
            "pricing_decisions": DocumentBuilder.build_pricing_decision_documents,
            "search_logs": DocumentBuilder.build_search_log_documents,
        }

        builder = builders.get(data_type)
        if not builder:
            logger.error(f"Unknown data type: {data_type}")
            return []

        if not isinstance(data, list):
            data = [data]

        return builder(data)
