# BehaviorIQ ML Service

BehaviorIQ ML Service is a FastAPI microservice providing behavioral commerce intelligence for e-commerce. It delivers session intent scoring, RFM churn prediction, user behavioral vectors, hybrid search with intent-aware reranking, and a RAG-powered AI commerce assistant.

**Live instances:**
- https://biq-ml.nxs.bd — ML service
- https://biq-api.nxs.bd — API service + Swagger docs
- https://biq-web.nxs.bd — Storefront & dashboard
- https://biq-grafana.nxs.bd — Grafana observability

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![status](https://img.shields.io/badge/status-experimental-yellow)

---

## Quick Start

```bash
# Install
python -m pip install -r requirements.txt

# Run locally
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Or with full observability stack
docker-compose up --build
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PINECONE_API` | cloud only | — | Pinecone API key for hybrid search |
| `HF_API_KEY` | cloud only | — | Hugging Face API key for embeddings |
| `PINECONE_INDEX` | no | `behavioriq-products` | Pinecone index name |
| `PINECONE_CLOUD` | no | `aws` | Pinecone serverless cloud |
| `PINECONE_REGION` | no | `us-east-1` | Pinecone serverless region |
| `LLM_PROVIDER` | no | `gemini` | RAG LLM: `claude`, `openai`, or `gemini` |
| `ANTHROPIC_API_KEY` | for claude | — | Anthropic API key |
| `OPENAI_API_KEY` | for openai | — | OpenAI API key |
| `GOOGLE_API_KEY` | for gemini | — | Google AI API key |
| `UPSERT_PRODUCTS_ON_STARTUP` | no | `false` | Set `true` to upsert products to Pinecone on boot |
| `PRODUCT_CATALOG_PATH` | no | `data/products.json` | Path to shared product catalog |
| `LOKI_URL` | no | `http://localhost:3100` | Loki log push endpoint |

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /ml/intent-score` | Session purchase intent score (0–100) with per-signal breakdown |
| `POST /ml/churn-predict` | Churn probability (0–1) with RFM breakdown + recommended action |
| `POST /ml/user-vector` | Build user behavioral vector from recent product views |
| `POST /ml/product-embed` | Embed a single product (TF-IDF) and register in the index |
| `POST /ml/search` | **Full pipeline**: hybrid retrieval → intent analysis → behavior-aware rerank |
| `POST /ml/search-rerank` | Legacy rerank: cosine similarity + keyword + pricing |
| `GET /ml/search/index-status` | Pinecone index status |
| `POST /ml/chat` | RAG-powered AI commerce assistant (products, orders, churn, pricing, search logs) |
| `POST /ml/ingest` | Ingest data into a namespaced Pinecone index |
| `POST /ml/ingest/demo` | Seed demo data across all RAG namespaces |
| `GET /ml/search/index-status` | Hybrid index status |
| `GET /metrics` | Prometheus metrics |
| `GET /health` | Health check |

See [project_markdowns/behavioriq-ml-integration.md](project_markdowns/behavioriq-ml-integration.md) for full request/response schemas.

### Quick Examples

```bash
# Intent score
curl -X POST http://localhost:8001/ml/intent-score \
  -H 'Content-Type: application/json' \
  -d '{"product_visit_count":0.7,"time_on_product_page":0.6,"cart_add_events":0.2,"scroll_depth":0.5,"avg_spend_score":0.5,"session_recency":0.9}'

# Churn prediction
curl -X POST http://localhost:8001/ml/churn-predict \
  -H 'Content-Type: application/json' \
  -d '{"days_since_last_purchase":42,"total_order_count":3,"avg_order_value":78.5}'

# Full search pipeline
curl -X POST http://localhost:8001/ml/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"cheap running shoes","churn_score":0.2,"top_k":20}'

# RAG chat
curl -X POST http://localhost:8001/ml/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are my top selling products?"}'
```

## Project Layout

```
ml-service/
  main.py                          # FastAPI app, startup, all endpoints
  catalog.py                       # Shared product catalog loader
  logger.py / metrics.py           # Logging (stdout + Loki) + Prometheus metrics

  models/
    intent.py                      # Deterministic intent scoring formula
    churn_model.py                 # LogisticRegression churn (Kaggle-trained + fallback)
    churn.py                       # Legacy RFM formula (deprecated)
    embedder.py                    # TF-IDF product/user vector builder
    preprocessor.py                # Feature normalization for intent
    reranker.py                    # Legacy rerank compatibility shim
    reranker_updated.py            # Full pipeline: CloudEmbedder, SparseEncoder,
                                   #   PineconeManager, IntentAnalyzer, BehaviorIQReranker

  rag/
    vector_store.py                # Multi-namespace Pinecone store for RAG
    document_builder.py            # Converts raw data to RAG documents
    prompts.py                     # LLM prompt templates
    demo_data.py                   # Demo data generator (orders, churn, pricing, search)

  chains/
    commerce_assistant.py          # LangChain RAG chain (Claude/OpenAI/Gemini)

  api/
    chat.py                        # /ml/chat endpoint
    ingest.py                      # /ml/ingest endpoints

  schemas/
    requests.py                    # All Pydantic request/response models

  seeds/
    generator.py                   # TF-IDF embedder init at startup

  saved_models/                    # Pickled artifacts (embedder.pkl, churn_model.pkl)
  data/
    train.py                       # Churn model training on Kaggle Online Retail dataset
    synthetic_churn.py             # Synthetic data generators

  project_markdowns/               # Architecture + integration docs
  grafana/                         # Pre-configured dashboards + data sources
  test_results/                    # Manual test reports

  Dockerfile / docker-compose.yml  # Containerized deployment
  requirements.txt
```

## Architecture

**Search Pipeline** (`/ml/search`):
```
query → CloudEmbedder (HF dense) + SparseEncoder (BM25)
      → Pinecone hybrid retrieval (or in-memory TF-IDF fallback)
      → IntentAnalyzer (price_sensitive / premium / urgent_buy / gift / comparison / exploratory)
      → BehaviorIQReranker (vector × intent × pricing × churn signals)
      → Ranked results with explanations
```

**Churn Model** (`/ml/churn-predict`):
- Trained on real [Kaggle Online Retail](https://www.kaggle.com/datasets/tunguz/online-retail) dataset
- RFM features → QuantileTransformer → LogisticRegression with Platt scaling
- Falls back to synthetic training if no saved model exists
- Blends ML probability with calibrated formula for SME-relevant ranges

**RAG Chat** (`/ml/chat`):
- Multi-namespace retrieval from Pinecone (products, orders, churn predictions, pricing decisions, search logs)
- LangChain LCEL chain with configurable LLM (Claude Sonnet, GPT-4, Gemini 2.5 Flash)

## Observability

The `docker-compose.yml` spins up:
- **Prometheus** — scrapes `/metrics` every 15s
- **Loki** — log aggregation
- **Grafana** — pre-provisioned dashboards + data sources

All stack components are connected via the `observability` Docker network.

## Documentation

- [Integration contract (schemas + examples)](project_markdowns/behavioriq-ml-integration.md)
- [RAG / LangChain integration](project_markdowns/langchain-rag-integration.md)
- [Project summary & gaps](project_markdowns/project-summary.md)
- [FastAPI Swagger UI](http://localhost:8001/docs) (when running)

## Contributing

Open an issue or submit a PR to the `main` branch. Keep changes focused and testable.

**Maintainer:** `smabdullah2002` (GitHub)

---

Built as a hackathon demo. Integrates scikit-learn, FastAPI, LangChain, Pinecone, and Hugging Face.






