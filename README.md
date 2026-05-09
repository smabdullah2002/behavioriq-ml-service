# BehaviorIQ ML Service

BehaviorIQ ML Service is a lightweight FastAPI microservice that provides behavioral commerce intelligence for e-commerce applications. It exposes fast endpoints for session intent scoring, RFM churn prediction, user behavioral vector construction, and a full search pipeline (semantic retrieval → intent analysis → behavior-aware reranking).

Badges
---
<!-- Replace placeholders with your CI status / coverage badges if available -->
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![status](https://img.shields.io/badge/status-experimental-yellow)

Why this project
---
- Fast, reproducible ML endpoints for demoing behavioral personalization
- Full search pipeline with Pinecone hybrid retrieval (dense + sparse) and intent-aware reranking
- Deterministic intent formula + trained churn model for explainability

Contents of this README
---
- What the project does
- Quick start (install, run)
- API summary and examples
- Deployment & environment notes
- Links to integration docs and design plans
- How to contribute and get help

Quick start
---

Prerequisites
- Python 3.10+
- (Optional) Docker & Docker Compose for containerized runs

Install dependencies

```bash
python -m pip install -r requirements.txt
```

Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Run with Docker Compose

```bash
docker-compose up --build
```

Environment variables
- `PINECONE_API` — Pinecone API key (optional; service falls back to TF-IDF offline)
- `HF_API_KEY` — Hugging Face API key for SentenceTransformers embeddings (optional)

Project layout (important files)
---

```
ml-service/
  main.py                   # FastAPI app and endpoints
  models/                   # ML logic: intent, churn, embedder, reranker_updated
  schemas/                  # Pydantic request/response models
  plan_markdowns/           # Architecture and integration docs (contracts)
  saved_models/             # Pickled artifacts (embedder, churn model)
  requirements.txt
  Dockerfile
  docker-compose.yml
```

API summary (short)
---

Use [plan_markdowns/behavioriq-ml-integration.md](plan_markdowns/behavioriq-ml-integration.md) for full request/response contracts.

- `POST /ml/intent-score` — session intent score (0–100)
- `POST /ml/churn-predict` — churn probability (0–1)
- `POST /ml/user-vector` — build a per-user behavioral vector
- `POST /ml/search` — BehaviorIQ full search pipeline (semantic retrieval → intent → rerank)
- `GET /ml/search/index-status` — Pinecone index status (if configured)
- `GET /metrics` — Prometheus metrics
- `GET /health` — health check

Example: intent score

```bash
curl -X POST http://localhost:8001/ml/intent-score \
  -H 'Content-Type: application/json' \
  -d '{
    "product_visit_count": 0.7,
    "time_on_product_page": 0.6,
    "cart_add_events": 0.2,
    "scroll_depth": 0.5,
    "avg_spend_score": 0.5,
    "session_recency": 0.9
  }'
```

Example: churn predict

```bash
curl -X POST http://localhost:8001/ml/churn-predict \
  -H 'Content-Type: application/json' \
  -d '{
    "days_since_last_purchase": 42,
    "total_order_count": 3,
    "avg_order_value": 78.5
  }'
```

Example: full search (rerank)

```bash
curl -X POST http://localhost:8001/ml/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "cheap running shoes",
    "churn_score": 0.2,
    "top_k": 20
  }'
```

Development notes
---

- The code includes a hybrid search pipeline (`models/reranker_updated.py`) which uses a `CloudEmbedder` (HF embeddings with TF-IDF fallback), a `SparseEncoder` for sparse signals, and `PineconeManager` for hybrid retrieval or an in-memory fallback.
- At startup the service seeds product vectors from `data/products.json` (see `seeds/`)
- The service exposes Prometheus metrics and basic tracing via `metrics.py`

Documentation & integration
---

- Integration contract (schemas + examples): [plan_markdowns/behavioriq-ml-integration.md](plan_markdowns/behavioriq-ml-integration.md)
- Architecture & hackathon plan: [plan_markdowns/cloudcamp-hackathon.md](plan_markdowns/cloudcamp-hackathon.md)
- Project summary: [plan_markdowns/project-summary.md](plan_markdowns/project-summary.md)

Contributing & support
---

If you want to contribute:

- Open an issue describing the change or feature
- Create a focused PR against the `main` branch; keep changes small and testable

For questions and help, open an issue in the repository.

Maintainers
---

- Repository owner: `smabdullah2002` (GitHub)

License
---

This repository does not include a license file. Add `LICENSE` to the repo to make usage terms explicit.

Acknowledgements
---

This project was built as part of a hackathon demo and integrates open-source libraries such as scikit-learn, FastAPI, and optional cloud services (Hugging Face, Pinecone).

---

If you'd like, I can also:
- add a minimal `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`
- add CI badges and a sample GitHub Actions workflow
- generate OpenAPI client snippets for frontend teams



