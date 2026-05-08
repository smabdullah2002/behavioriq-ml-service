# BehaviorIQ ML Service - Project Summary

This markdown summarizes what has been built in the current project, what is strong about it, and what still needs work.

## What has been done

The project is a FastAPI-based ML service for BehaviorIQ. It now covers four main capabilities:

- intent scoring for live session behavior
- churn prediction from RFM features
- user vector building from recent product views
- search reranking for product discovery

The service has both a legacy/simple path and a newer BehaviorIQ path.

### 1. Intent scoring

The intent endpoint is implemented as a deterministic formula. It takes normalized session signals such as page views, time on page, cart adds, scroll depth, spend affinity, and session recency. This is good for low-latency scoring because it does not need a trained model.

### 2. Churn prediction

The churn pipeline is based on real Kaggle Online Retail data. The training script in `data/train.py`:

- loads and cleans the dataset
- engineers RFM features
- labels churn
- trains a logistic regression model
- calibrates the probabilities
- saves both model and scaler artifacts

The runtime service loads the trained churn model and uses the saved scaler at inference time.

### 3. User vectors and search reranking

The older rerank path builds user vectors from product history and combines cosine similarity with keyword score. That path is still present.

There is also a newer BehaviorIQ reranker in `models/reranker_updated.py` that adds a more complete flow:

- semantic retrieval
- search intent analysis
- behavior-aware reranking
- pricing-aware scoring

The newer code also supports a Pinecone hybrid style setup and a cloud-embedding fallback path when keys are missing.

### 4. Observability and service plumbing

The service has logging, metrics, startup loading, artifact checks, and health reporting. The app also exposes endpoints for the ML service to be monitored and debugged more easily.

## Strong points

- The service is end-to-end, not just a model notebook.
- The churn model is trained on real data instead of only synthetic data.
- Inference uses the saved scaler, which keeps training and prediction aligned.
- The rerank work has moved beyond plain semantic matching toward behavior-aware scoring.
- The code returns breakdown fields in several places, which helps explain predictions.
- The project includes monitoring, logging, and test artifacts, which makes it easier to debug.
- The newer reranker design is flexible and can support future search intent logic.

## Weaknesses

- The repository still has both legacy code and newer architecture code, so the project is not fully cleaned up yet.
- Some docs are behind the code and still describe older scoring logic.
- The reranking system is still evolving. It now has intent, vector, and pricing ideas, but the intent service itself is still fairly simple.
- Pinecone and cloud embeddings are designed in the code, but they depend on environment keys and fallback behavior, so the production path is not fully guaranteed in every setup.
- The current reranker logic still needs tuning, especially weights and price rules.
- There are multiple files with overlapping responsibilities, which can make maintenance confusing.
- The project still has hackathon-style structure in a few places, so the architecture is improving but not fully polished.

## Current status in one sentence

This is now a working BehaviorIQ ML service with real churn training, deterministic intent scoring, user vectors, and an evolving behavior-aware reranker, but it still needs cleanup, tuning, and doc alignment.

## Best next improvements

- tune rerank weights and pricing rules
- finish the search intent analyzer
- align the markdown docs with the current code
- remove or isolate older legacy reranker paths if they are no longer needed
- add stronger offline evaluation for reranking quality
