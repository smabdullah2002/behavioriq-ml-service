# How the RAG + LangChain Chat Works

## Overview

When you send a question to `/ml/chat`, the system goes through 6 steps to return a grounded, data-backed answer.

```
Question
   ↓
Embed (HuggingFace)
   ↓
Search Pinecone (5 namespaces)
   ↓
Format context
   ↓
LangChain builds prompt → Gemini answers
   ↓
Return answer + sources
```

---

## Step 1 — Turn the Question into a Vector

```
"high value orders"
        ↓
HuggingFace API (CloudEmbedder)
        ↓
[0.12, -0.33, 0.55, ...] (768 numbers)
```

The question gets converted into numbers that represent its **meaning**.
Similar meanings produce similar numbers. This is called an **embedding**.

---

## Step 2 — Search Pinecone with Those Numbers

```
Query all 5 namespaces:
  - products          → 3 most similar docs
  - orders            → 3 most similar docs
  - churn_predictions → 3 most similar docs
  - pricing_decisions → 3 most similar docs
  - search_logs       → 3 most similar docs
```

Pinecone finds the documents whose vectors are **closest** to the question's vector.
This is how it knows "high value orders" is related to order totals — not by keyword matching, but by meaning.

---

## Step 3 — Format the Retrieved Docs into a Context String

The raw Pinecone matches get formatted into readable text:

```
1. [Order] Order ID: ORD-1019 - Total: $466.7 (Customer: cust_4, Date: 2026-05-22)
2. [Order] Order ID: ORD-1010 - Total: $334.51 (Customer: cust_6, Date: 2026-04-18)
3. [Churn Risk] Customer: cust_7 - Risk: 0.665 (Last Purchase: 103 days ago)
4. [Pricing] Product: p1 - Discount: 32.2% (Decision: loyalty_reward)
5. [Search] Query: athletic wear - Results: 35 (Conversions: 4)
...
```

---

## Step 4 — LangChain Builds the Prompt

This is LangChain's job. It fills in a template with the context and question:

```
You are an AI commerce analyst for BehaviorIQ...

Context:
1. [Order] Order ID: ORD-1019 - Total: $466.7...
2. [Order] Order ID: ORD-1010 - Total: $334.51...
3. [Churn Risk] Customer: cust_7 - Risk: 0.665...
...

Question: high value orders

Answer:
```

LangChain takes the **prompt template** + **context** + **question** and assembles the final text to send to the LLM.

In code this looks like:

```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"context": context, "question": question})
```

That single pipe `|` is LCEL (LangChain Expression Language) — it chains the prompt, model, and parser together.

---

## Step 5 — Gemini Reads the Context and Answers

```
Assembled prompt
        ↓
   Gemini API
        ↓
"The highest value orders are ORD-1019 ($466.7)
 from cust_4 and ORD-1010 ($334.51) from cust_6..."
```

Gemini answers **only based on what's in the context**. This is why it doesn't hallucinate — it has real business data in front of it, not just general knowledge.

---

## Step 6 — Return to the Frontend

```json
{
  "answer": "The highest value orders are ORD-1019 ($466.7) from cust_4...",
  "sources": [
    { "name": "Order ORD-1019", "category": "Order" },
    { "name": "Order ORD-1010", "category": "Order" },
    { "name": "Churn Risk - cust_7", "category": "Churn Risk" }
  ],
  "pipeline_ms": 6631
}
```

---

## What Each Part Is Responsible For

| Component | Role |
|---|---|
| **HuggingFace Embedder** | Converts question and documents into vectors |
| **Pinecone** | Stores all business data as vectors, finds the most relevant ones |
| **LangChain** | Wires the prompt template, LLM, and output parser together |
| **Gemini** | Reads the context and generates the final answer |
| **FastAPI** | Exposes the `/ml/chat` endpoint to the outside world |

---

## Why RAG Instead of Just Asking the LLM?

| Without RAG | With RAG |
|---|---|
| LLM guesses from general knowledge | LLM answers from your actual data |
| Hallucinated order IDs and amounts | Real order IDs and amounts |
| No awareness of your customers | Knows your specific customers |
| Stale — only knows up to training cutoff | Always up to date with your latest data |

---

## The Whole Thing in One Line

> **Convert question → find relevant data → give data + question to LLM → LLM answers using only that data.**

LangChain's only job is **Step 4** — wiring the prompt template, the LLM, and the output parser together so you don't have to write that plumbing code yourself.

---

## API Schemas for Backend Integration

These are the exact schemas your backend needs to send/receive.

### `POST /ml/chat` — Ask the AI Commerce Assistant

**Request** (`ChatRequest`):
```json
{
  "question": "high value orders",
  "conversation_history": null
}
```

| Field | Type | Description |
|---|---|---|
| `question` | `string` | The question to ask (required) |
| `conversation_history` | `list[dict]` or `null` | Optional past messages `[{"role": "user", "content": "..."}]` |

**Response** (`ChatResponse`):
```json
{
  "answer": "The highest value orders are ORD-1019 ($466.7) from cust_4...",
  "sources": [
    { "name": "Order ORD-1019", "category": "Order" },
    { "name": "Order ORD-1010", "category": "Order" },
    { "name": "Churn Risk - cust_7", "category": "Churn Risk" }
  ],
  "pipeline_ms": 6631
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | `string` | The AI-generated answer |
| `sources` | `list[dict]` or `null` | Source documents used — each has `name` and `category` |
| `pipeline_ms` | `float` or `null` | Total end-to-end time in milliseconds |

---

### `POST /ml/ingest` — Push Data into RAG Namespaces

**Request** (`IngestRequest`):
```json
{
  "namespace": "orders",
  "data": [ ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `namespace` | `string` | One of: `products`, `orders`, `churn_predictions`, `pricing_decisions`, `search_logs` |
| `data` | `list[object]` | Array of records (see per-namespace formats below) |

**Response** (`IngestResponse`):
```json
{
  "success": true,
  "namespace": "orders",
  "documents_ingested": 10,
  "message": "Successfully ingested 10 documents"
}
```

---

### `POST /ml/ingest/demo` — Seed Demo Data

**Response**:
```json
{
  "message": "Demo data ingestion completed",
  "results": {
    "products": { "success": true, "documents_ingested": 20, "message": "..." },
    "orders": { "success": true, "documents_ingested": 15, "message": "..." },
    ...
  }
}
```

---

### Data Formats Per Namespace

#### `products`
```json
{
  "id": "p1",
  "name": "Wireless Mouse",
  "desc": "Ergonomic wireless mouse with 2.4GHz",
  "category": "Electronics",
  "brand": "TechBrand",
  "price": 29.99,
  "rating": 4.5,
  "popularity": 0.85
}
```

#### `orders`
```json
{
  "id": "ord_1",
  "order_id": "ORD-1019",
  "customer_id": "cust_4",
  "total": 466.70,
  "items_count": 3,
  "date": "2026-05-22",
  "status": "completed"
}
```

#### `churn_predictions`
```json
{
  "id": "churn_1",
  "customer_id": "cust_7",
  "churn_probability": 0.665,
  "days_since_purchase": 103,
  "rfm_score": 2.3,
  "predicted_action": "re-engagement"
}
```

#### `pricing_decisions`
```json
{
  "id": "pd_1",
  "product_id": "p1",
  "product_name": "Wireless Mouse",
  "discount_percent": 32.2,
  "decision_type": "loyalty_reward",
  "effective_date": "2026-06-01",
  "reason": "Customer loyalty program"
}
```

#### `search_logs`
```json
{
  "id": "log_1",
  "search_query": "athletic wear",
  "results_count": 35,
  "conversions": 4,
  "conversion_rate": 0.114,
  "avg_position": 2.3,
  "date": "2026-05-20"
}
```

---

### Internal Document Format (What Gets Stored in Pinecone)

Each namespace record is converted into this internal structure before upsert:
```json
{
  "id": "orders#ORD-1019",
  "text": "Order ORD-1019 from customer cust_4 total 466.7 items 3",
  "metadata": {
    "source": "orders",
    "name": "Order ORD-1019",
    "category": "Order",
    "order_id": "ORD-1019",
    "customer_id": "cust_4",
    "total": 466.70,
    "items_count": 3,
    "date": "2026-05-22",
    "status": "completed"
  }
}
```

The `text` field is what gets embedded and searched. The `metadata` is returned as context when a match is found. The `id` is prefixed with `{namespace}#` to guarantee uniqueness across namespaces.

---

### Prompt Template (What LangChain Uses)

```text
You are an AI commerce analyst for BehaviorIQ, helping SME owners understand their business data.

You have access to data from multiple sources:
- [Order] - customer orders and revenue
- [Churn Risk] - customers at risk of leaving
- [Pricing] - active discounts and pricing decisions
- [Search] - what customers are searching for and which searches convert

Answer the question using the provided context. Be specific and reference the actual data.
If the context doesn't contain relevant information for the question, say so clearly.

Context:
{context}

Question: {question}

Answer:
```

LangChain fills in `{context}` and `{question}` dynamically, then sends the assembled prompt to the LLM.

---

### Full Service Integration Checklist

| Step | What Your Backend Does |
|---|---|
| **1. Ingest data** | `POST /ml/ingest` for each namespace with your business data |
| **2. (Optional) Seed demo** | `POST /ml/ingest/demo` for test data |
| **3. Ask questions** | `POST /ml/chat` with user questions |
| **4. Parse response** | Read `answer` (text), `sources` (citations), `pipeline_ms` (latency) |

The service handles embedding, search, prompt-building, and LLM generation — you just send data and questions.
