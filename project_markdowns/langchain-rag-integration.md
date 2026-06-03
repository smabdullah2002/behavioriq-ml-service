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
