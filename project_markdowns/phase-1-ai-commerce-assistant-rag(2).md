# Phase 1 - AI Commerce Assistant (LangChain + RAG-Powered)

## Goal
Transform BehaviorIQ from a behavioral analytics platform into an AI-powered commerce intelligence platform by introducing a Retrieval-Augmented Generation (RAG) assistant orchestrated with LangChain.

The assistant will allow SME owners to ask natural language questions about their business and receive context-aware answers grounded in real platform data.

## Core Idea

LangChain is the glue that connects the AI pieces together.

It is not the model itself and it is not the vector database. It simply coordinates the workflow:

```text
User asks a question
      ↓
Convert question to embedding
      ↓
Search Pinecone
      ↓
Get relevant documents
      ↓
Build prompt
      ↓
Send to GPT/Gemini
      ↓
Return answer
```

## Objectives

### Business Objectives

- Improve accessibility of analytics for non-technical users
- Provide actionable insights instead of raw metrics
- Increase visibility into pricing, churn, and search behavior
- Demonstrate practical AI capabilities beyond prediction models

### Technical Objectives

- Introduce a LangChain-based RAG architecture
- Reuse existing Pinecone infrastructure
- Reuse existing embeddings pipeline
- Reuse existing behavioral and pricing data
- Add conversational analytics capabilities
- Keep the current prediction models intact

## Existing BehaviorIQ Flow

The current platform already has the right outer shape:

```text
React Dashboard
      ↓
NestJS Backend
      ↓
FastAPI ML Service
      ↓
Pinecone
      ↓
Embeddings
```

For Phase 1, LangChain fits inside the ML service and orchestrates the RAG flow without replacing the rest of the system.

## Feature Overview

### AI Commerce Assistant

A chat interface inside the SME Dashboard.

Users can ask questions such as:

- Why are conversions dropping this week?
- Which users are most likely to churn?
- Why did this customer receive a discount?
- What products are trending?
- Which categories generate the highest revenue?
- What search terms are converting best?

## Where LangChain Goes

LangChain should live inside the ML service.

Suggested structure:

```text
ml-service/

main.py

rag/
    retriever.py
    prompts.py
    vector_store.py

chains/
    commerce_assistant.py

api/
    chat.py
```

LangChain will primarily live inside:

- `rag/`
- `chains/`

## System Architecture

```text
React Dashboard
        |
        v
NestJS Backend
        |
        v
POST /ml/chat
        |
        v
FastAPI ML Service
        |
        v
LangChain RAG Chain
        |
        +-- Retriever
        +-- Prompt Template
        +-- Conversation Context
        +-- LLM Call
                |
                v
             Pinecone
                |
                v
           Embeddings
                |
                v
          Generated Answer
```

## Data Sources

The assistant should retrieve information from the following sources.

### Products

Used for:

- Product recommendations
- Catalog questions
- Product comparisons

Database table:

```text
products
```

### Pricing Decisions

Used for:

- Discount reasoning
- Revenue analysis
- Pricing effectiveness

Database table:

```text
pricing_decisions
```

### Churn Predictions

Used for:

- Retention analysis
- Risk identification
- Customer segmentation

Database table:

```text
churn_predictions
```

### Orders

Used for:

- Revenue insights
- Customer spending patterns
- Trend analysis

Database table:

```text
orders
```

### Search Logs

Used for:

- Search analytics
- Intent analysis
- Product discovery insights

Database table:

```text
search_logs
```

## RAG Pipeline

### Step 1 - Data Extraction

Extract records from:

- products
- pricing_decisions
- churn_predictions
- orders
- search_logs

Convert records into structured documents.

Example:

```text
Product: Nike Air Zoom Pegasus

Category: Running Shoes

Price: $120

Brand: Nike

Description:
Lightweight running shoe designed for long-distance training.
```

### Step 2 - Embedding Generation

Generate embeddings using the existing embedding service.

Current options:

- Sentence Transformers
- Hugging Face Embeddings

Store vectors in Pinecone.

### Step 3 - Retrieval

When a user asks a question:

```text
Which products are trending?
```

LangChain handles the retrieval flow:

1. Generates query embedding
2. Searches Pinecone
3. Retrieves relevant documents
4. Builds a context package

### Step 4 - Prompt Construction

Example prompt:

```text
You are an AI commerce analyst.

Answer using ONLY the provided context.

Context:
{retrieved_documents}

Question:
{user_question}
```

### Step 5 - LLM Response

Provider:

- Claude
- OpenAI
- Gemini

Output:

```text
Running shoes are currently the fastest-growing category,
accounting for 28% of recent product views and generating
a 12% increase in revenue over the previous week.
```

## LangChain Workflow Example

Suppose the SME owner asks:

```text
Which customers are likely to churn?
```

### Step 1

The frontend sends:

```json
{
  "question": "Which customers are likely to churn?"
}
```

to:

```text
POST /ml/chat
```

### Step 2

LangChain receives the question.

```text
question = request.question
```

### Step 3

LangChain converts the question into an embedding.

```text
embedding = embeddings.embed_query(question)
```

Example:

```text
[0.12, -0.33, 0.55, ...]
```

### Step 4

LangChain searches Pinecone.

```text
docs = retriever.invoke(question)
```

Pinecone can return documents such as:

```text
Customer: John Doe

Churn Probability: 92%

Last Purchase:
45 days ago
```

and:

```text
Customer: Sarah Smith

Churn Probability: 87%

Last Purchase:
38 days ago
```

### Step 5

LangChain builds a prompt automatically.

```text
You are an AI commerce analyst.

Context:

Customer: John Doe
Churn Probability: 92%

Customer: Sarah Smith
Churn Probability: 87%

Question:
Which customers are likely to churn?
```

### Step 6

LangChain sends the prompt to GPT or Gemini.

```text
answer = chain.invoke(...)
```

The model returns:

```text
The highest-risk customers are John Doe and Sarah Smith.
Both have churn probabilities above 85% and have not made recent purchases.
```

### Step 7

Return the answer to the frontend.

```json
{
  "answer": "The highest-risk customers are..."
}
```

The dashboard displays it in chat.

## Where Pinecone Fits

Pinecone stores business knowledge as vectors.

For example:

```text
Product Data
Pricing Data
Search Logs
Churn Predictions
Orders
```

stored as embeddings.

```text
Question
      ↓
Embedding
      ↓
Pinecone
      ↓
Relevant Documents
```

LangChain talks to Pinecone and uses the retrieved context to build the final answer.

## Where OpenAI or Gemini Fits

The LLM only generates the final response.

```text
Context from Pinecone
      ↓
GPT / Gemini
      ↓
Answer
```

Without context:

```text
Hallucination
```

With context:

```text
Grounded Answer
```

That is the purpose of RAG.

## Do You Need LangChain?

Technically:

```text
No
```

You could write the retrieval and prompting logic yourself.

But LangChain already provides:

- Pinecone integration
- Prompt templates
- Retrieval chains
- Conversation memory
- Source citations
- Agent support later

So instead of writing:

```text
200-300 lines
```

you might write:

```text
20-30 lines
```

## Backend Changes

### New Endpoint

```text
POST /ml/chat
```

Request:

```json
{
  "question": "Why are conversions dropping?"
}
```

Response:

```json
{
  "answer": "Conversion rates have declined primarily due to..."
}
```

## New Folder Structure

```text
ml-service/
|
|-- rag/
|   |-- retriever.py
|   |-- vector_store.py
|   |-- document_builder.py
|   |-- embeddings.py
|   `-- prompts.py
|
|-- chains/
|   `-- commerce_assistant.py
|
|-- api/
|   `-- chat.py
|
`-- main.py
```

## Dashboard Changes

Add a new page:

```text
Dashboard
└── AI Commerce Assistant
```

Features:

- Chat interface
- Conversation history
- Suggested questions
- Source references
- AI-generated insights

## Suggested Starter Questions

```text
Why are conversions dropping?

Which customers are likely to churn?

What products are trending?

Which discounts generate the highest conversion rate?

What search terms lead to purchases?

Which category generates the most revenue?
```

## Success Metrics

### Functional

- Chat endpoint operational
- Pinecone retrieval working
- Grounded responses generated
- Dashboard integration complete

### User Experience

- Response time under 5 seconds
- Contextually accurate answers
- Clear business insights
- Minimal hallucinations

## Deliverables

### Backend

- RAG pipeline
- Chat endpoint
- Pinecone integration
- LangChain orchestration layer

### Frontend

- AI Assistant page
- Chat UI
- Suggested prompts

### AI

- Context retrieval
- Prompt generation
- Commerce-focused responses

## Expected Outcome

BehaviorIQ evolves from:

```text
Behavior Tracking Platform
```

into:

```text
AI Commerce Intelligence Platform
```

where SME owners can interact with their business data through natural language and receive grounded, actionable insights powered by LangChain + RAG.