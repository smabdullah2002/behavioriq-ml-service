"""LangChain-based commerce assistant chain."""

import os
import time
from typing import Optional, List, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from logger import get_logger
from rag.prompts import SIMPLE_RAG_PROMPT
from rag.vector_store import PineconeVectorStore

logger = get_logger(__name__)


class CommerceAssistant:
    """LangChain-based AI Commerce Assistant for RAG."""

    def __init__(self, vector_store: PineconeVectorStore, provider: str = "gemini"):
        """
        Initialize the commerce assistant with a specific LLM provider.

        Args:
            vector_store: PineconeVectorStore instance for retrieval
            provider: LLM provider ("claude", "openai", or "gemini")
        """
        self.vector_store = vector_store
        self.provider = provider.lower()
        self.llm = self._initialize_llm()
        self.chain = self._build_chain()

    def _initialize_llm(self):
        """Initialize the LLM based on the configured provider."""
        if self.provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return ChatAnthropic(api_key=api_key, model="claude-3-5-sonnet-20241022")

        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            return ChatOpenAI(api_key=api_key, model="gpt-4")

        elif self.provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            return ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.5-flash")

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _build_chain(self):
        """Build the LangChain RAG chain using LCEL."""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=SIMPLE_RAG_PROMPT,
        )
        # Use LCEL (LangChain Expression Language) for LangChain 1.x
        #prompt->gemini->StrOutputParser() is the LCEL expression to create the chain
        chain = prompt | self.llm | StrOutputParser()
        return chain

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: The user's question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer, sources, and timing
        """
        start_time = time.perf_counter()

        try:
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.retrieve(question, top_k=top_k)
            context = self.vector_store.format_documents(retrieved_docs)

            # Generate answer using the chain (LCEL invoke)
            result = self.chain.invoke({"context": context, "question": question})

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Extract sources from retrieved docs
            sources = [
                {
                    "name": doc.get("metadata", {}).get("name", "Unknown"),
                    "category": doc.get("metadata", {}).get("category", "N/A"),
                }
                for doc in retrieved_docs
            ]

            logger.info(
                f"Chat | provider={self.provider} | docs_retrieved={len(retrieved_docs)} | {elapsed_ms}ms"
            )

            return {
                "answer": result.strip(),
                "sources": sources,
                "pipeline_ms": elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "pipeline_ms": elapsed_ms,
            }
