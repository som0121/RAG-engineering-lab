"""
Monitoring and Logging for Production
Structured logging, metrics, and alerts
Using Google Gemini Chat and Embeddings
"""

import logging
import json
import time

from datetime import datetime, timezone
from threading import Lock

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.messages import HumanMessage
from langsmith import traceable

load_dotenv()


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for log aggregation."""

    def format(self, record):

        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging():
    """Setup structured JSON logging."""

    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


class MetricsCollector:
    """Collect and aggregate application metrics."""

    def __init__(self):

        self.lock = Lock()

        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,

            # Embedding-specific metrics
            "embedding_requests_total": 0,
            "embedding_errors_total": 0,
            "embedding_latency_sum": 0.0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ):

        with self.lock:

            self.metrics["requests_total"] += 1
            self.metrics["latency_sum"] += latency_ms
            self.metrics["latency_count"] += 1

            self.metrics["tokens_input"] += input_tokens
            self.metrics["tokens_output"] += output_tokens

            if error:
                self.metrics["errors_total"] += 1

            if cache_hit:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1

    def record_embedding(
        self,
        latency_ms: float,
        error: bool = False,
    ):

        with self.lock:

            self.metrics["embedding_requests_total"] += 1
            self.metrics["embedding_latency_sum"] += latency_ms

            if error:
                self.metrics["embedding_errors_total"] += 1

    def get_summary(self) -> dict:

        with self.lock:

            metrics = self.metrics.copy()

        requests = metrics["requests_total"]
        errors = metrics["errors_total"]

        avg_latency = (
            metrics["latency_sum"] / metrics["latency_count"]
            if metrics["latency_count"] > 0
            else 0
        )

        error_rate = errors / requests if requests > 0 else 0

        cache_total = (
            metrics["cache_hits"] + metrics["cache_misses"]
        )

        cache_hit_rate = (
            metrics["cache_hits"] / cache_total
            if cache_total > 0
            else 0
        )

        embedding_requests = metrics["embedding_requests_total"]

        avg_embedding_latency = (
            metrics["embedding_latency_sum"] / embedding_requests
            if embedding_requests > 0
            else 0
        )

        return {
            "total_chat_requests": requests,
            "total_chat_errors": errors,
            "chat_error_rate": f"{error_rate:.2%}",
            "avg_chat_latency_ms": round(avg_latency, 2),

            "estimated_input_tokens": metrics["tokens_input"],
            "estimated_output_tokens": metrics["tokens_output"],

            "cache_hit_rate": f"{cache_hit_rate:.2%}",

            "total_embedding_requests": embedding_requests,
            "total_embedding_errors": metrics[
                "embedding_errors_total"
            ],
            "avg_embedding_latency_ms": round(
                avg_embedding_latency, 2
            ),
        }


class InstrumentedLLM:
    """Gemini chat and embeddings with instrumentation."""

    def __init__(self):

        # Gemini chat model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

        # Gemini embedding model
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
        )

        self.metrics = MetricsCollector()
        self.logger = setup_logging()

    @traceable(name="gemini_chat_invoke")
    def invoke(self, query: str) -> str:
        """Generate a response using Gemini and record metrics."""

        start_time = time.perf_counter()

        try:

            response = self.llm.invoke(
                [HumanMessage(content=query)]
            )

            result = response.content

            # Estimate tokens from whitespace-separated words.
            # This is only a rough estimate, not actual token usage.
            input_tokens = len(query.split()) * 4 // 3

            output_text = (
                result
                if isinstance(result, str)
                else str(result)
            )

            output_tokens = len(output_text.split()) * 4 // 3

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            self.metrics.record_request(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False,
            )

            self.logger.info(
                "Gemini chat request completed",
                extra={
                    "extra_data": {
                        "latency_ms": round(latency_ms, 2),
                        "estimated_input_tokens": input_tokens,
                        "estimated_output_tokens": output_tokens,
                        "model": "gemini-2.5-flash",
                    }
                },
            )

            return output_text

        except Exception as e:

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            self.metrics.record_request(
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            self.logger.error(
                "Gemini chat request failed",
                extra={
                    "extra_data": {
                        "error": str(e),
                        "latency_ms": round(latency_ms, 2),
                    }
                },
            )

            raise

    @traceable(name="gemini_embedding")
    def embed_text(self, text: str) -> list[float]:
        """Generate a Gemini embedding and record metrics."""

        start_time = time.perf_counter()

        try:

            vector = self.embeddings_model.embed_query(text)

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            self.metrics.record_embedding(
                latency_ms=latency_ms,
                error=False,
            )

            self.logger.info(
                "Gemini embedding generated",
                extra={
                    "extra_data": {
                        "latency_ms": round(latency_ms, 2),
                        "vector_dimensions": len(vector),
                        "model": "gemini-embedding-001",
                    }
                },
            )

            return vector

        except Exception as e:

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            self.metrics.record_embedding(
                latency_ms=latency_ms,
                error=True,
            )

            self.logger.error(
                "Gemini embedding request failed",
                extra={
                    "extra_data": {
                        "error": str(e),
                        "latency_ms": round(latency_ms, 2),
                    }
                },
            )

            raise

def demo_monitoring():

    """Demonstrate Gemini chat, embeddings, and monitoring."""

    app = InstrumentedLLM()

    print("Gemini Monitoring Demo\n")

    # 1. Test Gemini chat
    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What is 2 + 2?",
    ]

    for query in queries:

        try:

            result = app.invoke(query)

            print(f"\nQuery: {query}")
            print(f"Response: {result[:150]}...")

        except Exception as e:

            print(f"\nChat request failed: {e}")

    # 2. Test Gemini embeddings


    print("\n" + "=" * 50)
    print("Testing Gemini Embeddings")
    print("=" * 50)

    text = "What is machine learning?"

    try:

        vector = app.embed_text(text)

        print(f"Text: {text}")
        print(f"Vector dimensions: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")

    except Exception as e:

        print(f"Embedding request failed: {e}")

    # 3. Print metrics summary

    print("\n" + "=" * 50)
    print("Metrics Summary")
    print("=" * 50)

    summary = app.metrics.get_summary()

    for key, value in summary.items():

        print(f"{key}: {value}")

if __name__ == "__main__":

    #logger = setup_logging()
    #logger.info("Logging setup complete", extra={"extra_data": {"app": "langgraph"}})
    demo_monitoring()