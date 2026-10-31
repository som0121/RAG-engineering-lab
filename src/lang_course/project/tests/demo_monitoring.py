import json

from app.monitoring import get_logger, MetricsCollector

# 1. Create a JSON logger

logger = get_logger("rag-engineering-lab")

# 2. Generate actual log messages

logger.info("RAG application started")

logger.info(
    "Document retrieved successfully",
    extra={
        "extra_data": {
            "document_id": "doc_101",
            "retrieval_method": "semantic_search",
            "results_count": 5,
        }
    },
)

logger.warning(
    "Cache miss",
    extra={
        "extra_data": {
            "query": "What is RAG?",
            "cache_hit": False,
        }
    },
)

logger.error(
    "Model request failed",
    extra={
        "extra_data": {
            "model": "gemini-3.5-flash-lite",
            "status_code": 500,
        }
    },
)

# 3. Collect application metrics

metrics = MetricsCollector()

metrics.record_request(
    latency_ms=120.5,
    input_tokens=100,
    output_tokens=200,
    error=False,
    cache_hit=True,
)

metrics.record_request(
    latency_ms=250.0,
    input_tokens=150,
    output_tokens=300,
    error=True,
    cache_hit=False,
)

metrics.record_request(
    latency_ms=90.0,
    input_tokens=80,
    output_tokens=160,
    error=False,
    cache_hit=True,
)

# 4. Print metrics summary as JSON

summary = metrics.get_summary()

print("\n========== METRICS SUMMARY ==========")
print(json.dumps(summary, indent=4))