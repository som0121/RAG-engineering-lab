from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable

from app.config import get_settings
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agent import productionAgent as ProductionAgent
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
)

load_dotenv()

logger = get_logger()

# Shared components, initialized in lifespan()
security = None
cache = None
metrics = None
agent = None


#=== Lifespan (startup/shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    This is the modern FastAPI pattern (replaces @app.on_event)

    """
    global security, cache, metrics, agent

    settings = get_settings()

    logger.info("Starting production API...", extra = {"extra_data":{

        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langchain_tracing_v2,

    }})

    #Initialize components
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds = settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()

    logger.info("All components initialized. Ready to serve requests.")

    yield # App is running

    #Shutdown

    logger.info("Shutting down...", extra = {"extra_data": metrics.get_summary()})

# === Rate Limiter Setup ===

limiter = Limiter(key_func= get_remote_address)

# ==== FastAPI APP ====

app = FastAPI(
    title= "Production LangGraph API",
    description= "A production-ready chat API with security, caching, and observability.",
    version= "1.0.0",
    lifespan= lifespan,
)

app.state.limiter = limiter

# === Exception Handlers ====

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return a structured 429 response when a client exceeds the rate limit."""

    client_ip = get_remote_address(request)

    logger.warning("Rate limit exceeded", extra = {"extra_data": {
        "client_ip": client_ip,
        "path": request.url.path,
        "limit": str(exc.detail),
    }})

    response = JSONResponse(
        status_code = 429,
        content = ErrorResponse(
            error = "Rate limit exceeded",
            detail = f"Too many requests: {exc.detail}. Please try again later.",
        ).model_dump(),
    )

    # Adds Retry-After / X-RateLimit-* headers if headers_enabled on the limiter
    return request.app.state.limiter._inject_headers(response, request.state.view_rate_limit)



@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so clients never see a raw stack trace."""

    logger.error(f"Unhandled error: {exc}", extra = {"extra_data": {
        "path": request.url.path,
        "error": str(exc),
    }})

    return JSONResponse(
        status_code = 500,
        content = ErrorResponse(
            error = "Internal server error",
            detail = "An unexpected error occurred.",
        ).model_dump(),
    )


#==== ENDPOINTS ===#

@app.post("/chat", response_model= ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name = "chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint

    Flow:
    1. Security check (injection + PII masking)
    2. Cache lookup
    3. LangGraph agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    with RequestTimer() as timer:
        security_notes = []

        #---- Step 1: Security Check ----

        is_allowed, cleaned_message, notes = security.check_input(body.message)

        if not is_allowed:
            logger.warning("Request blocked by security", extra = {"extra_data": {
                "reason": notes,
                "thread_id": body.thread_id,
            }})

            metrics.record_request(latency_ms = timer.elapsed_ms, error = True)
            raise HTTPException(
                status_code = 400,
                detail = "Your message was blocked by our security filters.",
            )

        security_notes.extend(notes)

        #---- Step 2: Cache Lookup ----

        cached_response = cache.get(cleaned_message)

        if cached_response is not None:
            metrics.record_request(latency_ms = timer.elapsed_ms, cache_hit = True)
            logger.info("Cache hit", extra = {"extra_data": {
                "thread_id": body.thread_id,
            }})

            return ChatResponse(
                response = cached_response,
                thread_id = body.thread_id,
                model_used = "cache",
                cached = True,
                processing_time_ms = timer.elapsed_ms,
            )

        #---- Step 3: Invoke LangGraph Agent ----

        try:
            # agent.invoke is synchronous, so run it off the event loop
            result = await run_in_threadpool(agent.invoke, cleaned_message)

        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", extra = {"extra_data": {
                "thread_id": body.thread_id,
                "error": str(e),
            }})

            metrics.record_request(latency_ms = timer.elapsed_ms, error = True)
            raise HTTPException(
                status_code = 500,
                detail = "An error occurred while processing your request.",
            )

        response_text = result["response"]
        model_used = result["model_used"]

        #---- Step 4: Output Validation ----

        is_valid, validated_response, output_warning = security.check_output(response_text)

        if not is_valid:
            security_notes.append(output_warning)

        #---- Step 5: Cache Store ----

        # Don't cache the fallback apology from the agent's error handler
        if model_used != "error_handler":
            cache.set(cleaned_message, validated_response)

        #---- Step 6: Log & Record Metrics ----

        # Rough token estimate (~1.3 tokens per word)
        input_tokens = int(len(cleaned_message.split()) * 1.3)
        output_tokens = int(len(validated_response.split()) * 1.3)

        metrics.record_request(
            latency_ms = timer.elapsed_ms,
            input_tokens = input_tokens,
            output_tokens = output_tokens,
            error = model_used == "error_handler",
            cache_hit = False,
        )

        if security_notes:
            logger.info("Security notes", extra = {"extra_data": {
                "notes": security_notes,
                "thread_id": body.thread_id,
            }})

        logger.info("Request completed", extra = {"extra_data": {
            "thread_id": body.thread_id,
            "model_used": model_used,
            "latency_ms": timer.elapsed_ms,
        }})

        return ChatResponse(
            response = validated_response,
            thread_id = body.thread_id,
            model_used = model_used,
            cached = False,
            processing_time_ms = timer.elapsed_ms,
        )


@app.get("/health", response_model= HealthResponse)
async def health():
    """Liveness check plus a quick status of each component."""

    settings = get_settings()

    return HealthResponse(
        status = "healthy",
        environment = settings.app_env,
        checks = {
            "agent": "ready" if agent else "not_initialized",
            "cache": cache.stats if cache else "not_initialized",
            "security": "ready" if security else "not_initialized",
        },
    )


@app.get("/metrics", response_model= MetricsResponse)
async def get_metrics():
    """Aggregated request, latency, token and cache metrics."""

    return MetricsResponse(**metrics.get_summary())

@app.get("/cache/stats")
async def cache_stats():
    """
    Cache performance statistics.
    """

    return cache.stats
