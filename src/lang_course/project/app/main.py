from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

from app.config import get_settings
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector
from app.agent import productionAgent as ProductionAgent

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

app = FastAPI(title="Production API", lifespan=lifespan)
