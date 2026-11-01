FROM python:3.12-slim

WORKDIR /app

# Print logs straight away, don't write .pyc files at runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

#Install uv(fast python package manager) from its official image, pinned to the local version
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# Create non-root user for security (before copying files, so no chown layer is needed)
RUN useradd --create-home appuser

#Copy dependency files first (Docker layer caching)

COPY pyproject.toml uv.lock ./

#Install runtime dependencies into /app/.venv
# The cache mount keeps downloaded wheels between builds, so a dependency change doesn't re-download everything
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Use the venv's python/uvicorn directly (no `uv run` needed at runtime)
ENV PATH="/app/.venv/bin:$PATH"

#Copy application code (keep it as the `app` package so `app.main` imports work)

COPY app/ ./app/

USER appuser

# Expose port
EXPOSE 8000

# Health check (python:3.12-slim has no curl, so use python's urllib)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
