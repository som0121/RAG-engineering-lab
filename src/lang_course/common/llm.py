import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_model(name: str | None = None, **kwargs):
    """Chat model from MODEL in .env, e.g. "openrouter:anthropic/claude-haiku-5.5"."""
    name = name or os.getenv("MODEL", "openrouter:anthropic/claude-haiku-5.5")
    if name.startswith("openrouter:"):
        return init_chat_model(
            name.removeprefix("openrouter:"),
            model_provider="openai",
            base_url=OPENROUTER_BASE_URL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            **kwargs,
        )
    return init_chat_model(name, **kwargs)


def get_embeddings(name: str | None = None):
    """Embedding model from EMBEDDING_MODEL in .env, e.g. "openrouter:openai/text-embedding-3-small"."""
    name = name or os.getenv("EMBEDDING_MODEL", "openrouter:openai/text-embedding-3-small")
    if name.startswith("openrouter:"):
        return init_embeddings(
            name.removeprefix("openrouter:"),
            provider="openai",
            base_url=OPENROUTER_BASE_URL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            check_embedding_ctx_length=False,  # OpenRouter takes raw text, not tiktoken IDs
        )
    return init_embeddings(name)
