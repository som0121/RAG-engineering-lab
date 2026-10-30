import os
from dotenv import load_dotenv
from langsmith import Client, traceable

load_dotenv(override=True)

@traceable(
    name="langsmith-connection-test",
    project_name="multi-agent-research-system",
)
def test_trace():
    return {
        "message": "Hello from RAG Engineering Lab",
        "status": "success",
    }

if __name__ == "__main__":
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        raise RuntimeError("LANGSMITH_TRACING is not true")

    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is missing")

    result = test_trace()
    print("Trace function result:", result)

    Client().flush()
    print("Trace submission finished. Check LangSmith.")