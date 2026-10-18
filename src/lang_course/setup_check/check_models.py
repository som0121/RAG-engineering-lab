import os
import sys
from importlib.metadata import version

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from lang_course.common.llm import get_model

sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print emoji otherwise

print(f"langchain-core version: {version('langchain-core')}")
print(f"langgraph version: {version('langgraph')}\n")


def check(label, model, prompt):
    """Invoke one model and print its reply, or the error, so one bad provider doesn't stop the rest."""
    try:
        print(f"[OK]   {label}: {model.invoke(prompt).content}\n")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}\n")
        return False


def main() -> None:
    results = []

    # MODEL from .env (OpenRouter by default)
    results.append(check(
        f"get_model ({os.getenv('MODEL')})",
        get_model(),
        "Say hello and give me one tip for learning LangChain.",
    ))

    # testing openai directly
    if os.getenv("OPENAI_API_KEY"):
        results.append(check(
            "ChatOpenAI (gpt-4o-mini)",
            ChatOpenAI(model="gpt-4o-mini", temperature=0),
            "Say 'setup complete!' in one word",
        ))
    else:
        print("[SKIP] ChatOpenAI: OPENAI_API_KEY is not set\n")

    # testing anthropic directly, or through OpenRouter if there is no Anthropic key
    if os.getenv("ANTHROPIC_API_KEY"):
        label, llm_anthropic = "ChatAnthropic (claude-sonnet-5-5)", ChatAnthropic(model="claude-sonnet-5-5", temperature=0)
    else:
        label = "Claude Sonnet 5.5 via OpenRouter (ANTHROPIC_API_KEY not set)"
        llm_anthropic = get_model("openrouter:anthropic/claude-sonnet-5.5", temperature=0)
    results.append(check(label, llm_anthropic, "Say 'setup complete!' in one word"))

    print(f"Setup complete: {sum(results)}/{len(results)} model checks passed")


if __name__ == "__main__":
    main()
