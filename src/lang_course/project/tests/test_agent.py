
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from app.agent import productionAgent


def create_agent():
    """Create an agent with mocked primary and fallback models."""

    settings = MagicMock()
    settings.primary_model = "primary-test-model"
    settings.fallback_model = "fallback-test-model"
    settings.google_api_key = "test-api-key"
    settings.max_retries = 1

    primary_llm = MagicMock()
    fallback_llm = MagicMock()

    with patch("app.agent.get_settings", return_value=settings):
        with patch(
            "app.agent.ChatGoogleGenerativeAI",
            side_effect=[primary_llm, fallback_llm],
        ):
            agent = productionAgent()

    return agent, primary_llm, fallback_llm


def test_primary_model_success():
    """The agent should return the primary model's response."""

    agent, primary_llm, fallback_llm = create_agent()

    primary_llm.invoke.return_value = AIMessage(
        content="Response from primary model"
    )

    result = agent.invoke("What is RAG?")

    assert result["response"] == "Response from primary model"
    assert result["model_used"] == "primary"
    assert result["error"] is None

    primary_llm.invoke.assert_called_once()
    fallback_llm.invoke.assert_not_called()


def test_fallback_model_success():
    """The fallback model should run if the primary model fails."""

    agent, primary_llm, fallback_llm = create_agent()

    primary_llm.invoke.side_effect = Exception("Primary model failed")

    fallback_llm.invoke.return_value = AIMessage(
        content="Response from fallback model"
    )

    result = agent.invoke("What is RAG?")

    assert result["response"] == "Response from fallback model"
    assert result["model_used"] == "fallback"
    assert result["error"] is None

    primary_llm.invoke.assert_called_once()
    fallback_llm.invoke.assert_called_once()


def test_both_models_fail():
    """The agent should return a graceful message if both models fail."""

    agent, primary_llm, fallback_llm = create_agent()

    primary_llm.invoke.side_effect = Exception("Primary model failed")
    fallback_llm.invoke.side_effect = Exception("Fallback model failed")

    result = agent.invoke("What is RAG?")

    assert "trouble processing your request" in result["response"]
    assert result["model_used"] == "error_handler"
    assert result["error"] == "Fallback model failed"

    primary_llm.invoke.assert_called_once()
    fallback_llm.invoke.assert_called_once()


def test_agent_receives_user_message():
    """The agent should pass the user's message to the primary model."""

    agent, primary_llm, _ = create_agent()

    primary_llm.invoke.return_value = AIMessage(
        content="Hello!"
    )

    result = agent.invoke("Hello")

    assert result["response"] == "Hello!"

    messages = primary_llm.invoke.call_args.args[0]

    assert messages[0].content == "Hello"
