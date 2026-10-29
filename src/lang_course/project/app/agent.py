

from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable

from app.config import get_settings

# === Agent State ===

class AgentState(TypedDict):
    """
    State for the production agent.
    Uses Annotated with add_messages reducer for message accumulation
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used:int

class productionAgent:
    """
    production LanGraph agent with:
    - Retry on failure (model fallback)
    - Graceful error handling
    - Langsmith tracing 
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatOpenAI(
            model = settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0, # We handle retries ourselves
        )

        self.fallback_llm = ChatOpenAI(
            model = settings.fallback_model,
            temperature= 0,
            timeout= 30,
            max_retries= 0,

        )
        self.max_retries = settings.max_retries
        self.graph = self.__build_graph()

    def _build_graph(self):
        """Build the Langraph state machine."""

        def process_message(state:AgentState) -> dict:
            """Try to process the message with the primary model."""

            try:
                response = self.primary_llm.invoke(state["message"])
                return{
                    "message": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return{
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary model."""

            try:
                response = self.fallback_llm.invoke(state["messages"])
                return{
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }

            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",

                }

        def handle_error(state: AgentState) -> dict:
            """Return a graceful error message."""

            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, I'm having trouble processing your request"
                        "right now. please try again in a moment."
                    ))
                ], "model_used": "error_handler",
            }

        def route_after_process(state:AgentState) -> str:
            """Decide what to do after primary model attempt."""

            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"

            else:
                return "error"

        def route_after_fallback(state:AgentState) -> str:
            """Decide what to do after fallback attempt"""
            if state.get("error") is None:
                return "done"

            else:
                return "error"

        #Build the graph

        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},

        )
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )

        graph.add_edge("error", END)

        return graph.compile()
    