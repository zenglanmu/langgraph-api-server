from deepagents import create_deep_agent
from langgraph_api import GraphRegistry
from .chat_model import get_chat_client

llm = get_chat_client()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def build_graph():
    agent = create_deep_agent(
        model=llm,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )
    return agent

GraphRegistry.registy_lg_graph('agent', build_graph)