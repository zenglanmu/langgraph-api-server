from langgraph_api import GraphRegistry

from .weather import build_graph as weather_agent
from .search import build_graph as search_agent


GraphRegistry.registy_lg_graph('search', search_agent)
GraphRegistry.registy_lg_graph('weather', weather_agent)
