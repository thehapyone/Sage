# Helps to initialize various tools that can be used by AI agents
from typing import List

from langchain.tools import Tool


def load_duck_search() -> Tool:
    """Load a DuckDuckGoSearch tool"""
    from langchain.tools import DuckDuckGoSearchRun

    search_description = (
        "Use this tool to search the Internet using the DuckDuckGo search engine and returns the first result.",
        "The input to this tool should be a typical search query",
    )

    return DuckDuckGoSearchRun(name="duckduckgo_search", description=search_description)

def get_tools(all: bool = True, tools: List[str] = None):
    """Helper to help load various tools"""
    pass
