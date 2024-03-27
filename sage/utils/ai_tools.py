# Helps to initialize various tools that can be used by AI agents
from typing import List

from constants import LLM_MODEL as llm
from langchain.tools import Tool

from sage.utils.source_qa import SourceQAService


def load_duck_search() -> List[Tool]:
    """Load a DuckDuckGoSearch tool"""
    from langchain_community.tools import DuckDuckGoSearchRun

    search_description: str = (
        "Useful for searching the Internet using the DuckDuckGo search engine and returns the first result."
        "The input to this tool should be a typical search query"
    )
    return [
        DuckDuckGoSearchRun(name="duckduckgo_search", description=search_description)
    ]


def load_calculator() -> Tool:
    """Loads a Calculator tool"""
    from langchain.chains.llm_math.base import LLMMathChain

    return Tool(
        name="calculator",
        description="Useful for when you need to answer questions about math.",
        func=LLMMathChain.from_llm(llm=llm).run,
        coroutine=LLMMathChain.from_llm(llm=llm).arun,
    )


def load_jira_tools() -> List[Tool]:
    """A toolbox for various Jira issue functionality"""
    from sage.utils.jira_agent import IssueAgent

    summarize_tool = Tool(
        name="summarize_jira_issue",
        description=(
            "Use it when you want to summarize or get a summary of Jira issues. "
            "The input is a jira issue key(e.g., 'PROJECT-1234')"
            "The summary is very elaborate and it's advised to use the summary directly instead of attempting to summarize it"
        ),
        func=IssueAgent().summarize,
        coroutine=IssueAgent().asummarize,
    )

    get_issue_tool = Tool(
        name="get_jira_issue",
        description=(
            "A tool for fetching all details about a Jira issue. The input is a jira issue key"
        ),
        func=IssueAgent().generate_issue_template,
    )

    return [summarize_tool, get_issue_tool]


def load_qa_tool() -> List[Tool]:
    """Loads a QA tool based on the available data sources"""
    return [SourceQAService(mode="tool").setup_tool()]