# chat.py
# TODO: Address issue with in the agent mode not able to run mutliple actions
# What is the weather in stockholm, malmo and lagos nigeria?
"""
The current weather conditions are as follows:

    Stockholm: Clear skies with temperatures around -10°C (14°F), feeling like -16°C (3°F) with no precipitation expected in the next 90 minutes and winds at 5 m/s.
    Unfortunately, I do not have the current weather conditions for Malmö and Lagos, Nigeria, as my capabilities to fetch real-time data are limited without further tool usage.

"""
import chainlit as cl

from utils.source_qa import SourceQAService
from utils.ai_tools import load_duck_search, load_jira_tools

sage_tools = load_duck_search() + load_jira_tools()
qa_chat = SourceQAService(mode="chat", tools=sage_tools)


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat environment"""
    await qa_chat.on_chat_start()


@cl.on_message
async def on_message(message: cl.Message):
    """Function to react user's message request"""
    await qa_chat.on_message(message)
