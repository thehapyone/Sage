from typing import List

from langchain.schema.runnable import RunnableSequence, RunnableConfig
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.llm_math.base import LLMMathChain
from langchain.tools import Tool
import chainlit as cl

from utils.source_qa import SourceQAService
from utils.ai_tools import load_duck_search, load_jira_tools
from constants import LLM_MODEL as model

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
