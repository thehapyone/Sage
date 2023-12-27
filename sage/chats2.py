from typing import List

from langchain.schema.runnable import RunnableSequence, RunnableConfig
from langchain.memory import ConversationBufferWindowMemory
import chainlit as cl

from utils.source_qa import SourceQAService

qa_chat = SourceQAService(mode="chat")


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat environment"""
    await qa_chat.on_chat_start()


@cl.on_message
async def on_message(message: cl.Message):
    """Function to react user's message request"""
    await qa_chat.on_message(message)
