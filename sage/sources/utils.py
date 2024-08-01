# Source Utility Functions
from datetime import datetime
from operator import itemgetter
from typing import Sequence

import chainlit as cl
from chainlit.user_session import UserSession
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.document import Document
from langchain.schema.runnable import RunnableLambda

from sage.constants import SENTINEL_PATH, logger
from sage.utils.sources import Source


def format_docs(docs: Sequence[Document]) -> str:
    """Format the output of the retriever by including html tags"""
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def get_git_source(metadata: dict) -> str:
    """Format and extract out the git source from the source metadata"""
    try:
        url_without_git: str = metadata["url"].removesuffix(".git")
        source = f"{url_without_git}/-/blob/{metadata['branch']}/{metadata['source']}"
    except KeyError:
        return metadata["source"]
    return source


def format_sources(docs: Sequence[Document]) -> list:
    """Helper for formatting sources. Used in citiation display"""
    formatted_sources = []
    for i, doc in enumerate(docs):
        if "url" in doc.metadata.keys() and ".git" in doc.metadata["url"]:
            source = get_git_source(doc.metadata)
        else:
            source = doc.metadata["source"]
        content = (
            doc.metadata.get("title", None)
            or doc.page_content.strip().rsplit(".")[0] + " ..."
        )
        metadata = {"id": i, "source": source, "content": content}
        formatted_sources.append(metadata)
    return formatted_sources


def get_time_of_day_greeting() -> str:
    """Helper to get a greeting based on the current time of day."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 21:
        return "Good evening"
    else:
        return "Hello"


async def check_for_data_updates() -> bool:
    """Check the data loader for any update"""
    if await SENTINEL_PATH.exists():
        content = await SENTINEL_PATH.read_text()
        if content == "updated":
            logger.info("Data update detected, reloading the retriever database")
            await SENTINEL_PATH.write_text("")
            return True
    return False


async def get_retriever(source_hash: str = "none"):
    """Loads a retrieval model from the source engine"""
    if source_hash == "none":
        return await Source().load(source_hash)

    loading_msg = cl.Message(
        content="Please bear with me for a moment. I'm preparing the data source - might take some time depending on the size of the source..."
    )
    await loading_msg.send()
    await cl.sleep(1)

    await check_for_data_updates()
    retriever = await Source().load(source_hash)

    loading_msg.content = "All set! You're good to go - start by entering your query."
    await loading_msg.update()

    return retriever


def generate_ui_actions(
    metadata: dict, action_name: str = "source_actions"
) -> list[cl.Action]:
    """Generate a list of actions representing the available sources or crews"""
    return [
        cl.Action(
            name=action_name,
            value=data_key,
            label=data_value if action_name == "source_actions" else data_key,
        )
        for data_key, data_value in metadata.items()
    ]


def get_memory(mode: str, user_session: UserSession) -> ConversationBufferWindowMemory:
    """Returns a memory instance for the runnable instance"""
    if mode == "chat":
        memory = user_session.get("memory")
        if memory is None:
            memory = ConversationBufferWindowMemory()
            user_session.set("memory", memory)
        return memory
    return ConversationBufferWindowMemory()


def load_chat_history(mode: str, user_session: UserSession):
    """Load chat history based on the mode and user session"""
    memory = get_memory(mode, user_session)
    return RunnableLambda(memory.load_memory_variables).with_config(
        run_name="ChatHistory"
    ) | itemgetter("history")
