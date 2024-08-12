# sage.sources.qa.py

from typing import List, Tuple

import chainlit as cl
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.runnable import (
    RunnableConfig,
    RunnableSequence,
    RunnableSerializable,
)
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.tools import Tool

from sage.constants import (
    LLM_MODEL,
    chat_starters,
    logger,
)
from sage.models.chat_prompt import ChatPrompt
from sage.sources.mode_handlers import ChatModeHandlers
from sage.sources.runnable import RunnableBase
from sage.sources.sources import Source
from sage.sources.utils import get_memory, get_time_of_day_greeting


class SourceQAService:
    """
    A module designed to interact with local and external sources defined in the config file.
    Can operate in chat and in tool mode
    """

    _retriever: VectorStoreRetriever = None

    _runnable: RunnableSerializable = None

    def __init__(self, mode: str = "tool", tools: List[Tool] = []) -> None:
        self.tools = tools
        self._runnable_handlers = RunnableBase(LLM_MODEL, mode)
        self._mode_handlers = ChatModeHandlers(self._runnable_handlers)
        self.mode = self._runnable_handlers.mode

    @property
    def _chat_memory(self):
        """Keeps track of chat conversation in buffer memory"""
        return ConversationBufferWindowMemory()

    def get_intro_message(self, profile: str):
        return ChatPrompt().generate_welcome_message(
            greeting=get_time_of_day_greeting(),
            source_repr=Source().sources_to_string(),
            profile=profile,
        )

    @cl.set_chat_profiles
    async def chat_profile():
        return [
            cl.ChatProfile(
                name="Chat Only",
                markdown_description="Run Sage in Chat only mode and interact with provided sources",
                icon="https://picsum.photos/200",
                default=False,
                starters=[
                    cl.Starter(
                        label="Home - Get Started",
                        message="/home",
                        icon="/public/avatars/home.png",
                    ),
                    *chat_starters,
                ],
            ),
            cl.ChatProfile(
                default=True,
                name="Agent Mode",
                markdown_description="Sage runs as an AI Agent with access to external tools and data sources.",
                icon="https://picsum.photos/250",
            ),
            cl.ChatProfile(
                name="File Mode",
                markdown_description="Ask Sage questions about files. No data sources just the file as the only data source",
                icon="https://picsum.photos/260",
            ),
        ]

    @cl.on_chat_start
    async def on_chat_start(self):
        """Initialize a new chat environment based on the selected profile."""
        if self.mode == "tool":
            raise ValueError("Tool mode is not supported here")

        chat_profile = cl.user_session.get("chat_profile")

        # Initialize memory for the new chat
        _ = get_memory(self.mode, cl.user_session)

        # Chat Only mode will be configured via starters instead
        if chat_profile == "Chat Only":
            return

        intro_message = self.get_intro_message(chat_profile)

        runnable = None

        if chat_profile == "Agent Mode":
            retriever, runnable = await self._mode_handlers.handle_agent_only_mode(
                intro_message
            )

        elif chat_profile == "File Mode":
            retriever = await self._mode_handlers.handle_file_mode(intro_message)
        else:
            retriever = await self._mode_handlers.handle_default_mode(intro_message)

        self._runnable_handlers.setup_runnable(retriever=retriever, runnable=runnable)

    @staticmethod
    def _get_starter_source_label(message: str) -> Tuple[str, str]:
        """Helper to extract the source label from the starter message"""
        if not message.endswith("%"):
            logger.warning("Can not extract label from starter message")
            return message, "none"
        try:
            main_message, label, _ = message.rsplit("%", 2)
            return main_message.strip(), label.strip()
        except ValueError as error:
            logger.warning("Error extracting label from starter %s", error)
        return message, "none"

    @cl.on_message
    async def on_message(self, message: cl.Message):
        """Function to react user's message request"""
        if self.mode == "tool":
            raise ValueError("Tool mode is not supported here")

        # Handle starter message only once
        if (
            cl.user_session.get("starter_message", True)
            and cl.user_session.get("chat_profile") == "Chat Only"
        ):
            cl.user_session.set("starter_message", False)
            chat_profile = cl.user_session.get("chat_profile")

            if message.content == "/home":
                retriever = await self._mode_handlers.handle_chat_only_mode(
                    self.get_intro_message(chat_profile), message.id
                )
                self._runnable_handlers.setup_runnable(retriever=retriever)
                return

            message.content, source_label = self._get_starter_source_label(
                message.content
            )
            await message.update()

            # Now we should set the retriever for the other messages
            retriever = await self._mode_handlers.handle_chat_only_mode(
                "", source_label=source_label
            )
            self._runnable_handlers.setup_runnable(retriever=retriever)

        runnable: RunnableSequence = cl.user_session.get("runnable")
        memory = get_memory(self.mode, cl.user_session)

        msg = cl.Message(content="")

        query = {"question": message.content}
        _sources = None
        _answer = None
        text_elements: list[cl.Text] = []

        run_name = getattr(runnable, "config", {}).get("run_name", "")

        async for chunk in runnable.astream(
            query,
            config=RunnableConfig(
                metadata={"run_name": run_name},
                run_name=run_name,
                callbacks=[
                    cl.AsyncLangchainCallbackHandler(
                        stream_final_answer=True,
                        to_ignore=[
                            "Runnable",
                            "<lambda>",
                            "CustomLiteLLM",
                            "_Exception",
                        ],
                        answer_prefix_tokens=[
                            "<final_answer>",
                            "<tool>",
                            "</tool>",
                            "tool_input",
                        ],
                    )
                ],
            ),
        ):
            _answer = chunk.get("answer")
            if _answer:
                await msg.stream_token(_answer)

            if chunk.get("sources"):
                _sources = chunk.get("sources")

        # process sources
        if _sources:
            for source_doc in _sources:
                source_name = f"[{source_doc['id']}]"
                source_content = source_doc["content"] + "\n" + source_doc["source"]
                text_elements.append(
                    cl.Text(
                        content=source_content,
                        name=source_name,
                        display="side",
                        size="medium",
                    )
                )

        msg.elements = text_elements
        await msg.send()

        # Save memory
        memory.chat_memory.add_ai_message(msg.content)
        memory.chat_memory.add_user_message(message.content)

    async def _run(self, query: str) -> str:
        """Answer the question in the query"""
        if not self._runnable:
            await self._runnable_handlers.asetup_runnable()
        response: dict = await self._runnable.ainvoke({"question": query})
        return response.get("answer")

    def setup_tool(self) -> Tool:
        """Create a tool object"""
        model_prompt = ChatPrompt()
        return Tool(
            name=model_prompt.tool_name,
            description=model_prompt.tool_description(Source().sources_to_string()),
            func=None,
            coroutine=self._run,
        )
