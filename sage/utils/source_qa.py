# source_qa.py

from functools import cached_property
from operator import itemgetter
from typing import List, Tuple, Sequence
from datetime import datetime
import asyncio

from langchain.schema.document import Document
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.runnable import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableConfig,
    RunnableMap,
    RunnableLambda,
    RunnableSerializable,
)
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
import chainlit as cl
from pydantic import BaseModel

from utils.sources import Source
from utils.exceptions import SourceException
from constants import LLM_MODEL, assets_dir


class SourceQAService:
    """
    A module designed to interact with local and external sources defined in the config file.
    Can operate in chat and in tool mode
    """

    ai_assistant_name: str = "Sage"

    _retriever: VectorStoreRetriever = None

    _runnable: RunnableSerializable = None

    tool_name: str = "multi_source_inquiry"
    description_pre: str = (
        "A comprehensive question-answering tool backed by local and external data sources. "
        "Designed to provide detailed and verified information for answering complex questions that require insights from data sources "
        "or evidence-backed explanations across a diverse range of topics. Use me when you have no answer to the question"
        "When to use: "
        "- You need to answer questions from external documents "
        "- The question could be linked to data from specific, known sources "
        "Input: A clear and concise question. "
        "Capabilities: "
        "- Retrieves and synthesizes information from relevant sources to construct accurate and relevant answers. "
        "- Provides citations from the data sources when applicable. "
        "Example input: "
        "- 'How many team members are in the Xeres Design Team?' "
        "- 'How are the current configurations for the platform test environments?' "
    )

    condensed_template: str = """
    Given a conversation and a follow-up inquiry, determine whether the inquiry is a continuation of the existing conversation or a new, standalone question. 
    If it is a continuation, use the conversation history encapsulated in the "chat_history" to rephrase the follow up question to be a standalone question, in its original language.
    If the inquiry is new or unrelated, recognize it as such and provide a standalone question without consdering the "chat_history".

    PLEASE don't overdo it and return ONLY the standalone question

    <chat_history>
    {chat_history}
    <chat_history/>

    Follow-Up Inquiry: {question}
    Standalone question::
    """

    qa_template: str = """
    As an AI assistant named Sage, your mandate is to provide accurate and impartial answers to questions while engaging in normal conversation.
    You must differentiate between questions that require answers and standard user chat conversations.

    In standard conversation, especially when discussing your own nature as an AI, footnotes or sources are not required, as the information is based on your programmed capabilities and functions.

    Your responses should adhere to a journalistic style, characterized by neutrality and reliance on factual, verifiable information.

    When formulating answers, you are to:

    - Be creative when applicable.
    - Don't assume you know the meaning of abbreviations unless you have explict context about the abbreviation.
    - Integrate information from the 'context' into a coherent response, avoiding assumptions without clear context.
    - Avoid redundancy and repetition, ensuring each response adds substantive value.
    - Maintain an unbiased tone, presenting facts without personal opinions or biases.
    - Use Sage's internal knowledge to provide accurate responses when appropriate, clearly stating when doing so.
    - When the context does not contain relevant information to answer a specific question, and the question pertains to general knowledge, use Sage's built-in knowledge.
    - Make use of bullet points to aid readability if helpful. Each bullet point should present a piece of information WITHOUT in-line citations.
    - Provide a clear response when unable to answer
    - Avoid adding any sources in the footnotes when the response does not reference specific context.
    - Citations must not be inserted anywhere in the answer only listed in a 'Footnotes' section at the end of the response.

    <context>
    {context}
    </context>

    Question: {question}

    REMEMBER: No in-line citations are allowed, and there should be no citation repetition. Clearly state the source of in the 'Footnotes' section or Sage's internal knowledge base.
    For standard conversation and questions about Sage's nature, no footnotes are required. Include footnotes only when they are directly relevant to the provided answer.

    Footnotes:
    [1] - Brief summary of the first source. (Less than 10 words)
    [2] - Brief summary of the second source.
    ...continue for additional sources, only if relevant and necessary.  
    """

    def __init__(self, mode: str = "tool") -> None:
        self._mode = mode

    @staticmethod
    def _get_time_of_day_greeting():
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

    @staticmethod
    def _format_docs(docs: Sequence[Document]) -> str:
        """Format the output of the retriever by inluding html tags"""
        formatted_docs = []
        for i, doc in enumerate(docs):
            doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
            formatted_docs.append(doc_string)
        return "\n".join(formatted_docs)

    @staticmethod
    def _get_git_source(metadata: dict) -> str:
        """Formate and extract out the git source from the source metadata"""
        try:
            url_without_git: str = metadata["url"].removesuffix(".git")
            source = (
                f"{url_without_git}/-/blob/{metadata['branch']}/{metadata['source']}"
            )
        except KeyError:
            return metadata["source"]
        return source

    def _generate_welcome_message(self):
        """Generate and format an introduction message."""
        greeting = self._get_time_of_day_greeting()
        sources = Source().sources_to_string()

        message = (
            f"{greeting} and welcome!\n"
            "I am Sage, your AI assistant, here to support you with information and insights. How may I assist you today?\n\n"
            "I can provide you with data and updates from a variety of sources including:\n"
            f"  {sources}\n\n"
            "To get started, simply type your query below or ask for help to see what I can do. Looking forward to helping you!"
        )
        return message.strip()

    @property
    def tool_description(self):
        """Generate a description for the source qa tool"""
        source_description = Source().sources_to_string().replace("\n  ", " ")
        description = f"{self.description_pre} Access to data and updates from sources including: {source_description}"
        return description

    def _format_sources(self, docs: Sequence[Document]):
        """Helper for formating sources. Used in citiation display"""
        formatted_sources = []
        for i, doc in enumerate(docs):
            if "url" in doc.metadata.keys() and ".git" in doc.metadata["url"]:
                source = self._get_git_source(doc.metadata)
            else:
                source = doc.metadata["source"]
            content = (
                doc.metadata.get("title", None)
                or doc.page_content.strip().rsplit(".")[0] + " ..."
            )
            metadata = {"id": i, "source": source, "content": content}
            formatted_sources.append(metadata)
        return formatted_sources

    async def _aget_retriever(self):
        """Loads a retrieval model from the source engine"""
        if not self._retriever:
            self._retriever = await Source().aload()
        return self._retriever

    def _get_retriever(self):
        """Loads a retrieval model from the source engine"""
        if not self._retriever:
            self._retriever = Source().load()
        return self._retriever

    @cached_property
    def _chat_memory(self):
        """Keeps track of chat conversation in buffer memory"""
        return ConversationBufferWindowMemory()

    @property
    def mode(self):
        """Gets the current configured mode"""
        if self._mode.lower() not in ["chat", "tool"]:
            raise ValueError(
                f"{self._mode} is not supported. Supported modes are: chat and tool"
            )
        return self._mode

    def _setup_runnable(self, retriever: VectorStoreRetriever):
        """Setups the runnable model"""

        if self.mode == "chat":
            memory: ConversationBufferWindowMemory = cl.user_session.get("memory")
        else:
            memory = self._chat_memory

        condense_question_prompt = PromptTemplate.from_template(self.condensed_template)

        qa_prompt = ChatPromptTemplate.from_template(self.qa_template)

        # define the inputs runnable to generate a standalone question from history
        _inputs = RunnableMap(
            standalone_question=RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter("history")
            )
            | condense_question_prompt
            | LLM_MODEL
            | StrOutputParser()
        )

        # retrieve the documents
        retrieved_docs = RunnableMap(
            docs=itemgetter("standalone_question") | retriever,
            question=itemgetter("standalone_question"),
        )

        # construct the inputs
        _context = {
            "context": lambda x: self._format_docs(x["docs"]),
            "question": lambda x: x["question"],
        }

        # construct the question and answer model
        qa_answer = RunnableMap(
            answer=_context | qa_prompt | LLM_MODEL | StrOutputParser(),
            sources=lambda x: self._format_sources(x["docs"]),
        )

        # create the complete chain
        _runnable = _inputs | retrieved_docs | qa_answer

        if self.mode == "chat":
            cl.user_session.set("runnable", _runnable)
        else:
            self._runnable = _runnable

    async def asetup_runnable(self):
        """Setup the runnable model for the chat"""
        retriever = await self._aget_retriever()
        if not retriever:
            raise SourceException("No source retriever found")

        self._setup_runnable(retriever)

    def setup_runnable(self):
        """Setup the runnable model for the chat"""
        retriever = self._get_retriever()
        if not retriever:
            raise SourceException("No source retriever found")

        self._setup_runnable(retriever)

    @cl.on_chat_start
    async def on_chat_start(self):
        """Initialize a new chat environment"""
        if self.mode == "tool":
            raise ValueError("Tool mode is not supported here")
        cl.user_session.set("memory", self._chat_memory)

        await cl.Avatar(
            name=self.ai_assistant_name, path=str(assets_dir / "ai-assistant.png")
        ).send()

        await cl.Avatar(name="User", path=str(assets_dir / "boy.png")).send()

        await cl.Message(content=self._generate_welcome_message()).send()

        await self.asetup_runnable()

    @cl.on_message
    async def on_message(self, message: cl.Message):
        """Function to react user's message request"""

        if self.mode == "tool":
            raise ValueError("Tool mode is not supported here")

        runnable: RunnableSequence = cl.user_session.get("runnable")
        memory: ConversationBufferWindowMemory = cl.user_session.get("memory")

        msg = cl.Message(content="")

        query = {"question": message.content}
        _sources = None
        _answer = None
        text_elements = []  # type: List[cl.Text]

        async for chunk in runnable.astream(
            query, config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
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
                        url=source_doc["source"],
                        name=source_name,
                    )
                )

        msg.elements = text_elements
        await msg.send()

        # Save memory
        memory.chat_memory.add_ai_message(msg.content)
        memory.chat_memory.add_user_message(message.content)

    async def _arun(self, query: str) -> str:
        """Answer the question in the query"""
        if not self._runnable:
            await self.asetup_runnable()
        return self._runnable.ainvoke({"question": query}).get("answer")

    def _run(self, query: str) -> str:
        """Answer the question in the query"""
        if not self._runnable:
            self.setup_runnable()
        return self._runnable.invoke({"question": query}).get("answer")

    def setup_tool(self) -> Tool:
        """Create a tool object"""

        _tool = Tool(
            name=self.tool_name,
            description=self.tool_description,
            func=self._run,
            coroutine=self._arun,
        )
        return _tool
