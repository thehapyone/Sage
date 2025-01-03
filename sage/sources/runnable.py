from operator import itemgetter
from typing import Sequence

from chainlit.user_session import UserSession, user_session
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableMap,
    RunnableSequence,
)
from langchain.schema.vectorstore import VectorStoreRetriever

from sage.agent.crew import CrewAIRunnable
from sage.models.chat_prompt import ChatPrompt
from sage.sources.utils import (
    format_docs,
    format_sources,
    get_retriever,
    load_chat_history,
)
from sage.utils.exceptions import SourceException
from sage.utils.supports import ChatLiteLLM
from sage.validators.crew_ai import CrewConfig


class SearchQueryGeneratorParser(JsonOutputParser):
    """Output parser for the search query to return a list of strings."""

    def parse(self, text: str) -> list[str]:
        """Output is like this - { "queries": [] } """
        result : dict[list[str]] = super().parse(text)
        queries = result["queries"]
        return queries

class RunnableBase:
    def __init__(
        self,
        llm_model: ChatLiteLLM,
        mode: str = "tool",
        user_session: UserSession = user_session,
    ):
        if mode.lower() not in ["chat", "tool"]:
            raise ValueError(
                f"{mode} is not supported. Supported modes are: chat and tool"
            )
        self.mode = mode
        self._runnable = None
        self._user_session = user_session
        self.base_model = llm_model

    def create_crew_runnable(
        self, crews: Sequence[CrewConfig]
    ) -> dict[str, RunnableLambda]:
        """Creates a CrewAI runnable instance that can be used"""
        return CrewAIRunnable(crews=crews).runnable()

    def _create_chat_runnable(
        self, _inputs, _retrieved_docs, _context
    ) -> RunnableSequence:
        """Implementation for creating chat runnable"""

        # construct the question and answer model
        qa_answer = RunnableMap(
            answer=_context
            | RunnableLambda(ChatPrompt().modality_prompt_router).with_config(
                run_name="Modality-Router"
            )
            | self.base_model
            | StrOutputParser(),
            sources=lambda x: format_sources(x["docs"]),
        ).with_config(run_name="Sage Assistant")

        # create the complete chain
        _runnable = _inputs | _retrieved_docs | qa_answer
        return _runnable

    def setup_runnable(
        self,
        retriever: VectorStoreRetriever | None = None,
        runnable: RunnableLambda | None = None,
    ):
        """
        Configures and initializes the runnable model based on the specified retriever and profile.

        """

        def standalone_chain_router(x: dict):
            """Helper for routing to the standalone chain"""
            if isinstance(retriever, RunnableLambda):
                return x.get("question")
            if not x.get("chat_history"):
                return x.get("question")
            return _search_generator_chain

        if runnable:
            (
                self._user_session.set("runnable", runnable)
                if self.mode == "chat"
                else setattr(self, "_runnable", runnable)
            )
            return

        # Loads the chat history
        chat_history_loader = load_chat_history(self.mode, self._user_session)

        # Condense Question Chain
        _standalone_chain = (
            ChatPrompt().condense_prompt | self.base_model | StrOutputParser()
        )

        # Search Query Generator Chain
        _search_generator_chain = (
            ChatPrompt().query_generator_prompt | self.base_model | JsonOutputParser()
        )

        # Constructs the Chain Inputs
        _inputs = RunnableMap(
            question=itemgetter("question"),
            search_queries={
                "question": lambda x: x["question"],
                "chat_history": chat_history_loader,
            }
            | RunnableLambda(standalone_chain_router).with_config(run_name="QueryGenerator"),
            image_data=itemgetter("image_data"),
        )

        # retrieve the documents
        _retrieved_docs = RunnableMap(
            docs=itemgetter("search_queries") | retriever,
            question=itemgetter("question"),
            image_data=itemgetter("image_data"),
        ).with_config(run_name="SourceRetriever")

        # rconstruct the context inputs
        _context = RunnableMap(
            context=lambda x: format_docs(x["docs"]),
            chat_history=chat_history_loader,
            question=itemgetter("question"),
            image_data=itemgetter("image_data"),
        )
        _runnable = self._create_chat_runnable(_inputs, _retrieved_docs, _context)

        (
            self._user_session.set("runnable", _runnable)
            if self.mode == "chat"
            else setattr(self, "_runnable", _runnable)
        )

    async def asetup_runnable(self):
        retriever = await get_retriever()
        if not retriever:
            raise SourceException("No source retriever found")
        self.setup_runnable(retriever=retriever)
