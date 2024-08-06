from operator import itemgetter

from chainlit.user_session import UserSession, user_session
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableMap,
    RunnableSequence,
)
from langchain.schema.vectorstore import VectorStoreRetriever

from sage.agent.crew import CrewAIRunnable
from sage.constants import LLM_MODEL, agents_crew
from sage.models.chat_prompt import ChatPrompt
from sage.sources.utils import (
    format_docs,
    format_sources,
    get_retriever,
    load_chat_history,
)
from sage.utils.exceptions import SourceException


class RunnableBase:
    def __init__(self, mode: str = "tool", user_session: UserSession = user_session):
        if mode.lower() not in ["chat", "tool"]:
            raise ValueError(
                f"{mode} is not supported. Supported modes are: chat and tool"
            )
        self.mode = mode
        self._runnable = None
        self._user_session = user_session

    def create_crew_runnable(self) -> dict[str, RunnableLambda]:
        """Creates a CrewAI runnable instance that can be used"""
        return CrewAIRunnable(crews=agents_crew).runnable()

    def _create_chat_runnable(
        self, _inputs, _retrieved_docs, _context
    ) -> RunnableSequence:
        """Implementation for creating chat runnable"""
        qa_prompt = ChatPrompt().qa_prompt

        # construct the question and answer model
        qa_answer = RunnableMap(
            answer=_context | qa_prompt | LLM_MODEL | StrOutputParser(),
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
            return (
                x.get("question")
                if isinstance(retriever, RunnableLambda) or not x.get("chat_history")
                else _standalone_chain
            )

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
        _standalone_chain = ChatPrompt().condense_prompt | LLM_MODEL | StrOutputParser()

        _inputs = RunnableMap(
            standalone={
                "question": lambda x: x["question"],
                "chat_history": chat_history_loader,
            }
            | RunnableLambda(standalone_chain_router).with_config(run_name="Condenser")
        )

        # retrieve the documents
        _retrieved_docs = RunnableMap(
            docs=itemgetter("standalone") | retriever,
            question=itemgetter("standalone"),
        ).with_config(run_name="Source Retriever")

        # rconstruct the context inputs
        _context = RunnableMap(
            context=lambda x: format_docs(x["docs"]),
            chat_history=chat_history_loader,
            question=itemgetter("question"),
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
