import asyncio
from functools import lru_cache
from operator import itemgetter
from typing import List, Tuple

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence, RunnableConfig, RunnableMap, RunnableLambda
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import format_document
from langchain.memory import ConversationBufferMemory
import chainlit as cl

from utils.sources import Source
from constants import LLM_MODEL

_retriever = None

condensed_template = """
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

qa_template = """
As a knowledgeable AI assistant named Sage, your role involves providing responses to various inquiries.
Your answers should be unbiased, adopting a journalistic style. It's essential to synthesize search results into a single, cohesive response, avoiding any repetition of information.
If there is no contextually relevant information available to answer a specific question, simply respond with, "I apologize, but I am unable to provide an answer to this question."
This response should only be used in the event of an unanswered question and not as a response to a general conversation starter.

The "context" HTML blocks below contain information retrieved from a knowledge bank and are not part of the user's conversation.

<context>
{context}
<context/>

Question: {question}
"""

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


async def get_retriever():
    """Loads a retrieval model form the sources"""
    global _retriever
    if not _retriever:
        _retriever = await Source().load()
    return _retriever


async def setup_runnable():
    retriever = await get_retriever()
    if not retriever:
        raise Exception("No retriever found")

    # Get the memory
    memory: ConversationBufferMemory = cl.user_session.get("memory")

    condense_question_prompt = PromptTemplate.from_template(
        condensed_template)

    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    # define the inputs runnable to generate a standalone question from history
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
        | condense_question_prompt
        | LLM_MODEL
        | StrOutputParser()
    )

    # retrieve the documents
    retrieved_docs = RunnableMap(
        docs=itemgetter("standalone_question") | retriever,
        question=itemgetter("standalone_question")
    )

    # construct the inputs
    _context = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": lambda x: x["question"],
    }

    qa_answer = RunnableMap(
        answer=_context | qa_prompt | LLM_MODEL | StrOutputParser(),
        sources=itemgetter("docs")
    )

    _runnable = (
        _inputs | retrieved_docs | qa_answer
    )

    cl.user_session.set("runnable", _runnable)


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat environment"""
    memory = ConversationBufferMemory()
    cl.user_session.set("memory", memory)

    await setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    """Function to react user's message request"""
    runnable: RunnableSequence = cl.user_session.get("runnable")
    memory: ConversationBufferMemory = cl.user_session.get("memory")

    msg = cl.Message(content="")

    query = {
        "question": message.content
    }
    _sources = None
    _answer = None
    text_elements = []  # type: List[cl.Text]

    async for chunk in runnable.astream(
            query,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])):
        _answer = chunk.get("answer")
        if _answer:
            await msg.stream_token(_answer)

        if chunk.get("sources"):
            _sources = chunk.get("sources")

    await msg.send()

    # process sources
    if _sources:
        answer = ""
        for source_idx, source_doc in enumerate(_sources):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    
        await cl.Message(content=answer, elements=text_elements).send()

    # Save memory
    memory.chat_memory.add_ai_message(msg.content)
    memory.chat_memory.add_user_message(message.content)
    # cl.user_session.set("chat_history", chat_history)
