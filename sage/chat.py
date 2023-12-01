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

_chain = None

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{history}
Follow Up Input: {question}
Standalone question:"""

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

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


ANSWER_PROMPT = ChatPromptTemplate.from_template(qa_template)


async def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

memory = ConversationBufferMemory(
    return_messages=True)


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
    | CONDENSE_QUESTION_PROMPT
)

# _context = {
#     "context": itemgetter("standalone_question") | retriever | _combine_documents,
#     "question": lambda x: x["standalone_question"],
# }
# conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()

print(_inputs.invoke(
    {
        "question": "where did harrison work?"
    }
))
