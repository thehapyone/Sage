import asyncio
from functools import lru_cache

from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence, RunnableConfig
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import chainlit as cl

from utils.sources import Source
from constants import LLM_MODEL

sage_source = Source()
_chain = None

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


async def get_retriever():
    """Loads a retrieval model form the sources"""
    return await sage_source.load()


async def get_chain():
    global _chain
    if _chain is None:
        retriever = await get_retriever()
        if not retriever:
            raise Exception("No retriever found")
        prompt = ChatPromptTemplate.from_template(qa_template)

        _chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | LLM_MODEL
            | StrOutputParser()
        )
    return _chain


async def run_query(query: str, chain: RetrievalQA) -> str:
    return await chain.arun(query)

# async def main():
#     # Vector - Similarity Search
#     # queries = [
#     #     # "What is the Gitlab CI Templates project",
#     #     # "Show an example of how to create a docker image using kaniko for a sample project with container scanning",
#     #     # "What does the basic.gitlab-ci.yml template contains? What does it do? What CI jobs are inluded? and how do you use it?",
#     #     # "What is the DTS or ACT Compliance Framework about? and what does it contain today?",
#     #     # "Is the ITS/DTS Gitlab validated? and if yes, what version was it validated with?",
#     #     # "What is the current version of the gitlab deployed in the Anthea environment?",
#     #     "Who is Ayodeji Ayibiowu? and what sort of impacts has he contributed today?"
#     # ]

#     qa_chain = await get_chain()

#     await asyncio.gather(*[run_query(query, qa_chain) for query in queries])


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat environment"""
    chain = await get_chain()
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    """Function to react user's message request"""
    chain: RunnableSequence = cl.user_session.get("chain")

    msg = cl.Message(content="")
    
    async for chunk in chain.astream(message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])):
        await msg.stream_token(chunk)

    await msg.send()
