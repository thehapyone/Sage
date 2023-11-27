import asyncio
from functools import lru_cache

from langchain.chains import RetrievalQA
import chainlit as cl

from utils.sources import Source
from constants import LLM_MODEL

sage_source = Source()
_chain = None


async def get_retriever():
    """Loads a retrieval model form the sources"""
    return await sage_source.load()


async def get_chain():
    global _chain
    if _chain is None:
        retriever = await get_retriever()
        if not retriever:
            raise Exception("No retriever found")
        _chain = RetrievalQA.from_chain_type(
            llm=LLM_MODEL, retriever=retriever)
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
    chain: RetrievalQA = cl.user_session.get("chain")

    msg = cl.Message(content="")

    msg.content = await run_query(message.content, chain)

    await msg.send()
