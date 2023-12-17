from operator import itemgetter
from typing import List, Tuple, Sequence
from datetime import datetime

from langchain.schema.document import Document
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence, RunnableConfig, RunnableMap, RunnableLambda
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
import chainlit as cl

from utils.sources import Source
from constants import LLM_MODEL, assets_dir

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
As an AI assistant named Sage, your mandate is to provide accurate and impartial answers to a variety of questions and also participate in normal conversation. You should be able to differentie between a question that needs answer to and standard user chat conversation.

Your responses should be in line with a journalistic style, which is characterized by neutrality and reliance on factual, verifiable information.

When formulating answers, you are to:

- Be creative when applicable
- Integrate information from search results into a single, coherent response.
- Avoid redundancy and repetition, ensuring that each piece of information adds substantive value.
- Maintain an unbiased tone throughout, focusing on presenting facts without personal opinions or biases.
- If the provided 'context' does not contain relevant information to answer a specific question, and the question pertains to general knowledge that Sage is capable of answering, then Sage should use its own database to provide an accurate response.  
- If a question can be answered using Sage's internal knowledge, provide an answer accordingly and add a note that the response is based on Sage's own data.  

The 'context' HTML blocks below contain information derived from a knowledge bank, which is separate from the user's direct conversation.
This information serves as the basis for your answers. However, Sage should not rely solely on this context and is expected to use its built-in knowledge when appropriate.  

In the event that no contextually relevant information is available to answer a specific question, your response should be:
"I apologize, but I am unable to provide an answer to this question."
This response is to be used exclusively when an answer cannot be constructed and should not be employed as a generic reply to open-ended conversation starters.

- Citations must not be inserted anywhere in the answer only listed in a 'Footnotes' section at the end of the response, with each source numbered as it appears in the context blocks.
- Select only the most pertinent search results for citation, directly relating to the answer provided.
- When different results refer to different entities sharing the same name, present separate answers to clearly distinguish between these entities.
- Keep the footnotes brief and the summaries MUST be lesss than 10 words always.

In your responses, make use of bullet points to aid readability if helpful. 
Each bullet point should present a piece of information WITHOUT in-line citations. Instead, compile the citations in the 'Footnotes' section at the end of the response.

<context>
{context}
</context>

Question: {question}

REMEMBER: No in-line citations is allowed and no citations repetition
If the answer is based on Sage's internal knowledge, state:  
"This response is based on Sage's internal knowledge base." NOTE: Only say that when applicable.

Footnotes:
[1] - Brief summary of the first source.
[2] - Brief summary of the second source.
...and so on for additional sources.
"""


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def generate_git_source(metadata: dict) -> str:
    """Formate and extract out the git source"""
    try:
        url_without_git: str = metadata["url"].removesuffix(".git")
        source = f"{url_without_git}/-/blob/{metadata['branch']}/{metadata['source']}"
    except KeyError:
        return metadata['source']
    return source


def format_docs(docs: Sequence[Document]):
    """Helper for formating documents"""
    formatted_sources = []
    for i, doc in enumerate(docs):
        if "url" in doc.metadata.keys() and ".git" in doc.metadata["url"]:
            source = generate_git_source(doc.metadata)
        else:
            source = doc.metadata["source"]
        content = doc.metadata.get(
            "title", None) or doc.page_content.strip().rsplit(".")[0] + " ..."
        metadata = {
            "id": i,
            "source": source,
            "content": content
        }
        formatted_sources.append(metadata)
    return formatted_sources


def get_time_of_day_greeting():  
    """Helper to get a greeting based on the current time of day."""  
    current_hour = datetime.now().hour  
    if 5 <= current_hour < 12:  
        return 'Good morning'  
    elif 12 <= current_hour < 17:  
        return 'Good afternoon'  
    elif 17 <= current_hour < 21:  
        return 'Good evening'  
    else:  
        return 'Hello'  

def generate_welcome_message():  
    """Generate and format an introduction message."""  
    greeting = get_time_of_day_greeting()
    sources = Source().sources_to_string()

    message = (f"{greeting} and welcome!\n"
               "I am Sage, your AI assistant, here to support you with information and insights. How may I assist you today?\n\n"
               "I can provide you with data and updates from a variety of sources including:\n"
               f"  {sources}\n\n"
               "To get started, simply type your query below or ask for help to see what I can do. Looking forward to helping you!")
    return message.strip()

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
    memory: ConversationBufferWindowMemory = cl.user_session.get("memory")

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
        "context": lambda x: format_docs(x["docs"]),
        "question": lambda x: x["question"],
    }

    qa_answer = RunnableMap(
        answer=_context | qa_prompt | LLM_MODEL | StrOutputParser(),
        sources=lambda x: format_docs(x["docs"])
    )

    _runnable = (
        _inputs | retrieved_docs | qa_answer
    )

    cl.user_session.set("runnable", _runnable)


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat environment"""
    memory = ConversationBufferWindowMemory()
    cl.user_session.set("memory", memory)
    
    await cl.Avatar(
        name="Sage",
        path=str(assets_dir / "ai-assistant.png")
    ).send()
    
    await cl.Avatar(
        name="User",
        path=str(assets_dir / "boy.png")
    ).send()
    
    await cl.Message(
        content=generate_welcome_message()).send()

    await setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    """Function to react user's message request"""
    runnable: RunnableSequence = cl.user_session.get("runnable")
    memory: ConversationBufferWindowMemory = cl.user_session.get("memory")

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

    # process sources
    if _sources:
        for source_doc in _sources:
            source_name = f"[{source_doc['id']}]"
            source_content = source_doc['content'] + \
                "\n" + source_doc['source']
            text_elements.append(
                cl.Text(content=source_content,
                        url=source_doc['source'], name=source_name)
            )

    msg.elements = text_elements
    await msg.send()

    # Save memory
    memory.chat_memory.add_ai_message(msg.content)
    memory.chat_memory.add_user_message(message.content)
