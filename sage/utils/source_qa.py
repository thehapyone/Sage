# sage.utils.source_qa.py

from datetime import datetime
from operator import itemgetter
from typing import List, Sequence, Tuple

import chainlit as cl
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
    RunnableSequence,
    RunnableSerializable,
)
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.tools import Tool

from sage.agent.crew import CrewAIRunnable
from sage.constants import (
    LLM_MODEL,
    SENTINEL_PATH,
    agents_crew,
    chat_starters,
    logger,
    validated_config,
)
from sage.utils.exceptions import AgentsException, SourceException
from sage.utils.sources import Source


async def check_for_data_updates() -> bool:
    """Check the data loader for any update"""
    if await SENTINEL_PATH.exists():
        # Read the sentinel file
        content = await SENTINEL_PATH.read_text()
        if content == "updated":
            logger.info("Data update detected, reloading the retriever database")
            # reset the file
            await SENTINEL_PATH.write_text("")
            return True
    return False


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
        "A tool for providing detailed and verified information for answering questions that require insights from various data sources. "
        "When to use: "
        "- You need answers that could be possibly found in external sources or documents."
        "- The question could be linked to data from specific, known sources. "
        "- You need answers to something outside your own knowledge. "
        "Input: A clear and concise question. "
        "Capabilities: "
        "- Retrieves and synthesizes data from relevant source database to construct answers. "
        "Example input: "
        "- 'How many team members are in the Xeres Design Team?' "
        "- 'What are the current configurations for the platform test environments?' "
    )

    condensed_template: str = """
    Given a conversation and a follow-up inquiry, determine whether the inquiry is a continuation of the existing conversation or a new, standalone question. 
    If it is a continuation, use the conversation history encapsulated in the "chat_history" to rephrase the follow up question to be a standalone question, in its original language.
    If the inquiry is new or unrelated, recognize it as such and provide a standalone question without consdering the "chat_history".

    PLEASE don't overdo it and return ONLY the standalone question.
    
    REMEMBER:
     - The inquiry is not meant for you at all. Don't refer new meanings or distort the original inquiry.
     - Always keep the original language. Not all inquires are questions.
     
    <chat_history>
    {chat_history}
    <chat_history/>

    Follow-Up Inquiry: {question}
    Standalone question::
    """

    qa_template_chat: str = """
    As an AI assistant named Sage, your mandate is to provide accurate and impartial answers to questions while engaging in normal conversation.
    You must differentiate between questions that require answers and standard user chat conversations.
    In standard conversation, especially when discussing your own nature as an AI, footnotes or sources are not required, as the information is based on your programmed capabilities and functions.
    Your responses should adhere to a journalistic style, characterized by neutrality and reliance on factual, verifiable information.
    When formulating answers, you are to:
    - Be creative when applicable.
    - Don't assume you know the meaning of abbreviations unless you have explicit context about the abbreviation.
    - Integrate information from the 'context' into a coherent response, avoiding assumptions without clear context.
    - Avoid redundancy and repetition, ensuring each response adds substantive value.
    - Maintain an unbiased tone, presenting facts without personal opinions or biases.
    - Use Sage's internal knowledge to provide accurate responses when appropriate, clearly stating when doing so.
    - When the context does not contain relevant information to answer a specific question, and the question pertains to general knowledge, use Sage's built-in knowledge.
    - Make use of bullet points to aid readability if helpful. Each bullet point should present a piece of information WITHOUT in-line citations.
    - Provide a clear response when unable to answer
    - Avoid adding any sources in the footnotes when the response does not reference specific context.
    - Citations must not be inserted anywhere in the answer, only listed in a 'Footnotes' section at the end of the response.
    <context>
    {context}
    </context>
    
    Here is the current chat history - use if relevant:
    <chat_history>
    {chat_history}
    <chat_history/>

    Question: {question}

    REMEMBER: No in-line citations and no citation repetition. State sources in the 'Footnotes' section.
    For standard conversation and questions about Sage's nature, no footnotes are required. Include footnotes only when they are directly relevant to the provided answer.
    Footnotes:
    [1] - Brief summary of the first source. (Less than 10 words)
    [2] - Brief summary of the second source.
    ...continue for additional sources, only if relevant and necessary.  
    """

    def __init__(self, mode: str = "tool", tools: List[Tool] = []) -> None:
        self._mode = mode
        self.tools = tools

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
        """Format the output of the retriever by including html tags"""
        formatted_docs = []
        for i, doc in enumerate(docs):
            doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
            formatted_docs.append(doc_string)
        return "\n".join(formatted_docs)

    @staticmethod
    def _get_git_source(metadata: dict) -> str:
        """Format and extract out the git source from the source metadata"""
        try:
            url_without_git: str = metadata["url"].removesuffix(".git")
            source = (
                f"{url_without_git}/-/blob/{metadata['branch']}/{metadata['source']}"
            )
        except KeyError:
            return metadata["source"]
        return source

    def _generate_welcome_message(self, profile: str = "chat"):
        """Generate and format an introduction message."""
        greeting = self._get_time_of_day_greeting()
        sources = Source().sources_to_string()

        if not profile:
            return ""

        if "file" in profile.lower():
            message = (
                f"{greeting} and welcome!\n"
                "I am Sage, your AI assistant, here to support you with information and insights. How may I assist you today?\n\n"
                "I can answer questions about the contents of the files you upload. To get started:\n\n"
                "  1. Upload one or more documents\n"
                "  2. Ask questions about the files and I will try to answer as best as I can\n\n"
                "Supported file types: Word Documents, PDFs, Text files, Excel files, JSON, and YAML files.\n"
                "Looking forward to our conversation!"
            )
        elif "agent" in profile.lower():
            message = (
                f"{greeting} and welcome!\n"
                "I am Sage, your AI assistant, here to help you orchestrate AI agents using the CrewAI framework.\n\n"
                "CrewAI empowers agents to work together seamlessly, tackling complex tasks through collaborative intelligence.\n"
                "**Note**: Each crew behaves based on its configuration, and responses may take some time.\n\n"
                "To get started, choose a crew from the list below. Then, send your message to the agents and wait for them to kickstart their tasks."
            )
        else:
            message = (
                f"{greeting} and welcome!\n"
                "I am Sage, your AI assistant, here to support you with information and insights. How may I assist you today?\n\n"
                "I can provide you with data and updates from a variety of sources including:\n"
                f"  {sources}\n\n"
                "To get started, simply select an option below; then begin typing your query or ask for help to see what I can do."
            )
        return message.strip()

    @property
    def tool_description(self):
        """Generate a description for the source qa tool"""
        source_description = Source().sources_to_string().replace("\n  ", " ")
        description = f"{self.description_pre}\n. I have access to the following sources: {source_description}"
        return description

    def _format_sources(self, docs: Sequence[Document]):
        """Helper for formatting sources. Used in citiation display"""
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

    async def _get_retriever(self, source_hash: str = "none"):
        """Loads a retrieval model from the source engine"""

        if source_hash == "none":
            return await Source().load(source_hash)

        # Inform the user that data loading is about to begin
        loading_msg = cl.Message(
            content="Please bear with me for a moment. I'm preparing the data source - might take some time depending on the size of the source..."
        )
        await loading_msg.send()
        await cl.sleep(1)

        # Check for data update
        await check_for_data_updates()

        retriever = await Source().load(source_hash)

        loading_msg.content = (
            "All set! You're good to go - start by entering your query."
        )
        await loading_msg.update()

        return retriever

    @property
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

    def _handle_error(self, error: Exception) -> str:
        return str(error)

    def _load_chat_history(self, memory: ConversationBufferWindowMemory):
        return RunnableLambda(memory.load_memory_variables).with_config(
            run_name="ChatHistory"
        ) | itemgetter("history")

    def _get_memory(self):
        """Returns a memory instance for the runnable instance"""
        if self.mode == "chat":
            memory: ConversationBufferWindowMemory = cl.user_session.get("memory")
            # TODO: Remove this when the reason for while the memory content is None has been resolved
            if memory is None:
                memory = self._chat_memory
                cl.user_session.set("memory", memory)
        else:
            memory = self._chat_memory
        return memory

    def _create_crew_runnable(self) -> dict[str, RunnableLambda]:
        """Creates a CrewAI runnable instance that can be used"""
        return CrewAIRunnable(crews=agents_crew).runnable()

    def _create_chat_runnable(self, _inputs, _retrieved_docs, _context):
        """Implementation for creating chat runnable"""
        qa_prompt = ChatPromptTemplate.from_template(self.qa_template_chat)
        # construct the question and answer model
        qa_answer = RunnableMap(
            answer=_context | qa_prompt | LLM_MODEL | StrOutputParser(),
            sources=lambda x: self._format_sources(x["docs"]),
        ).with_config(run_name="Sage Assistant")

        # create the complete chain
        _runnable = _inputs | _retrieved_docs | qa_answer
        return _runnable

    def _setup_runnable(
        self,
        retriever: VectorStoreRetriever | None = None,
        runnable: RunnableLambda | None = None,
    ):
        """
        Configures and initializes the runnable model based on the specified retriever and profile.

        This function sets up the question-answering pipeline, which includes condensing questions,
        retrieving documents, and generating answers. The pipeline components are assembled
        based on the operation mode and profile provided.

        In 'chat' mode, conversation history is loaded from the user session's memory.
        For other modes, a default chat memory buffer is used. The function constructs
        various runnable components such as inputs formatting, document retrieval, and answer generation,
        which are then composed into a final runnable sequence.

        Depending on the profile, the answer generation step may use an agent-based approach
        for parsing and responding to questions, or a chat-based prompt if the profile doesn't
        include 'agent'. The constructed runnable model is then stored in the user session or
        the instance's state for execution.

        Parameters:
            retriever: An instance of VectorStoreRetriever used to retrieve documents relevant to the question.
            runnable: An instance of runnable that has already be configured

        Side effects:
            - Sets up a complete question-answering runnable model based on the operation mode and profile.
            - Stores the runnable in the user session or instance state for later execution.

        Raises:
            - Any exceptions that may occur during the construction of the runnable components.
        """

        def standalone_chain_router(x: dict):
            """Helper for routing to the standalone chain"""
            # If retriever is a RunnableLambda or there is no valid chat history, return the question.
            if isinstance(retriever, RunnableLambda) or not x.get("chat_history"):
                return x.get("question")
            # Otherwise, return the standalone chain.
            return _standalone_chain

        if runnable:
            if self.mode == "chat":
                cl.user_session.set("runnable", runnable)
            else:
                self._runnable = runnable
            return

        # Loads the chat history
        chat_history_loader = self._load_chat_history(self._get_memory())

        # Condense Question Chain
        condense_question_prompt = PromptTemplate.from_template(self.condensed_template)
        _standalone_chain = condense_question_prompt | LLM_MODEL | StrOutputParser()

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
        ).with_config(
            run_name="Source Retriever",
        )

        # construct the context inputs
        _context = RunnableMap(
            context=lambda x: self._format_docs(x["docs"]),
            chat_history=chat_history_loader,
            question=itemgetter("question"),
        )
        _runnable = self._create_chat_runnable(_inputs, _retrieved_docs, _context)

        if self.mode == "chat":
            cl.user_session.set("runnable", _runnable)
        else:
            self._runnable = _runnable

    async def asetup_runnable(self):
        """Setup the runnable model for the chat"""
        retriever = await self._get_retriever()
        if not retriever:
            raise SourceException("No source retriever found")

        self._setup_runnable(retriever=retriever)

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

    @staticmethod
    def generate_ui_actions(
        metadata: dict, action_name: str = "source_actions"
    ) -> List[cl.Action]:
        """Generate a list of actions representing the available sources or crews"""
        actions = [
            cl.Action(
                name=action_name,
                value=data_key,
                label=data_value if action_name == "source_actions" else data_key,
            )
            for data_key, data_value in metadata.items()
        ]
        return actions

    async def _handle_file_mode(self, intro_message: str) -> VectorStoreRetriever:
        """Handles initialization for 'File Mode', where users upload files for the chat."""
        files = None
        # Wait for the user to upload a file
        while files is None:
            files = await cl.AskFileMessage(
                content=intro_message,
                accept={
                    "text/plain": [".txt"],
                    "application/pdf": [".pdf"],
                    "application/json": [".json"],
                    "application/x-yaml": [
                        ".yaml",
                        ".yml",
                    ],
                    "application/vnd.ms-excel": [".xls"],
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
                        ".xlsx"
                    ],
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
                        ".docx"
                    ],
                    "application/msword": [".doc"],
                },
                max_size_mb=validated_config.upload.max_size_mb,
                max_files=validated_config.upload.max_files,
                timeout=validated_config.upload.timeout,
            ).send()

        msg = cl.Message(content=f"Now, I will begin processing {len(files)} files ...")
        await msg.send()
        await cl.sleep(1)

        # Get the files retriever
        retriever = await Source().load_files_retriever(files)
        # Let the user know that the system is ready
        file_names = "\n  ".join([file.name for file in files])
        msg.content = (
            "The following files are now processed and ready to be used!\n"
            f"  {file_names}"
        )
        await cl.sleep(1)
        await msg.update()
        return retriever

    async def _handle_chat_only_mode(
        self, intro_message: str, root_id: str = None, source_label: str = None
    ) -> VectorStoreRetriever:
        """Handles initialization for 'Chat Only' mode, where users select a source to chat with."""
        # Get the sources labels that will be used to create the source actions
        sources_metadata = await Source().get_labels_and_hash()

        if source_label:
            hash_key = next(
                (k for k, v in sources_metadata.items() if v == source_label), "none"
            )
            return await self._get_retriever(hash_key)

        await cl.Message(id=root_id, content=intro_message).send()

        source_actions = self.generate_ui_actions(sources_metadata)

        action_response = None

        if source_actions:
            action_response = await cl.AskActionMessage(
                content="To start a conversation, choose a data source. If no selection is made before the time runs out, the default is 🙅‍♂️/🙅‍♀️ No Sources ⛔",
                timeout=300,
                actions=[
                    cl.Action(
                        name="source_actions",
                        value="all",
                        label="👌 All Sources 📚",
                    ),
                    cl.Action(
                        name="source_actions",
                        value="none",
                        label="🙅‍♂️/🙅‍♀️ No Sources ⛔",
                    ),
                    *source_actions,
                ],
            ).send()

        # initialize retriever with the selected source action
        selected_hash = action_response.get("value") if action_response else "none"
        return await self._get_retriever(selected_hash)

    async def _handle_agent_only_mode(
        self, intro_message: str, root_id: str = None, crew_label: str = None
    ) -> tuple[RunnableLambda, RunnableLambda]:
        """
        Handles initialization for 'Agent Only' mode, where users select a crew to chat with.

        Returns a tuple containing a black retriever instance and an optional instance for the crew
        """
        # Get the crew names that will be used to create the source actions
        crews_metadata = self._create_crew_runnable()

        if crew_label:
            crew_instance = crews_metadata.get(crew_label, None)
            if crew_instance:
                return RunnableLambda(lambda x: []), crew_instance
            raise AgentsException(f"The crew {crew_label} can not be found")

        await cl.Message(id=root_id, content=intro_message).send()

        crew_actions = self.generate_ui_actions(crews_metadata, "crew_actions")

        action_response = None

        if crew_actions:
            action_response = await cl.AskActionMessage(
                content="To start, please choose a crew to work with. If no selection is made before the time runs out, the default is 'No Agents ⛔'",
                timeout=300,
                actions=[
                    cl.Action(
                        name="crew_actions",
                        value="none",
                        label="No Agents ⛔",
                    ),
                    *crew_actions,
                ],
            ).send()

        selected_crew = action_response.get("value") if action_response else "none"

        if selected_crew == "none":
            return RunnableLambda(lambda x: []), None

        # Get the crew runnable
        runnable = crews_metadata.get(selected_crew)
        return RunnableLambda(lambda x: []), runnable

    async def _handle_default_mode(self, intro_message: str) -> VectorStoreRetriever:
        """Handles initialization for the default mode, which sets up the no retriever."""
        await cl.Message(content=intro_message).send()
        return await self._get_retriever("none")

    @cl.on_chat_start
    async def on_chat_start(self):
        """Initialize a new chat environment based on the selected profile."""
        if self.mode == "tool":
            raise ValueError("Tool mode is not supported here")

        chat_profile = cl.user_session.get("chat_profile")
        cl.user_session.set("memory", self._chat_memory)

        # Chat Only mode will be configured via starters instead
        if chat_profile == "Chat Only":
            return

        intro_message = self._generate_welcome_message(chat_profile)

        runnable = None

        if chat_profile == "Agent Mode":
            retriever, runnable = await self._handle_agent_only_mode(intro_message)

        elif chat_profile == "File Mode":
            retriever = await self._handle_file_mode(intro_message)
        else:
            retriever = await self._handle_default_mode(intro_message)

        self._setup_runnable(retriever=retriever, runnable=runnable)

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
                intro_message = self._generate_welcome_message(chat_profile)
                retriever = await self._handle_chat_only_mode(intro_message, message.id)
                self._setup_runnable(retriever=retriever)
                return

            message.content, source_label = self._get_starter_source_label(
                message.content
            )
            await message.update()

            # Now we should set the retriever for the other messages
            retriever = await self._handle_chat_only_mode("", source_label=source_label)
            self._setup_runnable(retriever=retriever)

        runnable: RunnableSequence = cl.user_session.get("runnable")
        memory: ConversationBufferWindowMemory = cl.user_session.get("memory")

        msg = cl.Message(content="")

        query = {"question": message.content}
        _sources = None
        _answer = None
        text_elements = []  # type: List[cl.Text]

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
            await self.asetup_runnable()
        response: dict = await self._runnable.ainvoke({"question": query})
        return response.get("answer")

    def setup_tool(self) -> Tool:
        """Create a tool object"""
        _tool = Tool(
            name=self.tool_name,
            description=self.tool_description,
            func=None,
            coroutine=self._run,
        )
        return _tool
