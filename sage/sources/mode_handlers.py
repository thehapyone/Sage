import chainlit as cl
from langchain.schema.runnable import (
    RunnableLambda,
)
from langchain.schema.vectorstore import VectorStoreRetriever

from sage.constants import agents_crew, validated_config
from sage.sources.runnable import RunnableBase
from sage.sources.sources import Source
from sage.sources.utils import (
    generate_ui_actions,
    get_retriever,
)
from sage.utils.exceptions import AgentsException


class ChatModeHandlers:
    def __init__(self, runnable_handler: RunnableBase, source: Source = Source()):
        self._runnable_handler = runnable_handler
        self.source = source

    async def handle_file_mode(self, intro_message: str) -> VectorStoreRetriever:
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
        retriever = await self.source.load_files_retriever(files)
        # Let the user know that the system is ready
        file_names = "\n  ".join([file.name for file in files])
        msg.content = (
            "The following files are now processed and ready to be used!\n"
            f"  {file_names}"
        )
        await cl.sleep(1)
        await msg.update()
        return retriever

    async def handle_chat_only_mode(
        self, intro_message: str, root_id: str = None, source_label: str = None
    ) -> VectorStoreRetriever:
        """Handles initialization for 'Chat Only' mode, where users select a source to chat with."""
        # Get the sources labels that will be used to create the source actions
        sources_metadata = await self.source.get_labels_and_hash()

        if source_label:
            hash_key = next(
                (k for k, v in sources_metadata.items() if v == source_label), "none"
            )
            return await get_retriever(source=self.source, source_hash=hash_key)

        await cl.Message(id=root_id, content=intro_message).send()

        source_actions = generate_ui_actions(sources_metadata)

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
        return await get_retriever(source=self.source, source_hash=selected_hash)

    async def handle_agent_only_mode(
        self, intro_message: str, root_id: str = None, crew_label: str = None
    ) -> tuple[RunnableLambda, RunnableLambda]:
        """
        Handles initialization for 'Agent Only' mode, where users select a crew to chat with.

        Returns a tuple containing a black retriever instance and an optional instance for the crew
        """
        # Get the crew names that will be used to create the source actions
        crews_metadata = self._runnable_handler.create_crew_runnable(agents_crew)

        if crew_label:
            crew_instance = crews_metadata.get(crew_label, None)
            if crew_instance:
                return RunnableLambda(lambda x: []), crew_instance
            raise AgentsException(f"The crew {crew_label} can not be found")

        await cl.Message(id=root_id, content=intro_message).send()

        crew_actions = generate_ui_actions(crews_metadata, "crew_actions")

        action_response = None

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

    async def handle_default_mode(self, intro_message: str) -> VectorStoreRetriever:
        """Handles initialization for the default mode, which sets up the no retriever."""
        await cl.Message(content=intro_message).send()
        return await get_retriever(source=self.source, source_hash="none")
