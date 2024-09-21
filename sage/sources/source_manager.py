# sources_manager.py
import asyncio
import threading
from pathlib import Path as SyncPath
from typing import Any, List

from anyio import Path
from anyio import Path as AsyncPath
from faiss import IndexFlatL2
from langchain.indexes import SQLRecordManager, aindex
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders.confluence import (
    ContentFormat,
)

from sage.constants import (
    EMBED_DIMENSION,
    EMBEDDING_MODEL,
    logger,
    sources_config,
)
from sage.sources.loaders import CustomConfluenceLoader, GitlabLoader, WebLoader
from sage.utils.exceptions import SourceException
from sage.utils.supports import CustomFAISS as FAISS
from sage.validators.config_toml import ConfluenceModel, Files, GitlabModel, Web


async def get_faiss_indexes(faiss_dir: AsyncPath | SyncPath) -> List[str]:
    """Returns a list of all available faiss indexes"""
    indexes = []

    if isinstance(faiss_dir, AsyncPath):
        async for file in faiss_dir.glob("*.faiss"):
            indexes.append(file.stem)
    else:
        for file in faiss_dir.glob("*.faiss"):
            indexes.append(file.stem)
    return indexes


def convert_sources_to_string():
    """Helper to format the sources dictionary into a readable string."""
    source_messages = []
    for source_name, source_info in vars(sources_config).items():
        if not source_info:
            continue
        # For each source, create a formatted string
        if source_name == "confluence":
            spaces = ", ".join(source_info.spaces)
            source_messages.append(f"- Confluence spaces: {spaces}")
        elif source_name == "gitlab":
            if source_info.groups:
                groups = ", ".join(source_info.groups)
                source_messages.append(f"- GitLab Groups: {groups}")
            if source_info.projects:
                projects = ", ".join(source_info.projects)
                source_messages.append(f"- GitLab repositories: {projects}")
        elif source_name == "web":
            links = ", ".join(source_info.links)
            source_messages.append(
                f"- Relevant documentation and resources available at: {links}"
            )
    return "\n  ".join(source_messages)


class AsyncRunner:
    """
    Helper class to run asynchronous coroutines in a separate thread.

    This class creates and manages a new event loop in a separate daemon thread,
    allowing synchronous code to run asynchronous coroutines and wait for their results.
    It's primarily used to bridge the gap between synchronous and asynchronous code.

    Methods:
        __init__: Initializes the AsyncRunner by creating a new event loop and starting it in a daemon thread.
        start_loop: Sets the event loop for the thread and runs it forever.
        run: Submits a coroutine to the event loop and waits for the result.
        shutdown: Cleanly shuts down the event loop and the associated thread.
    """

    def __init__(self):
        # Create a new event loop for this thread
        self.loop = asyncio.new_event_loop()
        # Start the event loop in a new daemon thread
        self.thread = threading.Thread(target=self.start_loop, daemon=True)
        self.thread.start()

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # Submit the coroutine to the event loop and wait for the result
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def shutdown(self):
        # Cleanly shutdown the event loop and thread
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


class SourceManager:
    """
    Helper class for creating and adding sources.
    It contains helpers for collecting source documents and saving it into a FAISS database
    """

    def __init__(
        self, source_dir: Path, record_manager_dir: Path | None = None
    ) -> None:
        """
        Initialize SourceManager with directories for source and record manager files.

        Args:
            source_dir (Path): The base directory for sources.
            record_manager_dir (Path, optional): The directory for recording manager files. Defaults to None.
        """
        self._async_runner = AsyncRunner()
        self.source_dir = source_dir
        self._record_manager_file: Path = (
            record_manager_dir or source_dir
        ) / "dbs_record_manager.sql"
        self.faiss_dir: Path = source_dir / "faiss"

    async def _get_or_create_faiss_db(self, source_hash: str) -> FAISS:
        """Create or return any existing FAISS Database if available on the local disk"""
        faiss_dbs_paths = await get_faiss_indexes(self.faiss_dir)
        if source_hash not in faiss_dbs_paths:
            # create an empty db and return it
            db = FAISS(
                embedding_function=EMBEDDING_MODEL,
                index=IndexFlatL2(EMBED_DIMENSION),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            return db
        # Load an existing db
        db = FAISS.load_local(
            folder_path=str(self.faiss_dir),
            index_name=source_hash,
            embeddings=EMBEDDING_MODEL,
            allow_dangerous_deserialization=True,
        )
        return db

    def _get_or_create_faiss_db_sync(self, source_hash: str) -> FAISS:
        """Create or return any existing FAISS Database if available on the local disk"""
        db = self._async_runner.run(self._get_or_create_faiss_db(source_hash))
        return db

    async def _get_record_manager(self, source_hash: str) -> SQLRecordManager:
        """Return a Record Manager connected to a given source"""

        record_manager = SQLRecordManager(
            namespace=source_hash,
            async_mode=True,
            db_url=f"sqlite+aiosqlite:///{self._record_manager_file}",
        )
        try:
            await record_manager.acreate_schema()
        except Exception as e:
            if "table upsertion_record already exists" in str(e):
                logger.warning(
                    "Table 'upsertion_record' already exists. Not creating new table - RecordManager."
                )
            else:
                raise
        return record_manager

    async def _create_and_save_db(
        self,
        source_hash: str,
        documents: List[Document],
        save_db: bool = True,
        cleanup: str | None = "full",
    ) -> None:
        """Creates and saves a vector store index DB to file."""
        logger.debug(f"Creating vector store for source with hash - {source_hash}")

        db = (
            await self._get_or_create_faiss_db(source_hash)
            if save_db
            else await FAISS.afrom_documents(
                documents=documents, embedding=EMBEDDING_MODEL
            )
        )

        if save_db:
            record_manager = await self._get_record_manager(source_hash)
            await aindex(
                docs_source=documents,
                record_manager=record_manager,
                vector_store=db,
                cleanup=cleanup,
                source_id_key="source",
            )
            db.save_local(str(self.faiss_dir), source_hash)
            logger.debug(
                f"Successfully created and saved vector store for source with hash - {source_hash}"
            )
        else:
            logger.debug(
                f"Successfully created vector store for source with hash - {source_hash}"
            )

    @staticmethod
    def get_text_splitter() -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300, length_function=len
        )

    async def _add_source(
        self,
        hash: str,
        documents: List[Document],
        save_db: bool = True,
        cleanup: str | None = "full",
    ) -> FAISS:
        """Adds documents to FAISS and handles its creation and save."""
        return await self._create_and_save_db(
            source_hash=hash,
            documents=self.get_text_splitter().split_documents(documents),
            save_db=save_db,
            cleanup=cleanup,
        )

    async def _add_text(
        self, hash: str, data: str, metadata: dict, cleanup: str | None = "full"
    ):
        """
        Adds and saves a raw text data source.

        Args:
            hash (str): The source hash
            data (str): The text data to be saved
            metadata (dict): The data metadata

        Raises:
            SourceException: Error creating or updating the data source
        """
        try:
            documents = [
                Document(page_content=data, metadata={**metadata, "hash": hash})
            ]
            logger.debug(f"Processed {len(documents)} documents from the text source")

            await self._add_source(hash, documents, cleanup=cleanup)

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading text source: {str(error)}"
            )

    async def _add_confluence(
        self,
        hash: str,
        source: ConfluenceModel,
        space: str,
        cleanup: str | None = "full",
    ):
        """
        Adds and saves a confluence data source

        Args:
            hash (str): The source hash
            source (ConfluenceModel): Confluence data model
            space (str): The confluence space to save

        Raises:
            SourceException: HTTP Error communicating with confluence
        """
        try:
            loader = CustomConfluenceLoader(
                url=source.server,
                username=source.username,
                api_key=source.password.get_secret_value(),
                space_key=space,
                include_attachments=False,
                limit=200,
                keep_markdown_format=True,
                content_format=ContentFormat.VIEW,
            )

            documents = loader.load()
            logger.debug(
                f"Processed {len(documents)} documents from the confluence source"
            )

            await self._add_source(hash, documents, cleanup=cleanup)

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading confluence source: {str(error)}"
            )

    async def _add_gitlab_source(
        self,
        hash: str,
        source: GitlabModel,
        groups: List[str] = [],
        projects: List[str] = [],
        cleanup: str | None = "full",
    ):
        """
        Adds and saves a gitlab data source.
        """
        try:
            loader = GitlabLoader(
                base_url=source.server,
                groups=groups,
                projects=projects,
                private_token=source.password,
                ssl_verify=True,
                max_concurrency=source.max_concurrency,
            )

            documents = await loader.load()

            logger.debug(f"Processed {len(documents)} documents from the Gitlab source")
            await self._add_source(hash, documents, cleanup=cleanup)

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading gitlab source: {str(error)}"
            )

    async def _add_web_source(
        self, hash: str, source: Web, link: str, cleanup: str | None = "full"
    ):
        """
        Adds and saves a web data source

        Args:
            hash (str): The source hash
            source (Web): Web data model
            link (str): The source link value

        Raises:
            SourceException: Exception raised interacting with web links
        """
        try:
            loader = WebLoader(
                nested=source.nested,
                ssl_verify=source.ssl_verify,
                urls=[link],
                headers=source.headers,
                max_concurrency=source.max_concurrency,
                max_depth=source.max_depth,
            )

            documents = await loader.load()

            if not documents:
                raise SourceException(
                    f"No document was parsed from the source link {link}"
                )
            logger.debug(f"Processed {len(documents)} documents from the Web source")

            await self._add_source(hash, documents, cleanup=cleanup)

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading web source: {str(error)}"
            )

    async def _add_files_source(
        self,
        hash: str,
        path: str,
        source: Files = None,
        save_db: bool = True,
        cleanup: str | None = "full",
    ) -> FAISS | None:
        """
        Adds and saves a files data source

        Args:
            hash (str): The source hash
            source (Files): Files data model

        Raises:
            SourceException: Exception raised interacting with the files
        """
        try:
            loader = UnstructuredFileLoader(file_path=path, mode="single")

            documents = loader.load()

            if not documents:
                raise SourceException(
                    f"No document was parsed from the source path: {path}"
                )
            logger.debug(f"Processed {len(documents)} documents from the Files source")

            return await self._add_source(
                hash, documents, save_db=save_db, cleanup=cleanup
            )

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading file source: {str(error)}"
            )

    async def add(
        self,
        hash: str,
        source_type: str,
        identifier: str,
        identifier_type: str = None,
        data: Any = None,
        cleanup: str | None = "full",
    ) -> None:
        """
        A helper for processing and adding various source types.

        Args:
            hash (str): The MD5 source hash.
            source_type (str): The type of source.
            identifier (str): Identifier of the source.
            identifier_type (str, optional): The type of identifier. Defaults to None.
            data (Any, optional): An optional data field. Defaults to None.
            cleanup (str | None, optional): An optional cleanup policy. Defaults to "full".

        Raises:
            SourceException: Raised if the source type is not known.
        """
        if source_type == "text":
            await self._add_text(
                hash=hash,
                data=data,
                metadata=identifier,
                cleanup=cleanup,
            )
        elif source_type == "confluence":
            await self._add_confluence(
                hash=hash,
                source=sources_config.confluence,
                space=identifier,
                cleanup=cleanup,
            )
        elif source_type == "web":
            await self._add_web_source(
                hash=hash, source=sources_config.web, link=identifier, cleanup=cleanup
            )
        elif source_type == "gitlab":
            if identifier_type == "groups":
                await self._add_gitlab_source(
                    hash=hash,
                    source=sources_config.gitlab,
                    groups=[identifier],
                    cleanup=cleanup,
                )
            else:
                await self._add_gitlab_source(
                    hash=hash,
                    source=sources_config.gitlab,
                    projects=[identifier],
                    cleanup=cleanup,
                )
        else:
            raise SourceException(f"Unknown source type: {source_type}")

    def add_sync(
        self,
        hash: str,
        source_type: str,
        identifier: str,
        identifier_type: str = None,
        data: Any = None,
        cleanup: str | None = "full",
    ) -> None:
        """
        An Helper for processing and adding various source types

        Args:
            hash (str): The MD5 source hash
            source_type (str): The type of source
            identifier (str): Identifier from the source
            identifier_type (str): The type of identifier
            data (Any): An optional data field
            cleanup (str | None): An optional cleanup policy

        Raises:
            SourceException: Raised if the source type is not known
        """
        self._async_runner.run(
            self.add(
                hash=hash,
                source_type=source_type,
                identifier=identifier,
                identifier_type=identifier_type,
                data=data,
                cleanup=cleanup,
            )
        )
