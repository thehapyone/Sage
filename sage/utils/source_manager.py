# sources_manager.py
from typing import List

from anyio import Path
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
from sage.utils.exceptions import SourceException
from sage.utils.loaders import CustomConfluenceLoader, GitlabLoader, WebLoader
from sage.utils.supports import CustomFAISS as FAISS
from sage.utils.validator import ConfluenceModel, Files, GitlabModel, Web


async def get_faiss_indexes(faiss_dir: Path) -> List[str]:
    """Returns a list of all available faiss indexes"""
    indexes: List[str] = []

    async for file in faiss_dir.glob("*.faiss"):
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


class SourceManager:
    """
    Helper class for creating and adding sources.
    It contains helpers for collecting source documents and saving it into a FAISS database
    """

    def __init__(self, source_dir: Path) -> None:
        """The base source dir"""
        self.source_dir = source_dir
        """The base source dir"""
        self._record_manager_file: Path = source_dir / "dbs_record_manager.sql"
        """Path to the sources record manager"""
        self.faiss_dir: Path = source_dir / "faiss"
        """Directory where the Faiss DBs are saved"""

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
        self, source_hash: str, documents: List[Document], save_db: bool = True
    ) -> FAISS | None:
        """Creates and save a vector store index DB to file"""
        logger.debug(f"Creating a vector store for source with hash - {source_hash}")

        if save_db:
            # Get vector store
            db = await self._get_or_create_faiss_db(source_hash)

            # Get record manager
            record_manager = await self._get_record_manager(source_hash)

            # Update the DBs via the Index API
            await aindex(
                docs_source=documents,
                record_manager=record_manager,
                vector_store=db,
                cleanup="full",
                source_id_key="source",
            )

            # Save DB to source directory
            db.save_local(str(self.faiss_dir), source_hash)
            logger.debug(
                f"Successfully created and saved vector store for source with hash - {source_hash}"
            )
            return
        logger.debug(
            f"Successfully created the vector store for source with hash - {source_hash}"
        )
        # Create vector index without any indexing api
        db = await FAISS.afrom_documents(documents=documents, embedding=EMBEDDING_MODEL)

        return db

    @staticmethod
    def splitter() -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300, length_function=len
        )

    async def _add_confluence(self, hash: str, source: ConfluenceModel, space: str):
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
            )

            confluence_documents = loader.load(
                space_key=space,
                include_attachments=False,
                limit=200,
                keep_markdown_format=True,
                content_format=ContentFormat.VIEW,
            )
            logger.debug(
                f"Processed {len(confluence_documents)} documents from the confluence source"
            )

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading confluence source: {str(error)}"
            )

        await self._create_and_save_db(
            source_hash=hash,
            documents=self.splitter().split_documents(confluence_documents),
        )

    async def _add_gitlab_source(
        self,
        hash: str,
        source: GitlabModel,
        groups: List[str] = [],
        projects: List[str] = [],
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
                max_concurrency=source.max_concurrency
            )

            gitlab_documents = await loader.load()

            logger.debug(
                f"Processed {len(gitlab_documents)} documents from the Gitlab source"
            )

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading gitlab source: {str(error)}"
            )

        await self._create_and_save_db(
            source_hash=hash,
            documents=self.splitter().split_documents(gitlab_documents),
        )

    async def _add_web_source(self, hash: str, source: Web, link: str):
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

            web_documents = await loader.load()

            if not web_documents:
                raise SourceException(
                    f"No document was parsed from the source link {link}"
                )
            logger.debug(
                f"Processed {len(web_documents)} documents from the Web source"
            )

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading web source: {str(error)}"
            )

        await self._create_and_save_db(
            source_hash=hash, documents=self.splitter().split_documents(web_documents)
        )

    async def _add_files_source(
        self, hash: str, path: str, source: Files = None, save_db: bool = True
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

            file_documents = loader.load()

            if not file_documents:
                raise SourceException(
                    f"No document was parsed from the source path: {path}"
                )
            logger.debug(
                f"Processed {len(file_documents)} documents from the Files source"
            )

        except Exception as error:
            raise SourceException(
                f"An error has occurred while loading file source: {str(error)}"
            )

        db = await self._create_and_save_db(
            source_hash=hash,
            documents=self.splitter().split_documents(file_documents),
            save_db=save_db,
        )
        return db

    async def add(
        self, hash: str, source_type: str, identifier: str, identifier_type: str
    ) -> None:
        """
        An Helper for processing and adding various source types

        Args:
            hash (str): The MD5 source hash
            source_type (str): The type of source
            identifier (str): Identifier from the source
            identifier_type (str): The type of identifier

        Raises:
            SourceException: Raised if the source type is not known
        """
        if source_type == "confluence":
            await self._add_confluence(
                hash=hash, source=sources_config.confluence, space=identifier
            )
        elif source_type == "web":
            await self._add_web_source(
                hash=hash, source=sources_config.web, link=identifier
            )
        elif source_type == "gitlab":
            if identifier_type == "groups":
                await self._add_gitlab_source(
                    hash=hash, source=sources_config.gitlab, groups=[identifier]
                )
            else:
                await self._add_gitlab_source(
                    hash=hash, source=sources_config.gitlab, projects=[identifier]
                )
        else:
            raise SourceException(f"Unknown source type: {source_type}")
