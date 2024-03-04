from hashlib import md5
from typing import List, Optional

from anyio import Path as aPath
from chainlit.types import AskFileResponse
from constants import (
    EMBEDDING_MODEL,
    core_config,
    logger,
    sources_config,
    validated_config,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders.confluence import (
    ContentFormat,
)
from langchain_community.vectorstores.faiss import FAISS
from utils.exceptions import SourceException
from utils.loaders import CustomConfluenceLoader, GitlabLoader, WebLoader
from utils.supports import (
    aexecute_concurrently,
)
from utils.validator import ConfluenceModel, Files, GitlabModel, Web


class Source:
    # TODO: Adds support for batching loading of the documents when generating the Faiss index. As it's easy to reach API throttle limits with OPENAI
    # TODO: Old sources metadata are not removed when the source change causing issue if old sources are used again as the source will not loaded because the metadata still exists

    _instance = None
    source_dir = aPath(core_config.data_dir) / "sources"
    _retriever_args = {"k": sources_config.top_k}
    """Custom retriever search args"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.source_refresh_list: List[dict] = list()

    @staticmethod
    def sources_to_string():
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

    @staticmethod
    def _get_hash(input: str) -> str:
        return md5(input.encode()).hexdigest()

    def _get_source_metadata_path(self, source_hash: str) -> aPath:
        """Get the source metadata"""
        return self.source_dir / source_hash

    async def _save_source_metadata(self, source_hash: str):
        """Save the source metadata to disk"""
        metadata = self._get_source_metadata_path(source_hash)
        await metadata.write_text("True")

    async def _source_exist_locally(self, source_hash: str):
        """Returns whether the given source exist or not"""
        return await self._get_source_metadata_path(source_hash).exists()

    def _get_source_hash(self, source_type: str, name: str) -> str:
        """Generate a hash for a given source"""
        return f"{source_type}-{self._get_hash(name.lower())}.source"

    async def _append_to_refresh_list(
        self, source_type: str, identifier: str, identifier_key: str
    ):
        """Add source to the refresh list"""
        source_hash = self._get_source_hash(source_type, identifier)
        if not await self._source_exist_locally(source_hash):
            self.source_refresh_list.append(
                {
                    "hash": source_hash,
                    "source_type": source_type,
                    "identifier": identifier,
                    "identifier_type": identifier_key,
                }
            )

    async def _check_source(
        self,
        source_type: str,
        source_data: ConfluenceModel | GitlabModel | Web,
        identifier_key: str,
    ):
        """Check if a source exist otherwise add to the refresh list"""
        for identifier in getattr(source_data, identifier_key, []):
            await self._append_to_refresh_list(source_type, identifier, identifier_key)

    async def check_sources_exist(self):
        """Checks if sources in the config file exists locally"""
        await self.source_dir.mkdir(exist_ok=True)
        for source_type, source_data in vars(sources_config).items():
            if not source_data:
                continue

            if isinstance(source_data, ConfluenceModel):
                await self._check_source(source_type, source_data, "spaces")
            elif isinstance(source_data, GitlabModel):
                await self._check_source(source_type, source_data, "groups")
                await self._check_source(source_type, source_data, "projects")
            elif isinstance(source_data, Web):
                await self._check_source(source_type, source_data, "links")

    async def _create_and_save_db(
        self, source_hash: str, documents: List[Document], save_db: bool = True
    ) -> FAISS | None:
        """Creates and save a vector store index DB to file"""
        logger.debug(f"Creating a vector store for source with hash - {source_hash}")

        # Create vector index
        db = await FAISS.afrom_documents(documents=documents, embedding=EMBEDDING_MODEL)

        if save_db:
            # Save DB to source directory
            dir_path = self.source_dir / "faiss"
            db.save_local(str(dir_path), source_hash)
            logger.debug(
                f"Successfully created and saved vector store for source with hash - {source_hash}"
            )
            return
        logger.debug(
            f"Successfully created the vector store for source with hash - {source_hash}"
        )
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
            )

            gitlab_documents = loader.load()

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
                nested=source.nested, ssl_verify=source.ssl_verify, urls=[link]
            )

            web_documents = loader.load()

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

    async def _process_source_type(
        self, hash: str, source_type: str, identifier: str, identifier_type: str
    ) -> None:
        """
        An Helper for processing the various source types

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

    async def add_source(
        self, hash: str, source_type: str, identifier: str, identifier_type: str
    ) -> None:
        """
        Adds and saves a data source

        Args:
            hash (str) | The source hash
            source_type (str) | The source type
            identifier (str) | The value of the source
            identifier_type (str) | The type of source value

        Raises:
            SourceException: Raised if the source type is unknown
        """

        source_ref = f"{source_type}: {identifier_type}={identifier}"
        logger.info(f"Processing source {source_ref} ...")
        try:
            await self._process_source_type(
                hash, source_type, identifier, identifier_type
            )
        except Exception as e:
            logger.error(f"An error has occurred processing source {source_ref}")
            logger.error(str(e))
            logger.error("Source will retry next re-run")
        else:
            await self._save_source_metadata(hash)
            logger.info(f"Done with source {source_ref}")

    async def run(self) -> None:
        """
        Process each source in the self.source_refresh_list.
        """
        await self.check_sources_exist()

        if len(self.source_refresh_list) == 0:
            logger.info("No changes to sources")
            return

        await aexecute_concurrently(
            func=self.add_source, items=self.source_refresh_list, input_type="dict"
        )

    async def _get_faiss_indexes(self) -> str | List[str]:
        """Returns a list of all available faiss indexes"""
        dir_path = self.source_dir / "faiss"

        indexes: List[str] = []

        async for file in dir_path.glob("*.faiss"):
            indexes.append(file.stem)

        return str(dir_path), indexes

    @staticmethod
    def _combine_dbs(dbs: List[FAISS]) -> FAISS:
        """
        Combines various DBs into a single one

        Args:
            dbs (List[FAISS]): A List of FAISS indexes
        """
        faiss_db: FAISS = dbs[0]

        for db in dbs[1:]:
            faiss_db.merge_from(db)

        return faiss_db

    def _compression_retriever(
        self, retriever: VectorStoreRetriever
    ) -> ContextualCompressionRetriever:
        """Loads a compression retriever"""

        ranker_config = validated_config.reranker

        if not ranker_config:
            raise SourceException("There is no valid reranker configuration found")

        try:
            if ranker_config.type == "cohere":
                from langchain.retrievers.document_compressors import CohereRerank

                _compressor = CohereRerank(
                    top_n=ranker_config.top_n,
                    model=ranker_config.cohere.name,
                    cohere_api_key=ranker_config.cohere.password.get_secret_value(),
                    user_agent=core_config.user_agent,
                )
            elif ranker_config.type == "huggingface":
                from utils.supports import BgeRerank

                _compressor = BgeRerank(
                    name=ranker_config.huggingface.name,
                    top_n=ranker_config.top_n,
                    cache_dir=str(core_config.data_dir + "/models"),
                    revision=ranker_config.huggingface.revision,
                )
            else:
                raise SourceException(
                    f"Reranker type {ranker_config.type} not supported has a valid compression retriever"
                )
        except Exception as error:
            raise SourceException(str(error))

        _compression_retriever = ContextualCompressionRetriever(
            base_compressor=_compressor, base_retriever=retriever
        )
        return _compression_retriever

    ## Helper to create a retriever while the data input is a list of files path
    async def load_files_retriever(
        self, files: List[AskFileResponse]
    ) -> ContextualCompressionRetriever | VectorStoreRetriever:
        """
        Create a retriever from a list of files input from the chainlit interface

        Args:
            files (List[AskFileResponse]): A list of files

        Returns:
            ContextualCompressionRetriever | VectorStoreRetriever: A retriever instance
        """

        ## TODO: Remove when chainlit set the file path to match the exact file name
        async def rename_file_path(file: AskFileResponse) -> AskFileResponse:
            """Extract the base path and file extension from the current file path"""
            file_path = aPath(file.path)
            new_path = file_path.with_name(file.name)

            try:
                await file_path.rename(new_path)
                file.path = str(new_path)
                return file
            except OSError as e:
                logger.warning(f"Error: {e.strerror}")
                return file

        async def process_file(file: AskFileResponse) -> FAISS:
            """Simple helper to process the files"""
            file = await rename_file_path(file)
            file_source = Files(paths=[file.path])
            db = await self._add_files_source(
                hash=file.id, source=file_source, path=file.path, save_db=False
            )
            return db

        async def cleanup_files(files: List[AskFileResponse]):
            """Deletes a list of files from the filesystem."""
            for file_obj in files:
                try:
                    file_path = aPath(file_obj.path)
                    if await file_path.is_file():
                        await file_path.unlink()
                        logger.debug(f"Deleted file: {file_path}")
                    else:
                        logger.warning(f"File not found: {file_path}")
                except OSError as e:
                    logger.error(f"Error deleting file {file_path}: {e.strerror}")

        dbs: List[FAISS] = await aexecute_concurrently(
            process_file, files, max_workers=10
        )
        faiss_db = self._combine_dbs(dbs)
        retriever = faiss_db.as_retriever(search_kwargs=self._retriever_args)

        await cleanup_files(files)

        if not validated_config.reranker:
            return retriever
        else:
            return self._compression_retriever(retriever)

    def _load_retriever(self, db_path: str, indexes: List[str]):
        """Loads a retriever"""
        if not indexes:
            return None

        dbs: List[FAISS] = []

        for index in indexes:
            db = FAISS.load_local(
                folder_path=db_path, index_name=index, embeddings=EMBEDDING_MODEL
            )
            dbs.append(db)

        faiss_db = self._combine_dbs(dbs)

        return faiss_db.as_retriever(search_kwargs=self._retriever_args)

    async def load(
        self,
    ) -> Optional[VectorStoreRetriever | ContextualCompressionRetriever]:
        """
        Returns either a retriever model from the FAISS vector indexes or compression based retriever model
        """

        await self.run()

        db_path, indexes = await self._get_faiss_indexes()

        _retriever = self._load_retriever(db_path, indexes)

        if not validated_config.reranker:
            return _retriever
        else:
            return self._compression_retriever(_retriever)
