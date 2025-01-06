from hashlib import md5
from typing import Any, List, Optional

from anyio import Path
from chainlit.types import AskFileResponse
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema.runnable import RunnableLambda
from langchain.schema.vectorstore import VectorStoreRetriever

from sage.constants import (
    EMBED_DIMENSION,
    EMBEDDING_MODEL,
    core_config,
    logger,
    sources_config,
    validated_config,
)
from sage.sources.retrievers import MultiSearchQueryRetriever
from sage.sources.source_manager import (
    SourceManager,
    convert_sources_to_string,
    get_faiss_indexes,
)
from sage.utils.exceptions import SourceException
from sage.utils.labels import generate_source_label
from sage.utils.supports import CustomFAISS as FAISS
from sage.utils.supports import ReRanker, aexecute_concurrently, asyncify
from sage.validators.config_toml import ConfluenceModel, Files, GitlabModel, Web


class Source:
    # TODO: Adds support for batching loading of the documents when generating the Faiss index. As it's easy to reach API throttle limits with OPENAI
    # TODO: Old sources metadata are not removed when the source change causing issue if old sources are used again as the source will not loaded because the metadata still exists

    _instance = None
    source_dir = core_config.data_dir / "sources"
    _retriever_args = {"k": sources_config.top_k}
    """Custom retriever search args"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.source_refresh_list: List[dict] = list()

        self.manager = SourceManager(
            embedding_model=EMBEDDING_MODEL,
            model_dimension=EMBED_DIMENSION,
            source_dir=self.source_dir,
        )
        self.force_data_refresh = False

    @staticmethod
    def sources_to_string():
        """Helper to format the sources dictionary into a readable string."""
        return convert_sources_to_string(sources_config)

    @staticmethod
    def _get_hash(input: str) -> str:
        return md5(input.encode()).hexdigest()

    def _get_source_metadata_path(self, source_hash: str) -> Path:
        """Get the source metadata"""
        return self.source_dir / source_hash

    async def _save_source_metadata(self, source_metadata: str, source_hash: str):
        """Save the source metadata to disk"""
        metadata_path = self._get_source_metadata_path(source_hash)
        if await metadata_path.exists():
            return
        source_label = await generate_source_label(source_metadata)
        await metadata_path.write_text(source_label)

    async def _source_exist_locally(self, source_hash: str):
        """Returns whether the given source exist or not"""
        return await self._get_source_metadata_path(source_hash).exists()

    def _get_source_hash(self, source_type: str, name: str) -> str:
        """Generate a hash for a given source"""
        return f"{source_type}-{self._get_hash(name.lower())}.source"

    async def _append_to_refresh_list(
        self, source_type: str, identifier: str, identifier_key: str
    ):
        """
        Add a source to the refresh list if it needs to be refreshed.

        This method checks whether a source should be added to the refresh list
        based on its hash. The check is performed by comparing the hash of the
        source with existing metadata. If metadata does not exist locally or
        if data refresh is forced, the source is added to the refresh list.

        Args:
            source_type (str): The type of the source (e.g. 'confluence', 'gitlab', 'web').
            identifier (str): The specific identifier for the source
                (e.g. space key for Confluence, group name or project name for Gitlab, or a URL for web sources).
            identifier_key (str): The key associated with the identifier
                (e.g. 'spaces' for Confluence, 'groups' or 'projects' for Gitlab, or 'links' for web sources).
        """
        source_hash = self._get_source_hash(source_type, identifier)
        if not self.force_data_refresh and await self._source_exist_locally(
            source_hash
        ):
            return

        source_data_mapping = {
            "confluence": sources_config.confluence,
            "web": sources_config.web,
            "gitlab": sources_config.gitlab,
        }

        data = source_data_mapping.get(source_type)
        if data is None:
            raise SourceException(f"Unknown source type: {source_type}")

        self.source_refresh_list.append(
            {
                "hash": source_hash,
                "source_type": source_type,
                "identifier": identifier,
                "identifier_type": identifier_key,
                "data": data,
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

    async def add_source(
        self,
        hash: str,
        source_type: str,
        identifier: str,
        identifier_type: str,
        data: Any,
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
            await self.manager.add(
                hash, source_type, identifier, identifier_type, data=data
            )
        except Exception as e:
            logger.error(f"An error has occurred processing source {source_ref}")
            logger.error(str(e), exc_info=True)
            logger.error("Source will retry next re-run")
        else:
            await self._save_source_metadata(source_ref, hash)
            logger.info(f"Done with source {source_ref}")

    async def run(self, refresh: bool = False) -> None:
        """
        Process each source in the self.source_refresh_list.

        Args:
            refresh (bool, optional): A flag for reindexing the data sources.

        """
        if refresh:
            logger.info("Sources will now be reindexed")
            self.force_data_refresh = refresh

        await self.check_sources_exist()

        if len(self.source_refresh_list) == 0:
            logger.info("No changes to sources")
            return

        await aexecute_concurrently(
            func=self.add_source, items=self.source_refresh_list, input_type="dict"
        )

        # Disable data refresh
        self.force_data_refresh = False

    @staticmethod
    def _combine_dbs(dbs: List[FAISS]) -> FAISS:
        """
        Combines various DBs into a single one

        Args:
            dbs (List[FAISS]): A List of FAISS indexes
        """
        faiss_db: FAISS = dbs[0]

        for db in dbs[1:]:
            try:
                faiss_db.merge_from(db)
            except ValueError:
                logger.warning(
                    f"Duplicate DB detected. DB merge will be skipped {db.index}"
                )

        return faiss_db

    def _compression_retriever(
        self, retriever: VectorStoreRetriever
    ) -> ContextualCompressionRetriever:
        """Loads a compression retriever"""

        ranker_config = validated_config.reranker

        if not ranker_config:
            raise SourceException("There is no valid reranker configuration found")

        try:
            _compressor = ReRanker(
                top_n=ranker_config.top_n,
                model=ranker_config.model,
                revision=ranker_config.revision,
                cache_dir=str(core_config.data_dir / "models"),
            )

            _compression_retriever = ContextualCompressionRetriever(
                base_compressor=_compressor, base_retriever=retriever
            )
        except Exception as e:
            logger.error(
                "An error has occurred while loading the compression retriever",
                exc_info=True,
            )
            raise e

        return _compression_retriever

    ## Helper to create a retriever while the data input is a list of files path
    async def load_files_retriever(
        self, files: List[AskFileResponse]
    ) -> MultiSearchQueryRetriever:
        """
        Asynchronously creates a retriever from a list of files provided by the Chainlit interface.

        This method performs several steps:
        - Renames file paths to match the exact file names, addressing a discrepancy with Chainlit.
        - Processes the files to build individual FAISS databases for each file.
        - Combines the FAISS databases into a single retriever instance.
        - Cleans up the files from the filesystem after processing.
        - Optionally wraps the retriever with contextual compression if enabled in the config.

        The function ensures that the files are deleted after processing to avoid cluttering
        the filesystem.

        Args:
            files: A list of AskFileResponse objects representing the files to be processed.

        Returns:
            A MultiSearchQueryRetriever instance that is either a ContextualCompressionRetriever or a
            VectorStoreRetriever, depending on whether contextual compression is enabled in
            the application's configuration.

        Raises:
            OSError: If an error occurs during file renaming or cleanup.
        """

        ## TODO: Remove when chainlit set the file path to match the exact file name
        async def rename_file_path(file: AskFileResponse) -> AskFileResponse:
            """Extract the base path and file extension from the current file path"""
            file_path = Path(file.path)
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
            db = await self.manager._add_files_source(
                hash=file.id, data=file_source, path=file.path, save_db=False
            )
            return db

        async def cleanup_files(files: List[AskFileResponse]):
            """Deletes a list of files from the filesystem."""
            for file_obj in files:
                try:
                    file_path = Path(file_obj.path)
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

        final_retriever = (
            retriever
            if not validated_config.reranker
            else self._compression_retriever(retriever)
        )

        return MultiSearchQueryRetriever(retriever=final_retriever)

    def _load_retriever(
        self, indexes: List[str], source_hash: str = "all"
    ) -> Optional[VectorStoreRetriever]:
        """Loads a retriever for selected sources"""
        if not indexes:
            return None

        if source_hash == "none":
            return None

        db_path = str(self.manager.faiss_dir)

        if source_hash != "all":
            db = FAISS.load_local(
                folder_path=db_path,
                index_name=source_hash,
                embeddings=EMBEDDING_MODEL,
                allow_dangerous_deserialization=True,
            )
            retriever = db.as_retriever(search_kwargs=self._retriever_args)
            return retriever

        dbs: List[FAISS] = []

        for index in indexes:
            db = FAISS.load_local(
                folder_path=db_path,
                index_name=index,
                embeddings=EMBEDDING_MODEL,
                allow_dangerous_deserialization=True,
            )
            dbs.append(db)

        faiss_db = self._combine_dbs(dbs)

        retriever = faiss_db.as_retriever(search_kwargs=self._retriever_args)
        return retriever

    async def load(
        self, source_hash: str = "none"
    ) -> Optional[MultiSearchQueryRetriever]:
        """
        Returns either a retriever model from the FAISS vector indexes or compression based retriever model.
        Supports creating a retriever for a selected source hash.
        """
        indexes = await get_faiss_indexes(self.manager.faiss_dir)

        retriever = await asyncify(self._load_retriever)(indexes, source_hash)

        if retriever is None:
            return RunnableLambda(lambda x: [])

        final_retriever = (
            retriever
            if not validated_config.reranker
            else self._compression_retriever(retriever)
        )

        return MultiSearchQueryRetriever(retriever=final_retriever)

    async def get_labels_and_hash(self) -> dict:
        """Returns a tuple containing any available source hash and their corresponding labels"""
        source_hashes = await get_faiss_indexes(self.manager.faiss_dir)

        # get the labels
        sources_repr = {}
        async for file in self.source_dir.glob("*.source"):
            if file.name in source_hashes:
                label = await file.read_text()
                sources_repr[file.name] = label

        return sources_repr
