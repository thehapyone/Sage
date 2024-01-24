from functools import lru_cache
import os
from anyio import Path as aPath
from pathlib import Path
import tempfile
from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
from urllib.parse import urljoin, urldefrag, urlparse
from queue import Queue
from pydantic import BaseModel, SecretStr
import requests
from time import sleep
from hashlib import md5

from bs4 import BeautifulSoup
from langchain_community.document_loaders.confluence import (
    ContentFormat,
    ConfluenceLoader,
)
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import UnstructuredFileLoader

from chainlit.types import AskFileResponse

from gitlab import Gitlab, GitlabGetError
from gitlab.v4.objects import Project
from gitlab.v4.objects import Group
from git import Blob, Repo

from constants import (
    validated_config,
    sources_config,
    core_config,
    EMBEDDING_MODEL,
    logger,
    app_name,
)
from utils.exceptions import SourceException
from utils.validator import ConfluenceModel, GitlabModel, Web, Files
from utils.supports import (
    markdown_to_text_using_html2text,
)


class CustomConfluenceLoader(ConfluenceLoader):
    """Confluence loader with an overide function"""

    def process_page(self, *args, **kwargs) -> Document:
        response = super().process_page(*args, **kwargs)

        if not kwargs.get("keep_markdown_format"):
            return response

        # parse the markdown response
        return Document(
            page_content=markdown_to_text_using_html2text(response.page_content),
            metadata=response.metadata,
        )


class RepoHandler(BaseModel):
    """
    A custom repo object that contains the required repo information
    """

    class Config:
        arbitrary_types_allowed = True

    git_url: str
    repo: Repo


class GitlabLoader(BaseLoader):
    """
    Load projects from a Gitlab source and then uses `git` to load
    the repository files.

    The Repository can be local on disk available at `repo_path`,
    or remote at `clone_url` that will be cloned to `repo_path`.
    Currently, supports only text files.

    Each document represents one file in the repository. The `path` points to
    the local Git repository, and the `branch` specifies the branch to load
    files from.
    """

    def __init__(
        self,
        base_url: str,
        private_token: SecretStr | None,
        groups: List[str] = [],
        projects: List[str] = [],
        ssl_verify: bool = True,
    ) -> None:
        self.base_url = base_url
        self.private_token = private_token
        self.ssl_verify = ssl_verify
        self.gitlab = self._initialize_gitlab()
        self.groups = self.validate_groups(groups)
        self.projects = self.validate_projects(projects)

    def _initialize_gitlab(self):
        gitlab = Gitlab(
            url=self.base_url,
            private_token=self.private_token.get_secret_value(),
            user_agent=app_name,
            ssl_verify=self.ssl_verify,
        )
        gitlab.auth()
        return gitlab

    def validate_groups(self, groups: List[str]) -> List[Group]:
        """
        Validates the provided groups are valid and returns the list of groups
        """
        group_list = []
        for group in groups:
            try:
                group_list.append(self.gitlab.groups.get(group))
            except GitlabGetError:
                raise SourceException(f"Group {group} does not exist")
        return group_list

    def validate_projects(self, projects: List[str]) -> List[Project]:
        """
        Function to validate the provided projects are valid

        Args:
            projects (List[str]): Project list

        Raises:
            SourceException: Project not found

        Returns:
            List[str]: A list of projects
        """
        project_list = []
        for project in projects:
            try:
                project_list.append(self.gitlab.projects.get(project))
            except GitlabGetError:
                raise SourceException(f"Project {project} does not exist")
        return project_list

    def _get_all_projects(self, group: Group):
        """
        Return all projects in the group including the projects in the subgroups.
        """
        projects = group.projects.list(get_all=True)
        # iterate over each subgroup and get their projects
        for subgroup in group.subgroups.list(get_all=True):
            full_subgroup = self.gitlab.groups.get(subgroup.id)
            projects += self._get_all_projects(full_subgroup)

        return projects

    def _clone_project(self, project: Project) -> RepoHandler | None:
        """
        Clone the project to a temporary directory.
        Returns the RepoHandler object
        """
        git_url = project.http_url_to_repo.replace(
            "https://", f"https://oauth2:{self.private_token.get_secret_value()}@"
        )

        logger.debug(f"Cloning gitlab project: {project.name}")

        try:
            repo_path = tempfile.mkdtemp()
            repo = Repo.clone_from(git_url, repo_path, branch=project.default_branch)
            handler = RepoHandler(git_url=project.http_url_to_repo, repo=repo)
            return handler
        except Exception as e:
            logger.warning(f"Error cloning project {project.name}: {e}")
            return None

    @staticmethod
    def _load(repo_data: RepoHandler | None):
        """
        Parse and load a repo as a document object and return a list of documents
        """

        docs: List[Document] = []

        if repo_data is None:
            return docs

        repo = repo_data.repo

        logger.debug(f"Processing local git repo at: {repo_data.git_url}")

        for item in repo.tree().traverse():
            if not isinstance(item, Blob):
                continue

            file_path = os.path.join(repo.working_dir, item.path)

            ignored_files = repo.ignored([file_path])  # type: ignore
            if len(ignored_files):
                continue

            rel_file_path = os.path.relpath(file_path, repo.working_dir)
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                    file_type = os.path.splitext(item.name)[1]

                    # loads only text files
                    try:
                        text_content = content.decode("utf-8")
                    except UnicodeDecodeError:
                        continue

                    metadata = {
                        "url": repo_data.git_url,
                        "branch": repo.active_branch.name,
                        "source": rel_file_path,
                        "file_name": item.name,
                        "file_type": file_type,
                    }
                    doc = Document(page_content=text_content, metadata=metadata)
                    docs.append(doc)
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
        return docs

    def _build_documents(
        self, repo_data_list: List[RepoHandler | None]
    ) -> List[Document]:
        """
        Function that helps to process the git repos and generate the required documents
        """
        documents: List[Document] = self.execute_concurrently(
            self._load, repo_data_list, result_type="extends", max_workers=10
        )
        return documents

    @staticmethod
    def execute_concurrently(
        func: Callable, items: List, result_type: str = "append", max_workers: int = 10
    ) -> List:
        """
        Executes a function concurrently on a list of items.

        Args:
            func (Callable): The function to execute. This function should accept a single argument.
            items (List): The list of items to execute the function on.
            result_type (str): The type of result to return. Can be "append" or "return". Defaults to "append".
            max_workers (int, optional): The maximum number of workers to use. Defaults to 10.

        Returns:
            List: A list of the results of the function execution.
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, item) for item in items]
            for future in as_completed(futures):
                if result_type == "append":
                    results.append(future.result())
                else:
                    results.extend(future.result())
        return results

    def process_groups(self, group: Group) -> List[Document]:
        """
        Helper to find all projects in a group and generate the corresponding documents
        """
        logger.debug(f"Fetching gitlab projects in group {group.name}")

        projects = self._get_all_projects(group)

        logger.debug(
            f"Fetched a total of {len(projects)} projects from gitlab group {group.name}"
        )

        repo_data_list = self.execute_concurrently(
            self._clone_project, projects, max_workers=50
        )

        return self._build_documents(repo_data_list)

    def load(self) -> List[Document]:
        """
        Loads documents for groups or projects

        Returns:
            List[Document]: List of documents
        """
        documents: List[Document] = []
        # process groups
        if self.groups:
            group_docs = self.execute_concurrently(
                self.process_groups, self.groups, result_type="extends"
            )
            documents.extend(group_docs)
        # process projects
        if self.projects:
            project_repos = self.execute_concurrently(
                self._clone_project, self.projects
            )
            documents.extend(self._build_documents(project_repos))
        return documents


class WebLoader(UnstructuredURLLoader):
    """
    An adapted web loader that is capable of using unstructured
    and supports recursive search of a given url
    """

    def __init__(self, nested: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nested = nested
        self.ssl_verify = kwargs.get("ssl_verify", True)

    def _load(self, url: str) -> List[Document]:
        """Load documents"""
        from unstructured.partition.auto import partition
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()

        try:
            if self._UnstructuredURLLoader__is_non_html_available():
                if self._UnstructuredURLLoader__is_headers_available_for_non_html():
                    elements = partition(
                        url=url, headers=self.headers, **self.unstructured_kwargs
                    )
                else:
                    elements = partition(url=url, **self.unstructured_kwargs)
            else:
                if self._UnstructuredURLLoader__is_headers_available_for_html():
                    elements = partition_html(
                        url=url, headers=self.headers, **self.unstructured_kwargs
                    )
                else:
                    elements = partition_html(url=url, **self.unstructured_kwargs)

            if self.mode == "single":
                text = "\n\n".join([str(el) for el in elements])
                metadata = {"source": url}
                return [Document(page_content=text, metadata=metadata)]

            for element in elements:
                metadata = element.metadata.to_dict()
                metadata["category"] = element.category
                docs.append(Document(page_content=str(element), metadata=metadata))

        except Exception as e:
            if self.continue_on_failure:
                logger.error(f"Error fetching or processing {url}, exception: {e}")
            else:
                raise e

        return docs

    @staticmethod
    def normalize_url(url):
        """Helps to normalize urls by removing ending slash"""
        if url.endswith("/"):
            return url[:-1]
        return url

    @lru_cache(maxsize=1)
    def get_base_path(self, url):
        """Get the base path of a URL, excluding any file names."""
        parsed = urlparse(url)
        path_parts = parsed.path.rsplit("/", 1)
        if "." in path_parts[-1]:
            return self.normalize_url(
                parsed.scheme + "://" + parsed.netloc + "/".join(path_parts[:-1])
            )
        else:
            return self.normalize_url(url)

    def find_links(self, base_url: str) -> List[Document]:
        """
        Helps to find child links from a given link source

        Args:
            base_url (str): A valid URL

        Returns:
            List[Document]: A list of documents
        """

        def worker():
            """Worker function to process URLs from the queue"""
            while True:
                with lock:
                    if not to_visit_links.empty():
                        url, depth = to_visit_links.get()
                    else:
                        break
                extract_links_docs(url, depth)

        def add_child_links(url: str, depth: int) -> None:
            """Find and add child links to the list"""
            try:
                response = session.get(url, timeout=10, verify=self.ssl_verify)
                response.raise_for_status()
            except (
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
                requests.exceptions.SSLError,
            ) as e:
                if self.continue_on_failure:
                    logger.error(f"Error occurred: {e}, url: {url}")
                    return
                else:
                    raise e
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"An unexpected error has occurred: {e}, url: {url}")
                    return
                else:
                    raise e

            soup = BeautifulSoup(response.text, "lxml")

            for a_tag in soup.find_all("a", href=True):
                link = a_tag["href"]
                if link.startswith(("mailto:", "javascript:", "#")) or link.endswith(
                    (".png", ".svg", ".jpg", ".jpeg", ".gif")
                ):
                    continue
                absolute_link = urljoin(url, link)

                with lock:
                    if absolute_link not in visited_links:
                        to_visit_links.put((absolute_link, depth + 1))

        def extract_links_docs(url, depth):
            """
            Extracts all unique links from a webpage and adds them to the queue.
            Also, it extracts the page content for the url
            """
            url = self.normalize_url(urldefrag(url)[0])

            with lock:
                if url in visited_links or not url.startswith(
                    self.get_base_path(base_url)
                ):
                    return
                visited_links.add(url)

            if depth > 10:
                logger.warning(f"Max depth reached - {url}")
                return

            logger.debug(f"Scraping {url}")

            documents = self._load(url)
            with docs_lock:
                docs.extend(documents)

            add_child_links(url, depth)

            sleep(0.5)

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100, pool_maxsize=100, max_retries=3
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) \
                    Chrome/89.0.4389.82 Safari/537.36"
        }

        visited_links = set()
        to_visit_links = Queue()
        lock = Lock()
        base_url = self.normalize_url(base_url)
        to_visit_links.put((base_url, 0))

        threads = []
        docs_lock = Lock()
        docs: List[Document] = list()

        for _ in range(5):
            t = Thread(target=worker)
            t.start()
            threads.append(t)

        while any(t.is_alive() for t in threads):
            with lock:
                if to_visit_links.qsize() >= 2 and len(threads) < 10:
                    t = Thread(target=worker)
                    t.start()
                    threads.append(t)
            sleep(0.5)

        for t in threads:
            t.join()

        logger.debug(f"Total links detected: {len(visited_links)}")

        return docs

    def load(self) -> List[Document]:
        """
        Loads and parse the URLS into documents

        Returns:
            _type_: List[Documents]
        """
        docs: List[Document] = list()

        if self.nested:
            with ThreadPoolExecutor() as executor:
                docs_futures = [
                    executor.submit(self.find_links, url) for url in self.urls
                ]
                for future in docs_futures:
                    docs.extend(future.result())
        else:
            for url in self.urls:
                docs.extend(self._load(url))
        return docs


class Source:
    # TODO: Adds support for batching loading of the documents when generating the Faiss index. As it's easy to reach API throttle limits with OPENAI
    # TODO: Old sources metadata are not removed when the source change causing issue if old sources are used again as the source will not loaded because the metadata still exists

    _instance = None
    source_dir = Path(core_config.data_dir) / "sources"
    _retriever_args = {"k": sources_config.top_k}
    """Custom retriever search args"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.source_dir.mkdir(exist_ok=True)
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

    def _get_source_metadata_path(self, source_hash: str) -> Path:
        return self.source_dir / source_hash

    def _save_source_metadata(self, source_hash: str):
        """Save the source metadata to disk"""
        metadata = self._get_source_metadata_path(source_hash)
        metadata.write_text("True")

    def _source_exist_locally(self, source_hash: str):
        """Returns whether the given source exist or not"""
        return self._get_source_metadata_path(source_hash).exists()

    def _get_source_hash(self, source_type: str, name: str) -> str:
        """Generate a hash for a given source"""
        return f"{source_type}-{self._get_hash(name.lower())}.source"

    def _append_to_refresh_list(
        self, source_type: str, identifier: str, identifier_key: str
    ):
        """Add source to the refresh list"""
        source_hash = self._get_source_hash(source_type, identifier)
        if not self._source_exist_locally(source_hash):
            self.source_refresh_list.append(
                {
                    "hash": source_hash,
                    "source_type": source_type,
                    "identifier": identifier,
                    "identifier_type": identifier_key,
                }
            )

    def _check_source(
        self,
        source_type: str,
        source_data: ConfluenceModel | GitlabModel | Web,
        identifier_key: str,
    ):
        """Check if a source exist otherwise add to the refresh list"""
        for identifier in getattr(source_data, identifier_key, []):
            self._append_to_refresh_list(source_type, identifier, identifier_key)

    def check_sources_exist(self):
        """Checks if sources in the config file exists locally"""
        for source_type, source_data in vars(sources_config).items():
            if not source_data:
                continue

            if isinstance(source_data, ConfluenceModel):
                self._check_source(source_type, source_data, "spaces")
            elif isinstance(source_data, GitlabModel):
                self._check_source(source_type, source_data, "groups")
                self._check_source(source_type, source_data, "projects")
            elif isinstance(source_data, Web):
                self._check_source(source_type, source_data, "links")

    def _create_and_save_db(
        self, source_hash: str, documents: List[Document], save_db: bool = True
    ) -> FAISS | None:
        """Creates and save a vector store index DB to file"""
        logger.debug(f"Creating a vector store for source with hash - {source_hash}")

        # Create vector index
        db = FAISS.from_documents(documents=documents, embedding=EMBEDDING_MODEL)

        if save_db:
            # Save DB to source directory
            dir_path = self.source_dir / "faiss"
            db.save_local(str(dir_path), source_hash)
            logger.debug(
                f"Succesfully created and saved vector store for source with hash - {source_hash}"
            )
            return
        logger.debug(
            f"Succesfully created the vector store for source with hash - {source_hash}"
        )
        return db

    @staticmethod
    def splitter() -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300, length_function=len
        )

    def _add_confluence(self, hash: str, source: ConfluenceModel, space: str):
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
                f"An error has occured while loading confluence source: {str(error)}"
            )

        self._create_and_save_db(
            source_hash=hash,
            documents=self.splitter().split_documents(confluence_documents),
        )

    def _add_gitlab_source(
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
                f"An error has occured while loading gitlab source: {str(error)}"
            )

        self._create_and_save_db(
            source_hash=hash,
            documents=self.splitter().split_documents(gitlab_documents),
        )

    def _add_web_source(self, hash: str, source: Web, link: str):
        """
        Adds and saves a web data source

        Args:
            hash (str): The source hash
            source (Web): Web data model
            link (str): The source link value

        Raises:
            SourceException: Exception rasied interating with web links
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
                f"An error has occured while loading web source: {str(error)}"
            )

        self._create_and_save_db(
            source_hash=hash, documents=self.splitter().split_documents(web_documents)
        )

    def _add_files_source(
        self, hash: str, source: Files, path: str, save_db: bool = True
    ) -> FAISS | None:
        """
        Adds and saves a files data source

        Args:
            hash (str): The source hash
            source (Files): Files data model

        Raises:
            SourceException: Exception rasied interating with the files
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
                f"An error has occured while loading file source: {str(error)}"
            )

        db = self._create_and_save_db(
            source_hash=hash,
            documents=self.splitter().split_documents(file_documents),
            save_db=save_db,
        )
        return db

    def add_source(
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
            if source_type == "confluence":
                self._add_confluence(
                    hash=hash, source=sources_config.confluence, space=identifier
                )
            elif source_type == "web":
                self._add_web_source(
                    hash=hash, source=sources_config.web, link=identifier
                )
            elif source_type == "gitlab":
                if identifier_type == "groups":
                    self._add_gitlab_source(
                        hash=hash, source=sources_config.gitlab, groups=[identifier]
                    )
                else:
                    self._add_gitlab_source(
                        hash=hash, source=sources_config.gitlab, projects=[identifier]
                    )
            else:
                raise SourceException(f"Unknown source type: {source_type}")
        except Exception as e:
            logger.error(f"An error has occurred processing source {source_ref}")
            logger.error(str(e))
            logger.error("Source will retry next re-run")
        else:
            self._save_source_metadata(hash)
            logger.info(f"Done with source {source_ref}")

    def run(self) -> None:
        """
        Starts a new thread for processing each source in self.source_refresh_list.
        """
        self.check_sources_exist()

        if len(self.source_refresh_list) == 0:
            logger.info("No changes to sources")
            return

        threads = []
        for source in self.source_refresh_list:
            thread = Thread(target=self.add_source, kwargs=source)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    async def _aget_faiss_indexes(self) -> str | List[str]:
        """Returns a list of all available faiss indexes"""
        dir_path = aPath(self.source_dir / "faiss")

        indexes: List[str] = []

        async for file in dir_path.glob("*.faiss"):
            indexes.append(file.stem)

        return str(dir_path), indexes

    def _get_faiss_indexes(self) -> str | List[str]:
        """Returns a list of all available faiss indexes"""
        dir_path = Path(self.source_dir / "faiss")

        indexes: List[str] = []

        for file in dir_path.glob("*.faiss"):
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
        """Loads a compresion retriever"""

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
        dbs: List[FAISS] = []
        for file in files:
            file_source = Files(paths=[file.path])
            db = self._add_files_source(
                hash=file.id, source=file_source, path=file.path, save_db=False
            )
            dbs.append(db)
        faiss_db = self._combine_dbs(dbs)
        retriever = faiss_db.as_retriever(search_kwargs=self._retriever_args)

        if not validated_config.reranker:
            return retriever
        else:
            return self._compression_retriever(retriever)

    def _load_retriever(self, db_path: str, indexes: List[str]):
        # Loads retriever
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

    async def aload(
        self,
    ) -> Optional[VectorStoreRetriever | ContextualCompressionRetriever]:
        """
        Returns either a retriever model from the FAISS vector indexes or compression based retriever model
        """

        self.run()

        db_path, indexes = await self._aget_faiss_indexes()

        _retriever = self._load_retriever(db_path, indexes)

        if not validated_config.reranker:
            return _retriever
        else:
            return self._compression_retriever(_retriever)

    def load(self) -> Optional[VectorStoreRetriever | ContextualCompressionRetriever]:
        """
        Returns either a retriever model from the FAISS vector indexes or compression based retriever model
        """

        self.run()

        db_path, indexes = self._get_faiss_indexes()

        _retriever = self._load_retriever(db_path, indexes)

        if not validated_config.reranker:
            return _retriever
        else:
            return self._compression_retriever(_retriever)
