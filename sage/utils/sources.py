import os
from pathlib import Path
import tempfile
from typing import Callable, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
from urllib.parse import urljoin, urldefrag
from queue import Queue
from pydantic import BaseModel, SecretStr
import requests
from time import sleep
from hashlib import md5

from bs4 import BeautifulSoup
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders.base import BaseLoader
from gitlab import Gitlab, GitlabGetError
from gitlab.v4.objects import Project
from gitlab.v4.objects import Group
from git import Blob, Repo

from constants import sources_config, core_config, EMBEDDING_MODEL, logger, app_name
from utils.exceptions import SourceException
from utils.validator import ConfluenceModel, GitlabModel, Web


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
            ssl_verify: bool = True
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
            ssl_verify=self.ssl_verify
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

    def _get_all_projects(self, group):
        """  
        Return all projects in the group including the projects in the subgroups.  
        """
        projects = group.projects.list(get_all=True)
        # iterate over each subgroup and get their projects
        for subgroup in group.subgroups.list(get_all=True):
            full_subgroup = self.gitlab.groups.get(subgroup.id)
            projects += self._get_all_projects(full_subgroup)

        return projects

    def _clone_project(self, project: Project) -> RepoHandler:
        """
        Clone the project to a temporary directory.
        Returns the RepoHandler object
        """
        git_url = project.http_url_to_repo.replace(
            'https://', f'https://oauth2:{self.private_token.get_secret_value()}@')

        try:
            repo_path = tempfile.mkdtemp()
            repo = Repo.clone_from(
                git_url, repo_path, branch=project.default_branch)
            handler = RepoHandler(
                git_url=project.http_url_to_repo,
                repo=repo
            )
            return handler
        except Exception as e:
            logger.error(f"Error cloning project {project.name}: {e}")
            raise e

    @staticmethod
    def _load(repo_data: RepoHandler):
        """
        Parse and load a repo as a document object and return a list of documents
        """

        docs: List[Document] = []

        repo = repo_data.repo

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
                    doc = Document(page_content=text_content,
                                   metadata=metadata)
                    docs.append(doc)
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
        return docs

    def _build_documents(self, repo_data_list: List[RepoHandler]) -> List[Document]:
        """
        Function that helps to process the git repos and generate the required documents
        """
        documents: List[Document] = self.execute_concurrently(
            self._load, repo_data_list, result_type="extends", max_workers=10)
        return documents

    @staticmethod
    def execute_concurrently(func: Callable, items: List, result_type: str = "append", max_workers: int = 10) -> List:
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
        projects = self._get_all_projects(group)

        repo_data_list = self.execute_concurrently(
            self._clone_project, projects, max_workers=50)

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
                self.process_groups, self.groups, result_type="extends")
            documents.extend(group_docs)
        # process projects
        if self.projects:
            project_repos = self.execute_concurrently(
                self._clone_project, self.projects)
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
                    elements = partition_html(
                        url=url, **self.unstructured_kwargs)

            if self.mode == "single":
                text = "\n\n".join([str(el) for el in elements])
                metadata = {"source": url}
                return [Document(page_content=text, metadata=metadata)]

            # self.mode == elements
            for element in elements:
                metadata = element.metadata.to_dict()
                metadata["category"] = element.category
                docs.append(Document(page_content=str(
                    element), metadata=metadata))

        except Exception as e:
            if self.continue_on_failure:
                logger.error(
                    f"Error fetching or processing {url}, exception: {e}")
            else:
                raise e

        return docs

    @staticmethod
    def normalize_url(url):
        """Helps to normalize urls by removing ending slash"""
        if url.endswith('/'):
            return url[:-1]
        return url

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
                response = session.get(url, timeout=10)
                response.raise_for_status()
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout, requests.exceptions.RequestException,
                    requests.exceptions.SSLError) as e:
                if self.continue_on_failure:
                    logger.error(f"Error occurred: {e}, url: {url}")
                    return
                else:
                    raise e
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(
                        f"An unexpected error has occurred: {e}, url: {url}")
                    return
                else:
                    raise e

            soup = BeautifulSoup(response.text, 'lxml')

            for a_tag in soup.find_all('a', href=True):
                link = a_tag['href']
                if link.startswith(('mailto:', 'javascript:', '#')) \
                        or link.endswith(('.png', '.svg', '.jpg', '.jpeg', '.gif')):
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
                if url in visited_links or base_url not in url:
                    return
                visited_links.add(url)

            if depth > 10:
                logger.warning(f"Max depth reached - {url}")
                return

            # logger.info(f"Scraping {url}")

            documents = self._load(url)
            with docs_lock:
                docs.extend(documents)

            add_child_links(url, depth)

            sleep(0.5)

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100, pool_maxsize=100, max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) \
                    Chrome/89.0.4389.82 Safari/537.36'
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

        logger.debug(f'Total links detected: {len(visited_links)}')

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
                docs_futures = [executor.submit(
                    self.find_links, url) for url in self.urls]
                for future in docs_futures:
                    docs.extend(future.result())
        else:
            for url in self.urls:
                docs.extend(self._load(url))
        return docs


class Source:
    # TODO: Adds support for batching loading of the documents when generating the Faiss index. As it's easy to reach API throttle limits with OPENAI
    # TODO: Improve a way to avoid reloading existing documents when the spaces, groups, and projects of a source changes
    # TODO: Old sources metadata are not removed when the source change causing issue if old sources are used again as the source will not loaded because the metadata still exists

    _instance = None
    source_dir = Path(core_config.data_dir) / "sources"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.source_dir.mkdir(exist_ok=True)
        self.source_refresh_list: List[dict] = list()

    @staticmethod
    def _get_hash(input: str) -> str:
        return md5(input.encode()).hexdigest()

    def _get_source_metadata(self, source: ConfluenceModel | GitlabModel | Web) -> str:
        """Generate a unique metadata for each source"""
        if isinstance(source, ConfluenceModel):
            spaces = "-".join(source.spaces)
            return f"confluence-{spaces}.source"
        elif isinstance(source, GitlabModel):
            if source.groups and source.projects:
                hash_links = self._get_hash(
                    "-".join(source.groups + source.projects))
            elif source.groups:
                hash_links = self._get_hash("-".join(source.groups))
            else:
                hash_links = self._get_hash("-".join(source.projects))
            return f"gitlab-{hash_links}.source"
        else:
            hash_links = self._get_hash("-".join(source.links))
            return f"web-{hash_links}.source"

    def _get_source_metadata_path(self, source: ConfluenceModel | GitlabModel | Web) -> Path:
        return self.source_dir / self._get_source_metadata(source)

    def _save_source_metadata(self, source: ConfluenceModel | GitlabModel | Web):
        """Save the source metadata to disk"""
        metadata = self._get_source_metadata_path(source)
        metadata.write_text("True")

    def _source_exist_locally(self, source: ConfluenceModel | GitlabModel | Web):
        """Returns whether the given source exist or not"""
        return self._get_source_metadata_path(source).exists()

    def check_sources_exist(self):
        for source_name, source_data in vars(sources_config).items():
            if source_data is None:
                continue
            if not self._source_exist_locally(source_data):
                self.source_refresh_list.append({
                    "id": source_name,
                    "data": source_data
                })

    def _create_and_save_db(self, source_name: str, documents: List[Document]) -> None:
        """Creates and save a vector store index DB to file"""
        # Create vector index
        db = FAISS.from_documents(
            documents=documents,
            embedding=EMBEDDING_MODEL)

        # Save DB to source directory
        dir_path = self.source_dir / "faiss"
        db.save_local(str(dir_path), source_name)

    @staticmethod
    def splitter() -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len
        )

    def _add_confluence(self, source: ConfluenceModel):
        """
        Adds and saves a confluence data source

        Args:
            source (ConfluenceModel): Confluence data model

        Raises:
            SourceException: HTTP Error communicating with confluence
        """
        try:
            loader = ConfluenceLoader(
                url=source.server,
                username=source.username,
                api_key=source.password.get_secret_value()
            )

            confluence_documents = []

            for space in source.spaces:
                documents = loader.load(
                    space_key=space,
                    include_attachments=False,
                    limit=200
                )
                confluence_documents.extend(documents)
        except Exception as error:
            raise SourceException(
                f"An error has occured while loading confluence source: {str(error)}")

        self._create_and_save_db(
            source_name="confluence",
            documents=self.splitter().split_documents(confluence_documents)
        )

    def _add_gitlab_source(self, source: GitlabModel):
        """
        Adds and saves a gitlab data source.
        """
        try:
            loader = GitlabLoader(
                base_url=source.server,
                groups=source.groups,
                projects=source.projects,
                private_token=source.password,
                ssl_verify=True
            )

            gitlab_documents = loader.load()

        except Exception as error:
            raise SourceException(
                f"An error has occured while loading gitlab source: {str(error)}")

        self._create_and_save_db(
            source_name="gitlab",
            documents=self.splitter().split_documents(gitlab_documents)
        )

    def _add_web_source(self, source: Web):
        """
        Adds and saves a web data source

        Args:
            source (Web): Web data model

        Raises:
            SourceException: Exception rasied interating with web links
        """
        try:

            loader = WebLoader(
                nested=source.nested,
                urls=source.links
            )

            web_documents = loader.load()

            if len(web_documents) < len(source.links):
                raise SourceException(
                    f"The total documents {len(web_documents)} parsed is less than the number of source links provided")

        except Exception as error:
            raise SourceException(
                f"An error has occured while loading web source: {str(error)}")

        self._create_and_save_db(
            source_name="web",
            documents=self.splitter().split_documents(web_documents)
        )

    def add_source(self, id: str, data: GitlabModel | ConfluenceModel | Web) -> None:
        """
        Adds and saves a data source

        Args:
            id (str): Source id
            data (GitlabModel | ConfluenceModel | Web): Source data

        Raises:
            SourceException: Raised if the source type is unknown
        """
        logger.info(f"Processing source {id}...")
        if isinstance(data, ConfluenceModel):
            self._add_confluence(data)
        elif isinstance(data, Web):
            self._add_web_source(data)
        elif isinstance(data, GitlabModel):
            self._add_gitlab_source(data)
        else:
            raise SourceException(
                f"Unknown source type: {type(data)}")

        self._save_source_metadata(data)
        logger.info(f"Done with source {id}")

    def run(self) -> None:
        """
        Starts a new thread for processing each source in self.source_refresh_list.
        """
        self.check_sources_exist()

        if len(self.source_refresh_list) == 0:
            logger.info("No changes to sources")

        threads = []
        for source in self.source_refresh_list:
            thread = Thread(target=self.add_source, kwargs=source)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
