# loaders.py
## TODO: Move towards an async operation instead of the current threading approach for concurrency
## Basically replacing execute_concurrently with aexecute_concurrently
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from queue import Queue
from threading import Lock, Thread
from time import sleep
from typing import List
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from constants import (
    app_name,
    logger,
)
from git import Blob, Repo
from gitlab import Gitlab, GitlabGetError
from gitlab.v4.objects import Group, Project
from langchain.schema import Document
from langchain_community.document_loaders import ConfluenceLoader, UnstructuredURLLoader
from langchain_community.document_loaders.base import BaseLoader
from pydantic import BaseModel, SecretStr
from sage.utils.exceptions import SourceException
from sage.utils.supports import (
    execute_concurrently,
    markdown_to_text_using_html2text,
)


class CustomConfluenceLoader(ConfluenceLoader):
    """Confluence loader with an override function"""

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
        documents: List[Document] = execute_concurrently(
            self._load, repo_data_list, result_type="extends", max_workers=10
        )
        return documents

    def process_groups(self, group: Group) -> List[Document]:
        """
        Helper to find all projects in a group and generate the corresponding documents
        """
        logger.debug(f"Fetching gitlab projects in group {group.name}")

        projects = self._get_all_projects(group)

        logger.debug(
            f"Fetched a total of {len(projects)} projects from gitlab group {group.name}"
        )

        repo_data_list = execute_concurrently(
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
            group_docs = execute_concurrently(
                self.process_groups, self.groups, result_type="extends"
            )
            documents.extend(group_docs)
        # process projects
        if self.projects:
            project_repos = execute_concurrently(self._clone_project, self.projects)
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
