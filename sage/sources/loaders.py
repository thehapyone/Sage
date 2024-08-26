# loaders.py
import asyncio
import os
import tempfile
from functools import lru_cache
from typing import List, Set
from urllib.parse import quote, urljoin, urlparse, urlunparse

import aiofiles
from aiohttp import ClientSession
from asyncer import asyncify
from bs4 import BeautifulSoup
from git import Blob, Repo
from gitlab import Gitlab, GitlabGetError
from gitlab.v4.objects import Group, Project
from langchain.schema import Document
from langchain_community.document_loaders import ConfluenceLoader, UnstructuredURLLoader
from langchain_community.document_loaders.base import BaseLoader
from pydantic import BaseModel, SecretStr

from sage.constants import (
    app_name,
    logger,
)
from sage.utils.exceptions import SourceException
from sage.utils.supports import (
    aexecute_concurrently,
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
        max_concurrency: int = 10,
    ) -> None:
        self.base_url = base_url
        self.private_token = private_token
        self.ssl_verify = ssl_verify
        self.max_concurrency = max_concurrency
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

    async def _clone_project(self, project: Project) -> RepoHandler | None:
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
            repo = await asyncify(Repo.clone_from)(
                git_url, repo_path, branch=project.default_branch
            )
            handler = RepoHandler(git_url=project.http_url_to_repo, repo=repo)
            return handler
        except Exception as e:
            logger.warning(f"Error cloning project {project.name}: {e}")
            return None

    @staticmethod
    async def _load(repo_data: RepoHandler | None):
        """
        Asynchronously parse and load a repo as a document object and return a list of documents
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

            ignored_files = await asyncify(repo.ignored)([file_path])
            if len(ignored_files):
                continue

            rel_file_path = os.path.relpath(file_path, repo.working_dir)

            try:
                async with aiofiles.open(file_path, "rb") as f:
                    content = await f.read()
                    file_type = os.path.splitext(item.name)[1]

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

    async def _build_documents(
        self, repo_data_list: List[RepoHandler | None]
    ) -> List[Document]:
        """
        Function that helps to process the git repos and generate the required documents
        """
        documents: List[Document] = await aexecute_concurrently(
            self._load,
            repo_data_list,
            result_type="extends",
            max_workers=self.max_concurrency,
        )
        return documents

    async def process_groups(self, group: Group) -> List[Document]:
        """
        Helper to find all projects in a group and generate the corresponding documents
        """
        logger.debug(f"Fetching gitlab projects in group {group.name}")

        projects = self._get_all_projects(group)

        logger.debug(
            f"Fetched a total of {len(projects)} projects from gitlab group {group.name}"
        )

        repo_data_list = await aexecute_concurrently(
            self._clone_project, projects, max_workers=self.max_concurrency
        )

        return await self._build_documents(repo_data_list)

    async def load(self) -> List[Document]:
        """
        Loads documents for groups or projects

        Returns:
            List[Document]: List of documents
        """
        documents: List[Document] = []
        # process groups
        if self.groups:
            group_docs = await aexecute_concurrently(
                self.process_groups,
                self.groups,
                result_type="extends",
                max_workers=self.max_concurrency,
            )
            documents.extend(group_docs)
        # process projects
        if self.projects:
            project_repos = await aexecute_concurrently(
                self._clone_project, self.projects, max_workers=self.max_concurrency
            )
            documents.extend(await self._build_documents(project_repos))
        return documents


class WebLoader(UnstructuredURLLoader):
    """
    An adapted web loader that is capable of using unstructured
    and supports recursive search of a given url
    """

    def __init__(self, nested: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nested = nested
        self.max_concurrency: int = kwargs.get("max_concurrency", 10)
        self.max_depth: int = kwargs.get("max_depth", 10)
        self.ssl_verify = kwargs.get("ssl_verify", True)

    async def _load(self, url: str) -> List[Document]:
        """Load documents"""
        from unstructured.partition.auto import partition
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()
        logger.debug(f"Scraping {url}")

        try:
            if self._UnstructuredURLLoader__is_non_html_available():
                if self._UnstructuredURLLoader__is_headers_available_for_non_html():
                    elements = await asyncify(partition)(
                        url=url, headers=self.headers, **self.unstructured_kwargs
                    )
                else:
                    elements = await asyncify(partition)(
                        url=url, **self.unstructured_kwargs
                    )
            else:
                if self._UnstructuredURLLoader__is_headers_available_for_html():
                    elements = await asyncify(partition_html)(
                        url=url, headers=self.headers, **self.unstructured_kwargs
                    )
                else:
                    elements = await asyncify(partition_html)(
                        url=url, **self.unstructured_kwargs
                    )

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

    @staticmethod
    def get_clean_url(url) -> str:
        """Returns the clean url with the # and extras"""
        parsed_url = urlparse(url)
        # Reconstruct the URL without the fragment
        main_url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                "",
            )
        )
        return main_url

    async def find_links(self, base_url: str) -> List[Document]:
        """
        Helps to find child links from a given link source

        Args:
            base_url (str): A valid URL

        Returns:
            List[Document]: A list of documents
        """

        async def fetch(url: str, session: ClientSession, depth: int):
            """Returns the raw data from a given URL"""
            async with semaphore:
                try:
                    async with session.get(
                        url, timeout=10, ssl=self.ssl_verify
                    ) as response:
                        response.raise_for_status()
                        text = await response.text()
                        return text, url, depth
                except Exception as e:
                    if self.continue_on_failure:
                        logger.error(f"Error occurred: {e}, url: {url}")
                    else:
                        raise e

        async def parse_links(text: str, url: str, depth: int):
            """Extract child links from a given URL data and appends to a the task list"""
            if depth > self.max_depth:
                logger.warning(f"Max depth reached - {url}")
                return

            soup = BeautifulSoup(text, "lxml")
            for a_tag in soup.find_all("a", href=True):
                link = a_tag["href"]
                if link.startswith(("mailto:", "javascript:", "#")) or link.endswith(
                    (".png", ".svg", ".jpg", ".jpeg", ".gif")
                ):
                    continue
                absolute_link = urljoin(url, link)
                clean_link = self.get_clean_url(absolute_link)
                if clean_link not in visited_links and clean_link.startswith(
                    self.get_base_path(base_url)
                ):
                    visited_links.add(clean_link)
                    await tasks.put(
                        asyncio.create_task(fetch(clean_link, session, depth + 1))
                    )

        async def worker():
            """Worker node for fetching all urls"""
            tasks_tracker = 0
            while not tasks.empty():
                current_tasks = []
                while not tasks.empty():
                    current_tasks.append(await tasks.get())
                    tasks_tracker = tasks_tracker + 1
                fetch_results = await asyncio.gather(*current_tasks)

                parse_child_tasks = [
                    asyncio.create_task(parse_links(*fetch_result))
                    for fetch_result in fetch_results
                    if fetch_result
                ]

                await asyncio.gather(*parse_child_tasks)
            # mark all tasks as done
            for _ in range(tasks_tracker):
                tasks.task_done()

        # Configures a semaphore for concurrency
        semaphore = asyncio.Semaphore(self.max_concurrency)

        visited_links: Set[str] = set()
        tasks: asyncio.Queue = asyncio.Queue()

        async with ClientSession() as session:
            # Set up the session headers here

            session.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
                }
            )

            if self.headers:
                session.headers.update(self.headers)

            base_url = quote(self.normalize_url(base_url), safe="/:")
            visited_links.add(base_url)
            await tasks.put(asyncio.create_task(fetch(base_url, session, 0)))

            await worker()

        logger.debug(f"Total links detected: {len(visited_links)}")

        # Now we loads the data into a document objects
        documents = await aexecute_concurrently(
            self._load,
            visited_links,
            max_workers=self.max_concurrency,
            result_type="extend",
        )

        logger.debug(f"Total documents created: {len(documents)}")
        return documents

    async def load(self) -> List[Document]:
        """
        Loads and parse the URLS into documents

        Returns:
            _type_: List[Documents]
        """
        if self.nested:
            docs = await aexecute_concurrently(
                self.find_links,
                self.urls,
                max_workers=self.max_concurrency,
                result_type="extend",
            )
        else:
            docs = await aexecute_concurrently(
                self._load, self.urls, max_workers=self.max_concurrency
            )

        return docs
