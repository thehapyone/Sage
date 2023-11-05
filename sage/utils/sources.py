from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from threading import Lock, Thread
from urllib.parse import urljoin, urldefrag
from queue import Queue
import requests
from time import sleep
from hashlib import md5

from bs4 import BeautifulSoup
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.document_loaders import UnstructuredURLLoader

from utils.validator import SourceData, ConfluenceModel, GitlabModel, Web
from constants import sources_config, core_config, EMBEDDING_MODEL, logger
from utils.exceptions import SourceException


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
        except Exception as e:
            if self.continue_on_failure:
                logger.error(
                    f"Error fetching or processing {url}, exception: {e}")
            else:
                raise e

        if self.mode == "single":
            text = "\n\n".join([str(el) for el in elements])
            metadata = {"source": url}
            return [Document(page_content=text, metadata=metadata)]

        # self.mode == elements
        docs: List[Document] = list()
        for element in elements:
            metadata = element.metadata.to_dict()
            metadata["category"] = element.category
            docs.append(Document(page_content=str(element), metadata=metadata))

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

            logger.info(f"Scraping {url}")

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

    def _get_source_metadata(self, source: SourceData | Web) -> str:
        """Generate a unique metadata for each source"""
        if isinstance(source, ConfluenceModel):
            spaces = "-".join(source.spaces)
            return f"confluence-{spaces}.source"
        elif isinstance(source, GitlabModel):
            spaces = "-".join(source.spaces)
            return f"gitlab-{spaces}.source"
        else:
            hash_links = self._get_hash("-".join(source.links))
            return f"web-{hash_links}.source"

    def _get_source_metadata_path(self, source: SourceData | Web) -> Path:
        return self.source_dir / self._get_source_metadata(source)

    def _save_source_metadata(self, source: SourceData | Web):
        """Save the source metadata to disk"""
        metadata = self._get_source_metadata_path(source)
        metadata.write_text("True")

    def _source_exist_locally(self, source: SourceData | Web):
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
                api_key=source.password
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

    def add_source(self, id: str, data: SourceData | Web) -> None:
        logger.info(f"Processing source {id}...")
        if isinstance(data, ConfluenceModel):
            self._add_confluence(data)
        elif isinstance(data, Web):
            self._add_web_source(data)

        self._save_source_metadata(data)
        logger.info(f"Done with source {id}")

    def run(self):
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
