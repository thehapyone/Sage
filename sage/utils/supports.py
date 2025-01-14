import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Coroutine, List, Optional, Sequence, Tuple

from asyncer import asyncify
from html2text import HTML2Text
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.docstore.base import AddableMixin
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from litellm import aembedding, arerank, embedding, rerank
from markdown import markdown
from pydantic import PrivateAttr, model_validator
from sentence_transformers import CrossEncoder

from sage.validators.config_toml import Config

text_maker = HTML2Text()
text_maker.ignore_links = False
text_maker.ignore_images = True
text_maker.ignore_emphasis = True

app_name = "sage.ai"


class LiteLLMEmbeddings(Embeddings):
    """An embedding class powered by LiteLLM"""

    def __init__(self, model: str, timeout: int = 600, dimensions: int = None):
        """Initialize the LiteLLM Embeddings"""
        self.model = model
        self.timeout = timeout
        self.dimensions = dimensions
        self.api_base = os.getenv("EMBEDDING_API_BASE")
        self.api_key = os.getenv("EMBEDDING_API_KEY")
        self.api_type = os.getenv("EMBEDDING_API_TYPE")
        self.api_version = os.getenv("EMBEDDING_API_VERSION")

        self.embedding_args = {
            "model": self.model,
            "timeout": self.timeout,
            "dimensions": self.dimensions,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "api_type": self.api_type,
            "api_version": self.api_version,
        }

    def embed_query(self, text: str) -> List[float]:
        # Synchronous embedding of a single query
        response = embedding(input=[text], **self.embedding_args)
        return response["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Synchronous embedding of multiple documents
        response = embedding(input=texts, **self.embedding_args)
        return [item["embedding"] for item in response["data"]]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # Asynchronous embedding of multiple documents
        response = await aembedding(input=texts, **self.embedding_args)
        return [item["embedding"] for item in response["data"]]

    async def aembed_query(self, text: str) -> List[float]:
        # Asynchronous embedding of a single query
        response = await aembedding(input=[text], **self.embedding_args)
        return response["data"][0]["embedding"]


class LocalEmbeddings(HuggingFaceEmbeddings):
    """An embedding class for running HuggingFace embedding models locally"""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """An async version of the documents embedding"""
        return await asyncify(self.embed_documents)(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """An async version of the query embedding"""
        return await asyncify(self.embed_query)(text)


class CustomFAISS(FAISS):
    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """An async version of the delete method"""
        return await asyncify(self.delete)(ids, **kwargs)

    def merge_from(self, target: FAISS) -> None:
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Calculate the offset to apply to the target indices
        index_offset = max(self.index_to_docstore_id.keys(), default=-1) + 1

        # Update the index_to_docstore_id with the target's index_to_docstore_id, applying the offset
        updated_index_to_docstore_id = {
            index + index_offset: doc_id
            for index, doc_id in target.index_to_docstore_id.items()
        }

        # Add the target's documents to the docstore, skipping duplicates
        for doc_id in updated_index_to_docstore_id.values():
            if doc_id not in self.index_to_docstore_id.values():
                doc = target.docstore.search(doc_id)
                if not isinstance(doc, Document):
                    raise ValueError("Document should be returned")
                self.docstore.add({doc_id: doc})

        # Merge the updated mapping with the original mapping
        self.index_to_docstore_id.update(updated_index_to_docstore_id)


class BgeRerank(BaseDocumentCompressor):
    """Document compressor that uses the BAAI BGE reranking models."""

    name: str = "BAAI/bge-reranker-large"
    """Model name to use for reranking."""
    top_n: int = 10
    """Number of documents to return."""
    cache_dir: str = None
    revision: str | None = None
    model_args: dict = {"cache_dir": cache_dir}
    model: Optional[CrossEncoder] = None
    """CrossEncoder instance to use for reranking."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "forbid"
        arbitrary_types_allowed = True

    def _rerank(self, query: str, documents: Sequence[Document]) -> Sequence[Document]:
        """Rerank the documents"""
        _inputs = [[query, doc.page_content] for doc in documents]
        if self.model is None:
            self.model = CrossEncoder(
                self.name, revision=self.revision, cache_dir=self.cache_dir
            )
        _scores = self.model.predict(_inputs)
        results: List[Tuple[int, float]] = sorted(
            enumerate(_scores), key=lambda x: x[1], reverse=True
        )[: self.top_n]

        final_results = []
        for r in results:
            doc_index, relevance_score = r
            doc = documents[doc_index]
            doc.metadata["relevance_score"] = relevance_score
            final_results.append(doc)

        return final_results

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        return self._rerank(query=query, documents=documents)


class ReRanker(BaseDocumentCompressor):
    """Document compressor using litellm Rerank"""

    model: str = "cohere/rerank-english-v3.0"
    """Model name to use for reranking."""
    top_n: int = 10
    """Number of documents to return."""
    cache_dir: str = None
    revision: Optional[str] = None
    provider: str = "litellm"
    _hugging_reranker: BgeRerank = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_provider(self) -> "ReRanker":
        """Initialize huggingface reranker if applicable"""
        if any(x in self.model for x in ("huggingface", "BAAI")):
            self.provider = "huggingface"
            self._hugging_reranker = BgeRerank(
                name=self.model,
                top_n=self.top_n,
                cache_dir=self.cache_dir,
                revision=self.revision,
            )
        return self

    @staticmethod
    def _parse_response(
        response: list[dict], documents: Sequence[Document]
    ) -> Sequence[Document]:
        """Parse rerank response and attach scores to documents"""
        final_results = []
        for r in response:
            doc = documents[r["index"]]
            doc.metadata["relevance_score"] = r["relevance_score"]
            final_results.append(doc)
        return final_results

    def _get_document_contents(self, documents: Sequence[Document]) -> list[str]:
        """Extract page contents from documents"""
        return [doc.page_content for doc in documents]

    def _rerank(self, query: str, documents: Sequence[Document]) -> Sequence[Document]:
        """Rerank the documents"""
        if not documents:
            return []

        if self._hugging_reranker:
            result = self._hugging_reranker.compress_documents(
                query=query, documents=documents
            )
            return result

        response = rerank(
            model=self.model,
            query=query,
            documents=self._get_document_contents(documents),
            top_n=self.top_n,
            return_documents=False,
        )
        return self._parse_response(response.results, documents)

    async def _arerank(
        self, query: str, documents: Sequence[Document]
    ) -> Sequence[Document]:
        """Rerank the documents"""
        if not documents:
            return []

        if self._hugging_reranker:
            result = self._hugging_reranker.compress_documents(
                query=query, documents=documents
            )
            return result

        response = await arerank(
            model=self.model,
            query=query,
            documents=self._get_document_contents(documents),
            top_n=self.top_n,
            return_documents=False,
        )
        return self._parse_response(response.results, documents)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        return self._rerank(query=query, documents=documents)

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        return await self._arerank(query=query, documents=documents)


def markdown_to_text_using_html2text(markdown_text: str) -> str:
    """Convert the markdown docs into plaintext using the html2text plugin

    Args:
        markdown_text (str): Markdown text

    Returns:
        str: Plain text
    """
    html = markdown(markdown_text)
    return text_maker.handle(html).replace("\\", "")


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


async def gather_with_concurrency(n, *tasks):
    """
    Run multiple asyncio tasks concurrently, with a limit on the number of tasks that can run simultaneously.

    This function uses an asyncio.Semaphore to enforce a concurrency limit. It ensures that no more than `n` tasks
    are running at the same time. It is useful for controlling resource usage during the execution of a large number
    of concurrent asynchronous tasks.

    Args:
        n (int): The maximum number of concurrent tasks that are allowed to run at the same time.
        *tasks (coroutine): A variable number of asyncio coroutine objects to be run concurrently.

    Returns:
        list: A list of results from the completed asyncio tasks, in the order they were originally passed.

    Example:
        async def fetch(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()

        urls = ['http://example.com', 'http://example.org', 'http://example.net']
        tasks = [fetch(url) for url in urls]
        results = await gather_with_concurrency(2, *tasks)
        # This will run at most 2 fetch tasks at a time until all URLs are fetched.

    Note:
        All tasks must be coroutine objects, typically created by calling an async function.
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


async def aexecute_concurrently(
    func: Callable[[any], Coroutine],
    items: List,
    result_type: str = "append",
    max_workers: int = 10,
    input_type: str = "single",
) -> List:
    """
    Executes an async function concurrently on a list of items.

    Args:
        func (Callable[[any], Coroutine]): The async function to execute. This function should accept a single argument.
        items (List): The list of items to execute the function on.
        result_type (str): The type of result to return. Can be "append" or "extend". Defaults to "append".
        max_workers (int, optional): The maximum number of workers to use. Defaults to 10.
        input_type (str, optional): The expected type of inputs to the function

    Returns:
        List: A list of the results of the function execution.
    """

    results = []
    if input_type == "single":
        tasks = [func(item) for item in items]
    elif input_type == "dict":
        tasks = [func(**item) for item in items]
    else:
        raise Exception(f"{input_type} is not supported")

    executed_tasks = await gather_with_concurrency(max_workers, *tasks)

    if result_type == "append":
        results.extend(executed_tasks)
    else:
        for result in executed_tasks:
            results.extend(result)

    return results


def load_language_model(logger, model_name: str) -> ChatLiteLLM:
    """
    Helper method for loading language model
    """
    try:
        llm_model = ChatLiteLLM(model_name=model_name, streaming=True, max_retries=0)
        # Attempts to use the provider to capture any potential missing configuration error
        llm_model.invoke("Hi")
    except Exception as e:
        logger.error(
            f"Error initializing the language model '{model_name}'. Please check all required variables are set. "
            "Provider docs here - https://litellm.vercel.app/docs/providers \n"
        )
        raise e
    else:
        logger.info(f"Loaded the language model {model_name}")
    return llm_model


def load_embedding_model(logger, config: Config):
    """Embedding model loading and dimension calculation logic"""
    if config.embedding.type == "huggingface":
        embedding_model = LocalEmbeddings(
            cache_folder=str(config.core.data_dir) + "/models",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            model_name=config.embedding.model,
        )
    elif config.embedding.type == "litellm":
        embedding_model = LiteLLMEmbeddings(
            model=config.embedding.model, dimensions=config.embedding.dimension
        )
    else:
        raise ValueError(f"Unsupported embedding type: {config.embedding.type}")

    embed_dimension = config.embedding.dimension
    if embed_dimension is None:
        embed_dimension = len(embedding_model.embed_query("dummy"))

    logger.info(f"Loaded the embedding model {config.embedding.model}")

    return embedding_model, embed_dimension


def singleton(cls):
    """
    Thread-safe Singleton decorator to ensure that only one instance of a class is created.

    This decorator can be applied to any class to enforce the singleton pattern. It maintains a
    dictionary of instances and uses a threading lock to ensure that only one instance of the class
    is created, even in a multi-threaded environment.

    Args:
        cls (type): The class to be decorated as a singleton.

    Returns:
        type: The singleton class instance.

    Example:
        @singleton
        class MyClass:
            def __init__(self):
                self.value = 42

        obj1 = MyClass()
        obj2 = MyClass()
        assert obj1 is obj2  # Both are the same instance.
    """
    instances = {}
    lock = threading.Lock()

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
