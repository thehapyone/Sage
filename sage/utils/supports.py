from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
from typing import Union

from transformers import AutoModel
from langchain.schema.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, AIMessagePromptTemplate
from langchain.tools import Tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.callbacks.manager import Callbacks

from sentence_transformers import CrossEncoder
from markdown import markdown
from html2text import HTML2Text
from unstructured.partition.auto import partition_md

text_maker = HTML2Text()
text_maker.ignore_links = False
text_maker.ignore_images = True
text_maker.ignore_emphasis = True


class JinaAIEmebeddings(Embeddings):
    """Am embedding class powered by hugging face jinaAI"""

    def __init__(
        self,
        cache_dir: str,
        revision: str = "7302ac470bed880590f9344bfeee32ff8722d0e5",
        jina_model: str = "jinaai/jina-embeddings-v2-base-en",
    ):
        """Initialize the Jina Embeddings"""
        Path(cache_dir).mkdir(exist_ok=True)
        self.name = jina_model
        self.revision = revision
        self.cache_dir = cache_dir
        self.model = None

    def embed_query(self, text: str) -> List[float]:
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.name,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                resume_download=True,
                revision=self.revision,
            )
        return self.model.encode(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.name,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                resume_download=True,
                revision=self.revision,
            )
        return self.model.encode(texts)


class BgeRerank(BaseDocumentCompressor):
    """Document compressor that uses the BAAI BGE reranking models."""

    name: str = "BAAI/bge-reranker-large"
    """Model name to use for reranking."""
    top_n: int = 10
    """Number of documents to return."""
    cache_dir: str = None
    revision: str = None
    model_args: dict = {
        "cache_dir": cache_dir,
        "resume_download": True,
        "revision": revision,
    }
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
                self.name,
                tokenizer_args=self.model_args,
                automodel_args=self.model_args,
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


def markdown_to_text_using_html2text(markdown_text: str) -> str:
    """Convert the markdown docs into plaintext using the html2text plugin

    Args:
        markdown_text (str): Markdown text

    Returns:
        str: Plain text
    """
    html = markdown(markdown_text)
    return text_maker.handle(html).replace("\\", "")


def convert_intermediate_steps(intermediate_steps: dict):
    """
    Convert intermediate steps from agents into string outputs
    """
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool>"
            f"<tool_input>{action.tool_input}</tool_input>"
            f"<observation>{observation}</observation>"
        )
    return log


def agent_prompt(instructions: str) -> ChatPromptTemplate:
    """Generate a prompt template for XML agents"""
    return ChatPromptTemplate.from_template(
        instructions
    ) + AIMessagePromptTemplate.from_template("{intermediate_steps}")


def convert_tools(tools: List[Tool]):
    """Logic for converting tools to string to for usage in prompt"""
    result = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    return result


class CustomXMLAgentOutputParser(XMLAgentOutputParser):
    """Parses tool invocations and final answers in XML format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    <tool>search</tool>
    <tool_input>what is 2 + 2</tool_input>
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    <final_answer>Foo</final_answer>
    ```
    """

    def parse(self, text: str) -> AgentAction | AgentFinish:
        if "</tool>" in text:
            tool, tool_input = text.split("</tool>")
            _tool = tool.split("<tool>")[1]
            _tool_input = tool_input.split("<tool_input>")[1]
            if "</tool_input>" in _tool_input:
                _tool_input = _tool_input.split("</tool_input>")[0]
            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
        elif "<final_answer>" in text:
            _, answer = text.split("<final_answer>")
            if "</final_answer>" in answer:
                answer = answer.split("</final_answer>")[0]
            return AgentFinish(return_values={"output": answer}, log=text)
        else:
            return AgentFinish(return_values={"output": text}, log=text)


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
