from pathlib import Path
from typing import List
from transformers import AutoModel
from langchain.schema.embeddings import Embeddings

import unmarkd
from markdown import markdown
from html2text import HTML2Text


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

        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=jina_model,
            trust_remote_code=True,
            cache_dir=cache_dir,
            resume_download=True,
            revision=revision,
        )

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)


def markdown_to_text_using_unmark(markdown_text: str) -> str:
    """Convert the markdown docs into plaintext

    Args:
        markdown_text (str): Markdown text

    Returns:
        str: Plain text
    """
    return unmarkd.unmark(markdown_text).replace("\\", "")


def markdown_to_text_using_html2text(markdown_text: str) -> str:
    """Convert the markdown docs into plaintext using the html2text plugin

    Args:
        markdown_text (str): Markdown text

    Returns:
        str: Plain text
    """
    text_maker = HTML2Text()
    text_maker.ignore_links = False
    text_maker.ignore_images = True
    text_maker.ignore_emphasis = False
    return text_maker.handle(markdown(markdown_text))
