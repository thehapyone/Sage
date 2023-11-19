from pathlib import Path
from typing import List
from transformers import AutoModel
from langchain.schema.embeddings import Embeddings


class JinaAIEmebeddings(Embeddings):
    """Am embedding class powered by hugging face jinaAI"""

    def __init__(self,
                 cache_dir: str,
                 revision: str = "7302ac470bed880590f9344bfeee32ff8722d0e5",
                 jina_model: str = "jinaai/jina-embeddings-v2-base-en"):
        """Initialize the Jina Embeddings"""
        Path(cache_dir).mkdir(exist_ok=True)

        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=jina_model,
            trust_remote_code=True,
            cache_dir=cache_dir,
            resume_download=True,
            revision=revision
        )

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)
