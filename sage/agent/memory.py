# A series of enhance CrewAI Memory implementation
import hashlib
from pathlib import Path
from typing import Any, Dict

from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.interface import Storage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage

from sage.sources.source_manager import SourceManager
from sage.utils.supports import CustomFAISS as FAISS


class EnhanceLongTermMemory(LongTermMemory):
    """A longtermMemory instance support custom storage class"""

    def __init__(self, storage: Storage = LTMSQLiteStorage()):
        self.storage = storage


class EnhanceShortTermMemory(ShortTermMemory):
    """A shortTermmemory instance support custom storage class"""

    def __init__(self, crew=None, embedder_config=None, storage=None):
        self.storage = (
            storage
            if storage
            else RAGStorage(
                type="short_term", embedder_config=embedder_config, crew=crew
            )
        )


class EnhanceEntityMemory(EntityMemory):
    """A EntityMemory instance support custom storage class"""

    def __init__(self, crew=None, embedder_config=None, storage=None):
        self.storage = (
            storage
            if storage
            else RAGStorage(type="entities", embedder_config=embedder_config, crew=crew)
        )


class CustomRAGStorage(Storage):
    """
    A RAG storage class for sage-based memory management.
    """

    def __init__(
        self,
        crew_name: str,
        storage_type: str,
        data_dir: Path,
        model: Any,
        dimension: int,
    ) -> None:
        self.hash = self._generate_hash(crew_name, storage_type)
        self.data_dir = data_dir
        self.refresh_needed = False
        self._faiss_db: None | FAISS = None

        # Ensure the data directory exists
        self._create_data_dir()

        # Initialize the source manager
        self.manager = SourceManager(
            embedding_model=model,
            model_dimension=dimension,
            source_dir=data_dir,
            record_manager_dir=data_dir,
        )

    def _normalize(self, input_str: str) -> str:
        return input_str.strip().lower()

    def _generate_hash(self, crew_name: str, storage_type: str) -> str:
        md5_hash = hashlib.md5(self._normalize(crew_name).encode("utf-8")).hexdigest()
        return f"{self._normalize(storage_type)}-{md5_hash}"

    def _create_data_dir(self):
        Path(self.data_dir).mkdir(exist_ok=True)

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        self.manager.add_sync(
            hash=self.hash,
            source_type="text",
            identifier=metadata,
            data=value,
            cleanup=None,
        )
        self.refresh_needed = True

    def _load_faiss_database(self) -> FAISS:
        return self.manager._get_or_create_faiss_db_sync(self.hash)

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: dict | None = None,
        score_threshold: float = 0.35,
    ) -> list[Dict[str, Any]]:
        if self.refresh_needed or self._faiss_db is None:
            self._faiss_db = self._load_faiss_database()
            self.refresh_needed = False

        docs_and_scores = self._faiss_db.similarity_search_with_relevance_scores(
            query, k=limit, score_threshold=score_threshold
        )
        results = [
            {"context": doc.page_content, "metadata": doc.metadata}
            for doc, _ in docs_and_scores
        ]
        return results

    def reset(self) -> None:
        try:
            self.manager.add_sync(
                hash=self.hash,
                source_type="text",
                identifier={},
                data="",
                cleanup="full",
            )
            self.refresh_needed = True
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the {self.hash} memory: {e}"
            )
