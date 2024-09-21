# A series of enhance CrewAI Memory implementation
import asyncio
from typing import Any, Dict

from anyio import Path
from crewai.memory import LongTermMemory, ShortTermMemory
from crewai.memory.storage.interface import Storage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage


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
