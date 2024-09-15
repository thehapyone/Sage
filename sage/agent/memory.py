# A series of enhance CrewAI Memory implementation
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.storage.interface import Storage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


class EnhanceLongTermMemory(LongTermMemory):
    """A longtermMemory instance support custom storage class"""

    def __init__(self, storage: Storage = LTMSQLiteStorage()):
        self.storage = storage
