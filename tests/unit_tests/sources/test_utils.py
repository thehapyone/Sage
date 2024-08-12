from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from anyio import Path
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.document import Document

from sage.sources.utils import (
    check_for_data_updates,
    format_docs,
    format_sources,
    generate_ui_actions,
    get_git_source,
    get_memory,
    get_retriever,
    get_time_of_day_greeting,
    load_chat_history,
)
from tests.unit_tests.extras import MockMessage, MockSession


def test_format_docs():
    docs = [
        Document("Page content 1", metadata={}),
        Document("Page content 2", metadata={}),
    ]
    result = format_docs(docs)
    expected = "<doc id='0'>Page content 1</doc>\n<doc id='1'>Page content 2</doc>"
    assert result == expected


def test_get_git_source_exception_keyerror():
    metadata = {"branch": "main", "source": "file.py"}
    result = get_git_source(metadata)
    expected = "file.py"
    assert result == expected


def test_get_git_source():
    metadata = {"url": "https://example.git", "branch": "main", "source": "file.py"}
    result = get_git_source(metadata)
    expected = "https://example/-/blob/main/file.py"
    assert result == expected


def test_format_sources():
    docs = [
        Document(
            "Content 1",
            metadata={
                "url": "https://example.git",
                "branch": "main",
                "source": "file1.py",
            },
        ),
        Document("Content 2", metadata={"source": "source2.py"}),
    ]
    result = format_sources(docs)
    expected = [
        {
            "id": 0,
            "source": "https://example/-/blob/main/file1.py",
            "content": "Content 1 ...",
        },
        {"id": 1, "source": "source2.py", "content": "Content 2 ..."},
    ]
    assert result == expected


@pytest.mark.parametrize(
    "hour, expected_greeting",
    [
        (6, "Good morning"),
        (13, "Good afternoon"),
        (18, "Good evening"),
        (22, "Hello"),
    ],
)
def test_get_time_of_day_greeting(monkeypatch, hour, expected_greeting):
    class MockDateTime(datetime):
        @classmethod
        def now(cls):
            return cls(2023, 1, 1, hour, 0, 0)

    monkeypatch.setattr("sage.sources.utils.datetime", MockDateTime)
    assert get_time_of_day_greeting() == expected_greeting


@pytest.mark.anyio
async def test_check_for_data_updates():
    sentinel = AsyncMock(Path)
    logger = MagicMock()
    sentinel.exists.return_value = True
    sentinel.read_text.return_value = "updated"

    result = await check_for_data_updates(sentinel, logger)
    assert result is True
    sentinel.write_text.assert_called_once_with("")
    logger.info.assert_called_once_with(
        "Data update detected, reloading the retriever database"
    )


@pytest.mark.anyio
async def test_get_retriever_none_hash():
    source = AsyncMock()
    source.load.return_value = "retriever"
    retriever = await get_retriever(source, "none")
    assert retriever == "retriever"


@pytest.mark.anyio
async def test_get_retriever(monkeypatch):
    source = AsyncMock()
    source.load.return_value = "retriever"

    monkeypatch.setattr("sage.sources.utils.check_for_data_updates", AsyncMock())
    monkeypatch.setattr("sage.sources.utils.cl.Message", MockMessage)

    retriever = await get_retriever(source, "valid_hash")
    assert retriever == "retriever"
    source.load.assert_awaited_once_with("valid_hash")


def test_generate_ui_actions():
    metadata = {"source1": "Source 1", "source2": "Source 2"}
    actions = generate_ui_actions(metadata)
    assert len(actions) == 2


def test_get_memory_tool_mode():
    mode = "tool"
    user_session = MockSession()
    memory = get_memory(mode, user_session)
    assert isinstance(memory, ConversationBufferWindowMemory)
    assert user_session.user_session == {}


def test_get_memory_chat_mode_empty_memory():
    mode = "chat"
    user_session = MockSession()
    memory = get_memory(mode, user_session)
    saved_memory = user_session.get("memory")
    assert memory == saved_memory


def test_get_memory_chat_mode_saved_memory():
    mode = "chat"
    user_session = MockSession()
    saved_memory = ConversationBufferWindowMemory()
    user_session.set("memory", saved_memory)
    memory = get_memory(mode, user_session)
    assert memory == saved_memory


def test_load_chat_history():
    mode = "chat"
    user_session = MockSession()

    result = load_chat_history(mode, user_session)

    assert result.first.config == {"run_name": "ChatHistory"}
