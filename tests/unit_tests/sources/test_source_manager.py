import asyncio
from pathlib import Path as SyncPath
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anyio import Path as AsyncPath

from sage.sources.source_manager import (
    AsyncRunner,
    Document,
    SourceManager,
    convert_sources_to_string,
    get_faiss_indexes,
)


@pytest.fixture
def source_manager():
    return SourceManager(
        embedding_model=MagicMock(),
        model_dimension=768,
        source_dir=AsyncPath("/mock/source/dir"),
        record_manager_dir=AsyncPath("/mock/source/dir"),
    )


def test_convert_sources_to_string():
    confluence = MagicMock(spaces=["space1", "space2"])
    gitlab = MagicMock(groups=["group1"], projects=["project1"])
    web = MagicMock(links=["http://example.com"])
    sources_config = MagicMock()
    sources_config.confluence = confluence
    sources_config.gitlab = gitlab
    sources_config.web = web

    result = convert_sources_to_string(sources_config)
    expected = (
        "- Confluence spaces: space1, space2\n"
        "  - GitLab Groups: group1\n"
        "  - GitLab repositories: project1\n"
        "  - Relevant documentation and resources available at: http://example.com"
    )

    assert result == expected


def test_get_faiss_indexes_sync():
    sync_dir_mock = MagicMock()
    sync_dir_mock.glob.return_value = [
        SyncPath("index1.faiss"),
        SyncPath("index2.faiss"),
    ]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    coro = get_faiss_indexes(sync_dir_mock)
    result = loop.run_until_complete(coro)
    assert result == ["index1", "index2"]


@pytest.mark.anyio
async def test_faiss_db_integration(source_manager):
    source_hash = "mock_source_hash"

    with patch(
        "sage.sources.source_manager.get_faiss_indexes", return_value=[source_hash]
    ):
        # mock_get_faiss_indexes.return_value.set_result([source_hash])

        with patch("sage.sources.source_manager.FAISS") as mock_faiss_constructor:
            mock_faiss = MagicMock(name="FAISS Instance")
            mock_faiss_constructor.return_value = mock_faiss

            db = await source_manager._get_or_create_faiss_db("new_source_hash")
            assert db == mock_faiss

        with patch(
            "sage.sources.source_manager.FAISS.load_local", return_value=MagicMock()
        ) as mock_load_local:
            mock_faiss = MagicMock(name="FAISS Instance")
            mock_load_local.return_value = mock_faiss

            db = await source_manager._get_or_create_faiss_db(source_hash)
            mock_load_local.assert_called_once()
            assert db == mock_faiss


@pytest.mark.anyio
async def test_record_manager_integration(source_manager):
    source_hash = "mock_source_hash"

    with patch("sage.sources.source_manager.SQLRecordManager") as mock_SQLRecordManager:
        mock_record_manager = MagicMock()
        mock_SQLRecordManager.return_value = mock_record_manager
        mock_record_manager.acreate_schema.side_effect = Exception(
            "table upsertion_record already exists"
        )

        result = await source_manager._get_record_manager(source_hash)
        assert result == mock_record_manager


@pytest.mark.anyio
async def test_add_source_integration(source_manager):
    hash = "mock_hash"
    documents = [Document(page_content="content", metadata={"key": "value"})]

    mock_db = MagicMock(name="FAISS Instance")

    with patch(
        "sage.sources.source_manager.SourceManager._create_and_save_db",
        return_value=mock_db,
    ) as mock_create_and_save_db:
        db = await source_manager._add_source(hash, documents)
        assert db is mock_db
        mock_create_and_save_db.assert_awaited_once()


@pytest.fixture
def mock_add_source(monkeypatch):
    mock_add_source = AsyncMock(name="add_source")
    monkeypatch.setattr(
        "sage.sources.source_manager.SourceManager._add_source", mock_add_source
    )
    mock_add_source.return_value = MagicMock(name="FAISS Instance")
    return mock_add_source


@pytest.fixture
def mock_documents():
    return [Document(page_content="content", metadata={"key": "value"})]


@pytest.mark.anyio
async def test_add_text_integration(source_manager, mock_documents, mock_add_source):
    hash = "mock_hash"
    data = "mock_data"
    metadata = {"key": "value"}

    expected_doc = [Document(page_content=data, metadata={**metadata, "source": hash})]

    await source_manager._add_text(hash, data, metadata, None)

    mock_add_source.assert_awaited_once_with(hash, expected_doc, cleanup=None)


@pytest.mark.anyio
async def test_add_confluence_integration(
    source_manager, mock_add_source, mock_documents
):
    hash = "mock_hash"
    data = MagicMock()
    space = "mock_space"

    with patch(
        "sage.sources.source_manager.CustomConfluenceLoader.load",
        return_value=mock_documents,
    ) as mock_loader_load:
        await source_manager._add_confluence(hash, data, space)

        mock_loader_load.assert_called_once
        mock_add_source.assert_awaited_once


@pytest.mark.anyio
async def test_add_gitlab_source_integration(
    source_manager, mock_add_source, mock_documents
):
    hash = "mock_hash"
    data = MagicMock()
    groups = ["mock_group"]
    projects = ["mock_project"]

    with patch(
        "sage.sources.source_manager.GitlabLoader", autospec=True
    ) as mock_gitlab_loader:
        mock_gitlab_loader_instance = mock_gitlab_loader.return_value
        mock_gitlab_loader_instance.load = AsyncMock(return_value=mock_documents)

        await source_manager._add_gitlab_source(
            hash, data, groups=groups, projects=projects, cleanup="full"
        )

        mock_gitlab_loader_instance.load.assert_any_await
        mock_add_source.assert_awaited_once


@pytest.mark.anyio
async def test_add_web_source_integration(
    source_manager, mock_add_source, mock_documents
):
    hash = "mock_hash"
    data = MagicMock()
    link = "mock_link"

    with patch(
        "sage.sources.source_manager.WebLoader.load", return_value=mock_documents
    ) as mock_loader_load:
        await source_manager._add_web_source(hash, data, link)
        mock_loader_load.assert_called_once
        mock_add_source.assert_awaited_once


def test_async_runner():
    async def sample_coroutine():
        return "result"

    runner = AsyncRunner()
    result = runner.run(sample_coroutine())
    assert result == "result"
    runner.shutdown()


@pytest.mark.anyio
async def test_add_files_source(source_manager, mock_documents):
    hash_ = "file_hash"
    path = "/mock/path/to/file"

    with patch(
        "sage.sources.source_manager.UnstructuredFileLoader", autospec=True
    ) as mock_loader:
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.load.return_value = mock_documents

        with patch.object(
            source_manager, "_add_source", new_callable=AsyncMock
        ) as mock_add_source:
            await source_manager._add_files_source(hash_, path)

    mock_loader.assert_called_once_with(file_path=path, mode="single")
    mock_add_source.assert_called_once_with(
        hash_, mock_documents, save_db=True, cleanup="full"
    )


@pytest.mark.anyio
async def test_add(source_manager):
    hash = "generic_hash"
    source_type = "text"
    identifier = {"name": "demo"}
    data = "some_text_data"

    with patch.object(
        source_manager, "_add_text", new_callable=AsyncMock
    ) as mock_add_text:
        await source_manager.add(
            hash=hash,
            source_type=source_type,
            identifier=identifier,
            data=data,
            cleanup=None,
        )

        mock_add_text.assert_called_once_with(
            hash=hash, metadata=identifier, data=data, cleanup=None
        )


def test_add_sync(source_manager):
    hash_ = "sync_hash"
    source_type = "text"
    identifier = {"name": "demo"}
    data = "sync_data"
    identifier_type = None

    with patch.object(source_manager, "add", new_callable=AsyncMock) as mock_add:
        source_manager.add_sync(
            hash_, source_type, identifier, identifier_type=identifier_type, data=data
        )

        mock_add.assert_called_once_with(
            hash=hash_,
            source_type=source_type,
            identifier=identifier,
            identifier_type=identifier_type,
            data=data,
            cleanup="full",
        )
