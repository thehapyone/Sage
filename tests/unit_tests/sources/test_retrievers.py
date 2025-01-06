from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from sage.sources.retrievers import MultiSearchQueryRetriever, _unique_documents
from tests.unit_tests.extras import create_mock


@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=BaseRetriever)
    retriever.ainvoke = AsyncMock(spec=BaseRetriever)
    return retriever


@pytest.fixture
def multi_search_query_retriever(mock_retriever):
    return MultiSearchQueryRetriever(retriever=mock_retriever)


@pytest.mark.anyio
async def test_aretrieve_documents(multi_search_query_retriever, mock_retriever):
    queries = ["query1", "query2"]
    run_manager = MagicMock()
    mock_retriever.ainvoke.side_effect = [
        [Document(page_content="doc1")],
        [Document(page_content="doc2"), Document(page_content="doc3")],
    ]

    documents = await multi_search_query_retriever.aretrieve_documents(
        queries, run_manager
    )

    assert len(documents) == 3
    assert documents[0].page_content == "doc1"
    assert documents[1].page_content == "doc2"
    assert documents[2].page_content == "doc3"


def test_retrieve_documents(multi_search_query_retriever, mock_retriever):
    queries = ["query1", "query2"]
    run_manager = MagicMock()
    mock_retriever.invoke.side_effect = [
        [Document(page_content="doc1")],
        [Document(page_content="doc2"), Document(page_content="doc3")],
    ]

    documents = multi_search_query_retriever.retrieve_documents(queries, run_manager)

    assert len(documents) == 3
    assert documents[0].page_content == "doc1"
    assert documents[1].page_content == "doc2"
    assert documents[2].page_content == "doc3"


@pytest.mark.anyio
async def test_aget_relevant_documents(multi_search_query_retriever, mock_retriever):
    query = "query1"
    mock_retriever.ainvoke.return_value = [Document(page_content="doc1")]

    documents = await multi_search_query_retriever._aget_relevant_documents(
        query, run_manager=MagicMock()
    )

    assert len(documents) == 1
    assert documents[0].page_content == "doc1"


def test_get_relevant_documents(multi_search_query_retriever, mock_retriever):
    query = "query1"
    mock_retriever.invoke.return_value = [Document(page_content="doc1")]

    documents = multi_search_query_retriever._get_relevant_documents(
        query, run_manager=MagicMock()
    )

    assert len(documents) == 1
    assert documents[0].page_content == "doc1"


def test_unique_documents():
    documents = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc1"),
    ]
    unique_docs = _unique_documents(documents)

    assert len(unique_docs) == 2
    assert unique_docs[0].page_content == "doc1"
    assert unique_docs[1].page_content == "doc2"
