import asyncio
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class MultiSearchQueryRetriever(BaseRetriever):
    """A retriever instance that allows running muiltple queries towards a retriever instance"""

    retriever: BaseRetriever

    async def _aget_relevant_documents(
        self,
        query: str | list[str],
        *,
        run_manager: callable,
    ) -> list[Document]:
        """Get relevant documents given a query or list of queries.

        Args:
            query: user query

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = query if isinstance(query, list) else [query]
        documents = await self.aretrieve_documents(queries, run_manager)
        return documents

    async def aretrieve_documents(
        self, queries: list[str], run_manager: callable
    ) -> list[Document]:
        """Retrieve docs for all queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
        document_lists = await asyncio.gather(
            *(
                self.retriever.ainvoke(
                    query, config={"callbacks": run_manager.get_child()}
                )
                for query in queries
            )
        )
        return list({doc for docs in document_lists for doc in docs})

    def _get_relevant_documents(
        self,
        query: str | list[str],
        *,
        run_manager: callable,
    ) -> list[Document]:
        """Get relevant documents given a query or list of queries.

        Args:
            query: user query

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = query if isinstance(query, list) else [query]
        documents = self.retrieve_documents(queries, run_manager)
        return documents

    def retrieve_documents(
        self, queries: list[str], run_manager: callable
    ) -> list[Document]:
        """Retrieve docs for all queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
        documents = set()
        for query in queries:
            docs = self.retriever.invoke(
                query, config={"callbacks": run_manager.get_child()}
            )
            documents.update(docs)

        return list(documents)