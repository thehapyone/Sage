# Welcome to Sage Chat! 🚀🤖

Sage Chat is an AI Assistant designed to interact with various data sources and provide answers to questions.

*The Chat UI interface is powered by Chainlit. Learn more in the [Chainlit Documentation](https://docs.chainlit.io).* 📚

## Chat Architecture

Sage Chat leverages the [RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) system to provide accurate and reliable answers to user queries.

> Retrieval-augmented generation (RAG) is a technique that enhances the capabilities of generative AI models by incorporating facts retrieved from external sources.

Sage is built on various components that power its functionality. Some components are mode-specific, while others are utilized across different modes:

- Config file
- Sources
- Embeddings
- LLMs
- Tools

### Indexing Process

The sequence from raw data to answer involves the following steps:

- **Load**: Data is loaded using loaders in the Sources Module.
- **Split**: Text splitters break large documents into smaller chunks, facilitating indexing and model input.
- **Store**: Chunks are stored and indexed using a VectorStore and Embeddings model, enabled by the FAISS Vector Database and a combination of Jina/OpenAI Embeddings.

### Retrieval and Generation Process

- **Retrieve**: Relevant document chunks are retrieved from storage in response to user input using a Retriever.
- **Generate**: A ChatModel / LLM formulates an answer using a prompt that incorporates both the question and the retrieved data.

## Chain Overview

The architecture of Sage Chat is depicted below:

      +-------------------+
      |   User Question   |
      +---------+---------+
                |
                v
      +---------+---------+
      | Query Generator   |
      |       Chain       |
      +---------+---------+
                |
                v
      +---------+-----------+
      | Generates 3 Queries |
      +---------+-----------+
                |
                v
      +-----------+-----------+-----------+
      |  Query 1  |  Query 2  |  Query 3  |
      +-----------+----+------+-----------+
           |           |           |
           v           v           v
      +----+--------+-------------+-------------+
      | Retriever 1 | Retriever 2 | Retriever 3 |
      +----+--------+-------------+-------------+
           |           |           |
           v           v           v
      +---------------------------------+
      |      Combine Docs/Context       |
      +---------------------------------+
                      |
                      v
       +-------------------------+
       |  Question & Sources     |
       |        Chain            |
       +-------------------------+
                |
                v
       +---------+---------+
       |   Final Answer    |
       +-------------------+


Key components include:
- Query Generator Chain
- Retriever and Reranking Chain
- Question & Sources Chain

### Query Generator Chain

This stage processes the user input question along with chat history to generate multiple relevant search queries:

```yaml
HUMAN:
question: "Tell me about the Eiffel Tower."
history: []

AI:
["Eiffel Tower history", "Eiffel Tower facts", "Eiffel Tower significance"]
```

### Retriever and Reranking Chain
 
This stage involves converting the generated queries to vector representations and performing similarity searches to retrieve related documents. If reranking is enabled, it further refines the search results to improve relevance.

Currently supported rerankers include:

 - Cohere Reranking (Proprietary)
 - BGE Reranker (Open-source)

### Question & Sources Chain
 
This final stage varies based on the user-selected mode in the chat interface:

 - Chat-only mode (default): The results from reranking/retrieval are passed to an LLM chain to answer the question using the retrieved documents as a knowledge source.
 - Agent mode: AI Agenets configured work on performing preselected task.
 - File mode: In this mode, Sage is not configured with any retriever and it instead creates a real-time retriever for user's uploaded documents and the user can then ask questions about the documents provided.
