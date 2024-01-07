# Welcome to Sage Chat! 🚀🤖

Sage Chat is an AI Assistant designed to interact with various data sources and provide answers to questions.

*The Chat UI interface is powered by Chainlit. Learn more in the [Chainlit Documentation](https://docs.chainlit.io).* 📚

## Chat Architecture

Sage Chat leverages the [RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) system to provide accurate and reliable answers to user queries.

> Retrieval-augmented generation (RAG) is a technique that enhances the capabilities of generative AI models by incorporating facts retrieved from external sources.

The core components of Sage Chat include:

- LLMs (Large Language Models)
- Embeddings
- Sources
- Prompts

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

![Alt text](sage/assets/sage_chain.png?raw=true "Sage Chain Overview")

Key components include:
- Condense Question LLM Chain
- Retriever and Reranking Chain
- Question & Sources plus Agent Executor Chain

### Condense Question LLM Chain

This stage processes the user input question along with chat history to generate a new standalone question. The output is a JSON object as shown below:

```yaml
HUMAN:
question: "Team members in the Xerex team?"
history: []

AI:
{
    "retriever": "YES",
    "standalone_question": "How many team members are in the Xerex team?"
}
```

The chain also outputs a "YES" or "NO" for the retriever field, indicating whether the question requires support from external sources. For instance:

```yaml
HUMAN:
question: "Nigeria's independence?"
history: []

AI:
{
    "retriever": "NO",
    "standalone_question": "When did Nigeria gain its independence?"
}
```

### Retriever and Reranking Chain
 
This stage involves converting the standalone question to a vector representation and performing a similarity search to retrieve related documents. If reranking is enabled, it further refines the search results to improve relevance.

Currently supported rerankers include:

 - Cohere Reranking (Proprietary)
 - BGE Reranker (Open-source)

### Question & Sources plus Agent Executor Chain
 
This final stage varies based on the user-selected mode in the chat interface:

 - Chat-only mode (default): The results from reranking/retrieval are passed to an LLM chain to answer the question using the retrieved documents as a knowledge source.
 - Agent mode: The agent executor, equipped with tools, actively combines data from retrieval/reranking and any available tools to answer the question, possibly involving multiple steps.

## Tips

If Sage is unable to answer a question, even with the correct information in the sources, users can improve the response by providing more detailed questions or guidance, which can significantly enhance the results.

For instance - To force the condense question chain from not rephraseing the orginial question you can add some quidance
```yaml
# -------Before--------------
HUMAN:
Question: "In your own words or possibility, how many years do humanaility have left to able to achieve AGI? I understand you don't know but I insist to give some estimates here - take a look at historical human advanment in the last few decades."
AI: "Based on the progress in AI over the past decades, what is an estimated timeline for the achievement of Artificial General Intelligence (AGI)?"
# -------After--------------
HUMAN:
Question: "In your own words or possibility, how many years do humanity have left to able to achieve AGI? I understand you don't know but I insist to give some estimates here - take a look at historical human advancement in the last few decades."

Return question as it is - DO NOT MODIFY PLEASE"
AI: "In your own words or possibility, how many years do humanity have left to able to achieve AGI? I understand you don't know but I insist to give some estimates here - take a look at historical human advancement in the last few decades."
```