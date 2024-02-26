# Welcome to Sage Chat! 🚀🤖

Sage Chat is an AI Assistant designed to interact with various data sources and provide answers to questions, leveraging advanced AI techniques to deliver reliable and contextually relevant information.

*The Chat UI interface is powered by Chainlit. For more details, refer to the [Chainlit Documentation](https://docs.chainlit.io).* 📚

## Architecture

Sage Chat employs the [Retrieval-Augmented Generation (RAG)](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) system, a technique that combines generative AI models with external data sources to enhance response accuracy.

CodeSage is built on various components that power its functionality. Some components are mode-specific, while others are utilized across different modes:

- Config file
- Sources
- Embeddings
- LLMs
- Tools

The core components of Sage Chat include:

- **Large Language Models (LLMs)**: Provide the foundation for generating human-like text.
- **Embeddings**: Represent text as vectors for efficient retrieval.
- **Sources**: Various data sources that Sage can interact with.
- **Prompts**: Structured inputs that guide the AI in generating responses.

### Indexing Process

The process from raw data to answer involves these steps:

1. **Load**: Data loaders in the Sources Module import data.
2. **Split**: Text splitters segment large documents into manageable chunks.
3. **Store**: Chunks are indexed using a VectorStore and Embeddings model, powered by FAISS and Jina/OpenAI Embeddings.

### Retrieval and Generation Process

1. **Retrieve**: The Retriever fetches relevant document chunks based on user input.
2. **Generate**: The ChatModel / LLM crafts an answer using a prompt that includes the question and retrieved data.

## Chain Overview

Sage Chat's architecture is visualized below:

![Sage Chain Overview](../sage/assets/sage_chain.png?raw=true "Sage Chain Overview")

Key components include:

- **Condense Question LLM Chain**
- **Retriever and Reranking Chain**
- **Question & Sources plus Agent Executor Chain**

### Condense Question LLM Chain

This component refines the user's input question and chat history to generate a concise, standalone question:

```yaml
# Example
HUMAN:
  question: "Who are the team members in the Xerex team?"
  history: []

AI:
  condensed_question: "List the team members in the Xerex team."

```

### Retriever and Reranking Chain

This component transforms the standalone question into a vector and retrieves related documents. Rerankers can further refine the results:

- **Cohere Reranking** (Proprietary)
- **BGE Reranker** (Open-source)

### Question & Sources plus Agent Executor Chain

The final stage depends on the user-selected mode:

- **Chat-only mode** (default): Uses retrieved documents to answer the question.
- **Agent mode**: The agent executor actively combines data and tools to respond.
- **File mode**: Creates a real-time retriever for user-uploaded documents for querying.

## Tips for Better Interactions

If Sage does not provide the expected answer, consider these tips:

- Provide detailed questions or guidance to improve accuracy.
- To prevent the condense question chain from rephrasing the original question, add specific instructions like "Return question as it is - DO NOT MODIFY PLEASE."

```yaml
# Example
HUMAN:
  Question: "Estimate the years left for humanity to achieve AGI, considering historical advancements."

AI:
  Response: "Estimating the timeline for AGI is speculative, but considering the pace of recent advancements, a rough estimate might be within the next few decades."
```

For instance - To force the condense question chain from not rephraseing the orginial question you can add some quidance
```yaml
# -------Before--------------
HUMAN:
    Question: "In your own words or possibility, how many years do humanaility have left to able to achieve AGI? I understand you don't know but I insist to give some estimates here - take a look at historical human advanment in the last few decades."

AI: 
    Question: "Based on the progress in AI over the past decades, what is an estimated timeline for the achievement of Artificial General Intelligence (AGI)?"

# -------After--------------
HUMAN:
    Question: "In your own words or possibility, how many years do humanity have left to able to achieve AGI? I understand you don't know but I insist to give some estimates here - take a look at historical human advancement in the last few decades."

    Return question as it is - DO NOT MODIFY PLEASE"
AI: 
    Question: "In your own words or possibility, how many years do humanity have left to able to achieve AGI? I understand you don't know but I insist to give some estimates here - take a look at historical human advancement in the last few decades."
```
