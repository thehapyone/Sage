# Large Language Models (LLMs) Supported by Sage

Sage leverages Large Language Models (LLMs) to power its core functionalities. Through integration with the LiteLLM framework, Sage offers the flexibility to utilize a variety of LLMs to suit different requirements.

Sage is designed to be compatible with a wide array of LLMs, ensuring seamless configuration and integration.

## Supported LLMs

Sage supports several LLMs, including but not limited to:

- **OpenAI's GPT Models**: Harness the power of models like GPT-3 to generate human-like text.
- **Azure's OpenAI Models**: Leverage Azure's implementation of models like GPT-3 for text generation.
- **Hugging Face Models**: Access a diverse collection of models from the Hugging Face community.
- ...and many others.

For a comprehensive list of supported LLMs, visit the LiteLLM documentation: [Supported LLM Providers](https://litellm.vercel.app/docs/providers)

## Interacting with LLMs

To use an LLM with Sage:

1. Submit a query that the configured LLM can process.
2. Sage will utilize the selected LLM to generate and return a response.

## Configuring LLMs

Selecting an LLM for Sage is as simple as specifying the model name in the `config.toml` file. API keys and other credentials are securely managed via environment variables set in your `.env` file.

Example configuration in `config.toml`:

```toml
[llm]
model = "groq/llama3-70b-8192"
```

For model names and required environment variables, refer to the LiteLLM documentation: [LLM Configuration](https://litellm.vercel.app/docs/providers)

## Embeddings

Sage employs various embedding engines to convert text inputs into vector representations, supporting both open-source and proprietary models.

> **Note:** Consistency in embedding models is crucial during application runtime to avoid dimensionality mismatches and errors. If you need to change embeddings, ensure previously indexed data is cleared.

Sage's embedding capabilities are enhanced by the LiteLLM embeddings API and the HuggingFace Hub, enabling support for numerous embeddings both online and locally.

### Example - Using Hugging Face Models Locally

```toml
[embedding]
type = "huggingface"
model = "nvidia/NV-Embed-v2"
#model = "jinaai/jina-embeddings-v2-base-en"
#dimension = 768
```

Explore all compatible Hugging Face embedding models: [Hugging Face Models](https://huggingface.co/models?pipeline_tag=feature-extraction)

### Example - Using Other Embeddings like OpenAI, Cohere, etc.

```toml
[embedding]
type = "litellm"
model = "text-embedding-ada-002"
```

Discover all supported embedding models: [Supported Embeddings](https://litellm.vercel.app/docs/embedding/supported_embedding)

## Best Practices

- Ensure your queries contain sufficient context for the most accurate responses.
- Be mindful of the limitations and potential costs associated with each LLM.
