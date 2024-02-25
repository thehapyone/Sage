# Large Language Models (LLMs) Supported by Sage

Large Language Models are fundamental to CodeSage's operation. The system can leverage various LLMs thanks to the integration with the LangChain framework, which provides the flexibility to use different models as required.

## Supported LLMs

- **OpenAI's GPT Models**: Utilize models like GPT-3 for generating human-like text.
- **Azure's OpenAI Models**: Utilize models like GPT-3 for generating human-like text.
- **Hugging Face Models**: Access a wide range of models from the Hugging Face community.

## Interacting with LLMs

To interact with an LLM through Sage:

1. Send a query that the LLM is configured to respond to.
2. Sage will process the query using the appropriate LLM and return a response.

## Configuring LLMs

Ensure you have the necessary API keys and tokens set up in your `.env` file and `config.toml`. For each LLM, specify the settings such as model type, API endpoint, and any other relevant parameters.


## Embeddings

CodeSage utilizes various embedding engines to transform text inputs into vector representations for different functionalities. Both open-source and proprietary models are supported.

> **Note:** It is advisable to consistently use the same embedding model throughout the application runtime. Changing embeddings can lead to dimensionality mismatches and potentially cause errors. To switch embeddings, you may need to clear previously indexed data.


## Best Practices

- Provide enough context in your queries to get the most accurate responses.
- Be aware of the limitations and usage costs associated with each LLM.

...

Stay informed on the latest updates and additions to the LLMs supported by Sage.
