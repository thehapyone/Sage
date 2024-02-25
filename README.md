# CodeSage: AI Agent for Your Workplace

## What is CodeSage?

CodeSage aims to be a fully-fledged Artificial General Intelligence (AGI) capable of autonomously working in a software development capacity. While the ultimate goal is ambitious, the current objectives of CodeSage are to:

- Analyze, generate, and debug code, perform static and dynamic analysis, detect faults, generate documentation, evaluate CI/CD, and more.
- Minimize the need for human code reviewers.
- Act as an action agent for issue management systems like Jira, effectively resolving tickets.

## Behavior

CodeSage is designed to:

- Provide professional and courteous interactions, offering insightful and constructive feedback to help users improve their coding skills.
- Identify potential bugs, errors, performance issues, and other code quality concerns.
- Be highly customizable and adaptable to user needs.

## Modes of Operation

CodeSage currently operates in three primary modes:

- **Sage Chat:** Engage in natural language conversations with users about code, data sources, and more.
- **Sage Reviewer:** Serve as an expert code reviewer, providing recommendations based on code changes and configurations.
- **Sage Jira:** Act as an autonomous Jira agent, managing and resolving tickets.

### Sage Reviewer

In the default deployment mode, Sage Reviewer, CodeSage functions as an expert code guru, analyzing code changes and suggesting improvements.

### Sage Chat

Sage Chat enables CodeSage to have natural conversations with users. The core technology behind Sage Chat is Retrieval Augmented Generation (RAG).

Learn more about Sage Chat [here](chainlit.md).

#### What is RAG?

RAG is a technique that enhances Large Language Models (LLMs) with additional data, enriching their responses.

#### RAG Architecture

A typical RAG setup includes:

- **Indexing:** A pipeline for ingesting and indexing data from various sources, usually performed offline.
- **Retrieval and Generation:** The real-time RAG process retrieves relevant data based on user queries and feeds it to the model for response generation.

## Sage Agent

In Agent mode, CodeSage performs specific actions autonomously, such as:

- Creating Merge Requests (MRs) to fix bugs.
- Managing Jira tickets by analyzing content and determining appropriate actions.

## Architecture

CodeSage is built on various components that power its functionality. Some components are mode-specific, while others are utilized across different modes:

- Config file
- Sources
- Embeddings
- LLMs
- Tools

### Configuration File

At the core of CodeSage is a TOML configuration file, which is used to configure various parts of the system. A Pydantic-based validator ensures the syntax and data values meet expected standards.

View a sample configuration [here](sage/config.toml).


### Embeddings

CodeSage utilizes various embedding engines to transform text inputs into vector representations for different functionalities. Both open-source and proprietary models are supported.

> **Note:** It is advisable to consistently use the same embedding model throughout the application runtime. Changing embeddings can lead to dimensionality mismatches and potentially cause errors. To switch embeddings, you may need to clear previously indexed data.



## Getting Started

To begin using CodeSage, you'll need to follow these setup steps:

1. **Installation:** Instructions for installing CodeSage and its dependencies.
2. **Configuration:** Guidelines for configuring CodeSage to suit your environment and use cases.
3. **Running CodeSage:** Step-by-step guide to starting CodeSage in one of its modes.

For detailed instructions, please refer to the [Installation Guide](installation.md), [Configuration Guide](configuration.md), and [Usage Guide](usage.md).

## Contributing

We welcome contributions from the community. If you're interested in contributing, please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

CodeSage is licensed under the ...

## Support

If you need assistance or want to report an issue, please file an issue on our [GitHub Issues](https://github.com/thehapyone/codesage/issues) page.

## Acknowledgments

We would like to thank all the contributors who have helped make CodeSage a reality, as well as the open-source projects that have provided inspiration and foundational technologies.

