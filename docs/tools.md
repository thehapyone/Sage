# Tools Overview for Sage

CodeSage utilizes tools to perform actions beyond the capabilities of LLMs alone. These tools enhance CodeSage's functionality and help to address some of the limitations inherent in LLMs. Configuration files dictate which tools CodeSage can employ to accomplish specific tasks.

Integration with a variety of tools helps to enhance the data interaction experience. 

Some of the functionalities enabled by tools include:

- Retrieving real-time weather forecasts.
- Accessing the latest information available on the internet.
- Checking the status of tickets in issue management systems.
- Querying databases for specific data.
- Publishing content to Confluence.
- Creating or commenting on pull requests in GitLab/GitHub.
- Summarizing Jira issues.
- And many more...

Below is an overview of the tools currently supported by Sage.

## List of Tools

- **Calculator**: Perform arithmetic operations and complex calculations.
- **Search Engine**: Query the internet for information.
- **Jira Issue Summarizer**: Summarize issues and projects from Jira.

## Using the Tools

Each tool can be accessed through the Sage Chat interface by typing a command or question related to the tool's function. For example:

- To use the calculator, you might ask, "What is the square root of 16?"
- To search the web, you could type, "Search for the latest AI research papers."

## Adding New Tools

To add a new tool to Sage:

1. Define the tool in the `config.toml` file with the necessary parameters.
2. Restart Sage to apply the changes.

...

Remember to check back regularly for updates as new tools are added to the Sage platform.
