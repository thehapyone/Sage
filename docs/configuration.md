# Configuration Guide for CodeSage

This document provides detailed instructions on how to configure the CodeSage application to suit your specific needs and environment. CodeSage uses a TOML configuration file to manage various settings and preferences.

## Configuration File

The primary way to configure CodeSage is through a `config.toml` file located in the root directory of the application. This file contains various sections that correspond to different parts of the application, such as core settings, source connections, and model configurations.

Path to the configuration file `config.toml` can be set via the environment variable `SAGE_CONFIG_PATH=/path/to/config.toml`

### Core Configuration

The `[core]` section of the configuration file specifies the fundamental settings of the application, such as the data directory and logging level.

```toml
[core]
data_dir = "/path/to/data/directory"
logging_level = "INFO" # Can be DEBUG, INFO, WARNING, ERROR, or CRITICAL
user_agent = "codesage.ai"
```

## Upload Configuration

The [upload] section defines the settings related to file uploads, such as maximum file size and upload timeout.
```toml
[upload]
max_size_mb = 10
max_files = 5
timeout = 300
```

## Azure Configuration

The [azure] section defines the settings related to azure configuration, such as endpoint and versions.
```toml
[azure]
endpoint = "https://your_azure_endpoint"
password = "your_azure_password"
revision = "your_azure_api_version"
```

## OpenAI Configuration

The [openai] section defines the settings for common openai settings.
```toml
[openai]
password = "your_openai_token"
organization = "your_organization"
```

## Jira Configuration

The [jira] section configures the connection to your Jira instance, including the URL, user credentials, and polling interval.

```toml
[jira]
url = "https://yourcompany.atlassian.net/"
username = "your_jira_username"
polling_interval = 300
project = "YOURPROJECT"
status_todo = "To Do"
```

## Source Configuration

The [source] section allows you to configure various data sources that CodeSage can interact with, such as Confluence, GitLab, web links, and files.

```toml
[source]
# The number of documents to retrieve from the retriever system
top_k = 10
# Schedule in cron format: minute, hour, day (month), month, day (week)  
# Example: "0 * * * *" means every hour at the 0th minute  
refresh_schedule = "0 0 * * SUN" # Every Sunday at midnight  

[source.confluence]
username = "your_confluence_username"
server = "https://yourcompany.atlassian.net/wiki"
spaces = ["SPACE1", "SPACE2"]

[source.gitlab]
server = "https://gitlab.com"
groups = ["group1", "group2"]
projects = ["project1", "project2"]

[source.web]
links = ["https://example.com", "https://anotherexample.com"]
nested = true
ssl_verify = true
# Adds an optional Basic Auth for the web links. E.g Directory listing servers
username = "username"
password = "password"

[source.files]
paths = ["/path/to/file1", "/path/to/file2"]
```

## Embeddings and LLM Configuration

The [embedding] and [llm] sections configure the embedding engines and Large Language Models (LLMs) used by Sage

```toml
[embedding]
type = "openai"

[llm]
type = "azure"
```

## Environment Variables for Sensitive Credentials

For security reasons, it is crucial to avoid storing sensitive credentials such as passwords and API keys directly in the configuration file. Instead, CodeSage is designed to retrieve these credentials from environment variables.

The `validator.py` file contains logic to fall back to environment variables if a password is not provided in the `config.toml` file. This approach allows you to keep sensitive information secure.

To set an environment variable, you can use the following commands in your terminal:

For Unix-based systems (Linux/macOS):

```shell
export CONFLUENCE_PASSWORD="your_confluence_password"
export GITLAB_PASSWORD="your_gitlab_password"
export JIRA_PASSWORD="your_jira_password"
# ... and so on for other services
```

For Windows:

```shell
set CONFLUENCE_PASSWORD=your_confluence_password
set GITLAB_PASSWORD=your_gitlab_password
set JIRA_PASSWORD=your_jira_password
# ... and so on for other services
```

Make sure to replace `your_confluence_password`, `your_gitlab_password`, and `your_jira_password` with your actual credentials.

After setting the environment variables, Sage will automatically use these values when the corresponding fields in the config.toml file are left empty.

## Finalizing Configuration

Once you have edited the `config.toml` file and set the necessary environment variables, save the file and restart the **CodeSage** application to apply the changes.

For more information on the specific fields and acceptable values in the configuration file, please refer to the inline documentation within the [validator.py](../sage/utils/validator.py) file or the sample [config.toml](../sage/config.toml) provided with the application.

## Support

If you encounter any issues or have questions about configuring CodeSage, please file an issue on our [GitHub Issues](https://github.com/thehapyone/codesage/issues) page.
