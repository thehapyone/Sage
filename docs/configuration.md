# Configuration Guide for Sage

This document provides detailed instructions on how to configure the Sage application to suit your specific needs and environment. Sage uses a TOML configuration file to manage various settings and preferences.

## Starters Configuration

Configuring starters is straightforward and requires editing a YAML configuration file. Each starter consists of a user-friendly label, a pre-defined message that sets the context or action for the AI, an optional icon to visually represent the action, and an optional source identifier that specifies the context for the message.

Here's how you can configure starters:

1. Open the `starters.yaml` file in your code editor.
2. Define each starter with the following attributes:
   - `label`: A short, descriptive name for the quick-action prompt.
   - `message`: The initial message or command that the starter will send to Sage.
   - `icon`: A URL to an image that will appear as an icon for the starter in the UI.
   - `source`: (Optional) A string that denotes the source to be concatenated with the `message` when sent to Sage.

Starters are configured via a yaml file that is set in the sage config.toml file under the core section like below

```toml
[core]
...
starters_path = "/path/to/starters.yaml"
...
```

### Example Starters Configuration

Below is an example of how starters can be configured in the `starters.yaml` file:

```yaml
starters:
  - label: "Casual Wedding Invite"
    message: "Draft a casual message to invite a friend as my guest to a wedding next month, ensuring it feels light-hearted and stress-free."
    icon: "https://picsum.photos/200"
  - label: "Superconductors Simplified"
    message: "Describe superconductors in a way that a five-year-old could understand."
    icon: "https://picsum.photos/300"
    source: "Confluence: SF Space Details"
  - label: "Python Email Automation Script"
    message: "Generate a Python script for automating daily email reports."
    icon: "https://picsum.photos/400"
    source: "Confluence: Development Docs"
```
When the user selects a starter, the message (concatenated with the source, if provided) is sent to Sage, triggering the respective action or query as if the user had typed the message themself.

## Agents & Crew Configuration

Configuring agents/crews for utilization is straightforward. Each Crew is defined in its own configuration YAML file, and all Crews are placed in a specified directory.

### Steps to Configure Agents & Crews

1. **Create Crews Directory**
    - First, define and create a directory to hold all your Crews. For example:
      ```sh
      mkdir agents
      ```
    
2. **Create Crew Files**
    - Create one or more crew files and add them to the created directory.

3. **Update Sage Configuration**
    - Update the Sage TOML configuration file to point to the location of the agents directory under the core section:
      ```toml
      [core]
      ...
      agents_dir = "/path/to/agents/directory"
      ...
      ```

**Note**: Sage will treat any `.yaml` or `.yml` files in that directory as potential crew configurations.

### Crew Layout

All options that can be used in the Crew Layout are based on the CrewAI definition. Please visit the [CrewAI Documentation](https://docs.crewai.com/) to explore all available options. 

The following options are dynamically injected when the application starts
 - if `memory:True`, then all memory instances are automatically initialized
 - If no LLM are defined for the crew (`manager_llm`) and the agents (`llm`), the application injects the default sage LLM for all agents and crews

This structure allows you to specify the agents and tasks for your crew, ensuring each agent has a clear role, goal, and backstory, and each task is clearly described and assigned to the appropriate agent.

#### Basic Template

```yaml
name: <string>  # The name of the crew

agents:  # A list of agents for the crew
  - role: <string>  # Role of the agent (must be unique within the crew)
    goal: <string>  # Goal of the agent
    backstory: <string>  # Backstory of the agent
    ...

tasks:  # A list of tasks
  - description: <string>  # The task description that will be assigned to the agent. Must include an {input} placeholder for chat inputs.
    agent: <string>  # The agent role assigned to this task
    expected_output: <string>  # Description of the agent's expected output
    ...
```

Example Crew Configuration

```yaml
name: GameStartup  # The name of the crew

agents:  # A list of agents for the crew
  - role: Game Designer
    goal: Design engaging and innovative game mechanics
    backstory: An expert with over a decade of experience in game design and is known for creating unique and popular game mechanics.

  - role: Marketing Strategist
    goal: Develop a marketing strategy to launch the game successfully
    backstory: You have worked on several successful game launches and excel at creating buzz and engaging the gaming community.

tasks:  # A list of tasks
  - description: Research and design the core mechanics of games. This is the game instructions: {input}
    agent: Game Designer
    expected_output: A detailed report on the game mechanics including sketches and flowcharts

  - description: Conduct a competitor analysis for similar games. Game details: {input}
    agent: Marketing Strategist
    expected_output: A report on competitor strengths, weaknesses, and market positioning

  - description: Develop the initial concept art and prototypes for the game. Game details: {input}
    agent: Game Designer
    expected_output: Concept art and prototype sketches

  - description: Create a comprehensive marketing plan for the game launch. Game details: {input}
    agent: Marketing Strategist
    expected_output: A complete marketing strategy document with timelines, channels, and key messages
```

## Configuration File

The primary way to configure Sage is through a `config.toml` file located in the root directory of the application. This file contains various sections that correspond to different parts of the application, such as core settings, source connections, and model configurations.

Path to the configuration file `config.toml` can be set via the environment variable `SAGE_CONFIG_PATH=/path/to/config.toml`

### Core Configuration

The `[core]` section of the configuration file specifies the fundamental settings of the application, such as the data directory and logging level.

```toml
[core]
data_dir = "/path/to/data/directory"
logging_level = "INFO" # Can be DEBUG, INFO, WARNING, ERROR, or CRITICAL
user_agent = "Sage.ai"
starters_path = "/path/to/starters.yaml"
agents_dir = "/path/to/agents/directory"
disable_crewai_telemetry = true # Enables or disable the default crewai telemetry. Default is True
```

## Upload Configuration

The [upload] section defines the settings related to file uploads, such as maximum file size and upload timeout.
```toml
[upload]
max_size_mb = 10
max_files = 5
timeout = 300
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

The [source] section allows you to configure various data sources that Sage can interact with, such as Confluence, GitLab, web links, and files.

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
## LLM and ReRanker Configuration are based on LiteLLM providers - https://docs.litellm.ai/docs/providers/
[llm]
model = "gpt-4-turbo"

[embedding]
type = "huggingface"
model = "jinaai/jina-embeddings-v2-base-en"

[reranker]
top_n = 5
model = "cohere/rerank-english-v2.0"
#model = "BAAI/bge-reranker-large"
revision = "55611d7bca2a7133960a6d3b71e083071bbfc312"

```

## Environment Variables for Sensitive Credentials

For security reasons, it is crucial to avoid storing sensitive credentials such as passwords and API keys directly in the configuration file. Instead, Sage is designed to retrieve these credentials from environment variables.

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

Once you have edited the `config.toml` file and set the necessary environment variables, save the file and restart the **Sage** application to apply the changes.

For more information on the specific fields and acceptable values in the configuration file, please refer to the inline documentation within the [validator.py](../sage/utils/validator.py) file or the sample [config.toml](../sage/config.toml) provided with the application.

## Support

If you encounter any issues or have questions about configuring Sage, please file an issue on our [GitHub Issues](https://github.com/thehapyone/Sage/issues) page.
