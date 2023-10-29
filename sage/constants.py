import os
import toml
from langchain.chat_models import AzureChatOpenAI, ChatOllama

from utils.exceptions import ConfigException

# Load the configuration file only once
config_path = os.getenv("SAGE_CONFIG_PATH", "config.toml")

try:
    config = toml.load(config_path)
    jira_config = config["jira"]
    sources_config = config.get("source")
except (FileNotFoundError, KeyError) as error:
    # Raise a ConfigException for both file not found and missing key errors
    raise ConfigException(
        f"The required configuration key or file is not found - {str(error)}")

# Get the JIRA password from environment variables
jira_password = os.getenv("JIRA_PASSWORD")

if not jira_password:
    raise ConfigException(
        f"The JIRA password is missing. Please add it via an env variable - 'JIRA_PASSWORD'")

jira_config["password"] = jira_password

# Parse the sources section if available


JIRA_QUERY = 'project = "{project}" and status = "{status}" and assignee = "{assignee}" ORDER BY created ASC'

# Load the model
DEPLOYMENT_NAME = "gpt4-8k"
LLM_MODEL = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME)

# llm = ChatOllama(model="llama2:13b",
#                  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
