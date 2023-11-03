import os
import toml
from langchain.chat_models import AzureChatOpenAI, ChatOllama
from langchain.embeddings import OpenAIEmbeddings
from pydantic import ValidationError
from pathlib import Path
from utils.exceptions import ConfigException
from utils.validator import Config

# Load the configuration file only once
config_path = os.getenv("SAGE_CONFIG_PATH", "config.toml")

try:
    config = toml.load(config_path)
    validated_config = Config(**config)
    core_config = validated_config.core
    jira_config = validated_config.jira
    sources_config = validated_config.source
except ValidationError as error:
    raise ConfigException(
        f"The configuration file is not valid - {str(error)}")
except (FileNotFoundError, KeyError) as error:
    # Raise a ConfigException for both file not found and missing key errors
    raise ConfigException(
        f"The required configuration key or file is not found - {str(error)}")

# Create the main data directory
Path(core_config.data_dir).mkdir(exist_ok=True)

JIRA_QUERY = 'project = "{project}" and status = "{status}" and assignee = "{assignee}" ORDER BY created ASC'

# Load the model
DEPLOYMENT_NAME = "gpt4-8k"
LLM_MODEL = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME)

# Document Embeddings
EMBEDDING_MODEL = OpenAIEmbeddings(
    deployment="ada-embeddings",
)

# llm = ChatOllama(model="llama2:13b",
#                  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
