import os
import sys
import toml
from langchain_community.chat_models import ChatOllama
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from pydantic import ValidationError
from pathlib import Path
from utils.exceptions import ConfigException
from utils.validator import Config
from utils.logger import CustomLogger
from utils.supports import JinaAIEmebeddings

# Load the configuration file only once
config_path = os.getenv("SAGE_CONFIG_PATH", "config.toml")
assets_dir = Path(__file__).parent / "assets"

app_name = "codesage.ai"
logger = CustomLogger(name=app_name)

try:
    config = toml.load(config_path)
    validated_config = Config(**config)
    core_config = validated_config.core
    jira_config = validated_config.jira
    sources_config = validated_config.source
except ValidationError as error:
    logger.error(f"The configuration file is not valid - {str(error)}", exc_info=False)
    sys.exit(1)
except (FileNotFoundError, KeyError) as error:
    logger.error(
        f"The required configuration key or file is not found - {str(error)}",
        exc_info=False,
    )
    sys.exit(1)

# Create the main data directory
Path(core_config.data_dir).mkdir(exist_ok=True)

# Update the logging level
logger.setLevel(core_config.logging_level)

JIRA_QUERY = 'project = "{project}" and status = "{status}" and assignee = "{assignee}" ORDER BY created ASC'

# Load the model
if validated_config.llm.type == "azure":
    azure_config = validated_config.llm.azure

    LLM_MODEL = AzureChatOpenAI(
        azure_endpoint=azure_config.endpoint,
        api_version=azure_config.revision,
        azure_deployment=azure_config.name,
        api_key=azure_config.password.get_secret_value(),
        streaming=True,
    )

elif validated_config.llm.type == "ollama":
    ollama_config = validated_config.llm.azure

    LLM_MODEL = ChatOllama(
        base_url=ollama_config.endpoint, model=ollama_config.name, streaming=True
    )

# Load the Embeddings model
if validated_config.embedding.type == "jina":
    jina_config = validated_config.embedding.jina

    EMBEDDING_MODEL = JinaAIEmebeddings(
        cache_dir=str(core_config.data_dir) + "/models",
        jina_model=jina_config.name,
        revision=jina_config.revision,
    )
elif validated_config.embedding.type == "openai":
    openai_config = validated_config.embedding.openai

    EMBEDDING_MODEL = OpenAIEmbeddings(
        deployment=openai_config.name, openai_api_version=openai_config.revision
    )
