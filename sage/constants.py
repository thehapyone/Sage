import asyncio
import os
import sys

import toml
from anyio import Path
from langchain_community.chat_models import ChatOllama
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from pydantic import ValidationError

from sage.utils.exceptions import ConfigException
from sage.utils.logger import CustomLogger
from sage.utils.supports import JinaAIEmbeddings
from sage.utils.validator import Config

# Load the configuration file only once
config_path = os.getenv("SAGE_CONFIG_PATH", "config.toml")
assets_dir = Path(__file__).parent / "assets"

# Initialize the logger
app_name = "codesage.ai"
logger = CustomLogger(name=app_name)


def load_language_model(config: Config):
    if config.llm.type == "azure":
        llm_model = AzureChatOpenAI(
            azure_endpoint=config.azure.endpoint,
            api_version=config.azure.revision,
            azure_deployment=config.llm.azure.name,
            api_key=config.azure.password.get_secret_value(),
            streaming=True,
        )
    elif config.llm.type == "ollama":
        llm_model = ChatOllama(
            base_url=config.llm.ollama.endpoint,
            model=config.llm.ollama.name,
            streaming=True,
        )
    elif config.llm.type == "openai":
        llm_model = ChatOpenAI(
            model=config.llm.openai.name,
            api_key=config.openai.password.get_secret_value(),
            organization=config.openai.organization,
            streaming=True,
        )
    else:
        raise ConfigException(f"Unsupported LLM type: {config.llm.type}")

    return llm_model


def load_embedding_model(config: Config):
    # Embedding model loading and dimension calculation logic...
    if config.embedding.type == "jina":
        embedding_model = JinaAIEmbeddings(
            cache_dir=str(config.core.data_dir) + "/models",
            jina_model=config.embedding.jina.name,
            revision=config.embedding.jina.revision,
        )
    elif config.embedding.type == "azure":
        embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=config.embedding.azure.name,
            azure_endpoint=config.azure.endpoint,
            api_version=config.azure.revision,
            api_key=config.azure.password.get_secret_value(),
        )
    elif config.embedding.type == "openai":
        embedding_model = OpenAIEmbeddings(
            model=config.embedding.openai.name,
            api_key=config.openai.password.get_secret_value(),
            organization=config.openai.organization,
        )
    else:
        raise ValueError(f"Unsupported embedding type: {config.embedding.type}")

    embed_dimension = config.embedding.dimension
    if embed_dimension is None:
        embed_dimension = len(embedding_model.embed_query("dummy"))

    return embedding_model, embed_dimension


# Validate the configuration file
try:
    config = toml.load(config_path)
    validated_config = Config(**config)
    core_config = validated_config.core
    jira_config = validated_config.jira
    sources_config = validated_config.source
except (ValidationError, ConfigException) as error:
    logger.error(f"The configuration file is not valid - {str(error)}", exc_info=False)
    sys.exit(1)
except (FileNotFoundError, KeyError) as error:
    logger.error(
        f"The required configuration key or file is not found - {str(error)}",
        exc_info=False,
    )
    sys.exit(1)


# Create the main data directory
async def create_data_dir() -> None:
    await core_config.data_dir.mkdir(exist_ok=True)


asyncio.run(create_data_dir())

# Update the logging level
logger.setLevel(core_config.logging_level)

JIRA_QUERY = 'project = "{project}" and status = "{status}" and assignee = "{assignee}" ORDER BY created ASC'

# Set the Large languageModel
LLM_MODEL = load_language_model(validated_config)

# Load the Embeddings model
EMBEDDING_MODEL, EMBED_DIMENSION = load_embedding_model(validated_config)

# Define the path for the sentinel file
SENTINEL_PATH = validated_config.core.data_dir / "data_updated.flag"
