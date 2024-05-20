import asyncio
import os
import sys

import toml
from anyio import Path
from pydantic import ValidationError

from sage.utils.exceptions import ConfigException
from sage.utils.logger import CustomLogger
from sage.utils.supports import CustomLiteLLM, LiteLLMEmbeddings, LocalEmbeddings
from sage.utils.validator import Config

# Load the configuration file only once
config_path = os.getenv("SAGE_CONFIG_PATH", "config.toml")
assets_dir = Path(__file__).parent / "assets"

# Initialize the logger
app_name = "codesage.ai"
logger = CustomLogger(name=app_name)


def load_language_model(model_name: str) -> CustomLiteLLM:
    try:
        llm_model = CustomLiteLLM(model_name=model_name, streaming=True, max_retries=0)
        # Attempts to use the provider to capture any potential missing configuration error
        llm_model.invoke("Hi")
    except Exception as e:
        logger.error(
            f"Error initializing the language model '{model_name}'. Please check all required variables are set. "
            "Provider docs here - https://litellm.vercel.app/docs/providers \n"
            f"Error: {e}"
        )
        sys.exit(2)
    else:
        logger.info(f"Loaded the language model {model_name}")
    return llm_model


def load_embedding_model(config: Config):
    # Embedding model loading and dimension calculation logic...
    if config.embedding.type == "huggingface":
        embedding_model = LocalEmbeddings(
            cache_folder=str(config.core.data_dir) + "/models",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            model_name=config.embedding.model,
        )
    elif config.embedding.type == "litellm":
        embedding_model = LiteLLMEmbeddings(
            model=config.embedding.model, dimensions=config.embedding.dimension
        )
    else:
        raise ValueError(f"Unsupported embedding type: {config.embedding.type}")

    embed_dimension = config.embedding.dimension
    if embed_dimension is None:
        embed_dimension = len(embedding_model.embed_query("dummy"))

    logger.info(f"Loaded the embedding model {config.embedding.model}")

    return embedding_model, embed_dimension


# Create the main data directory
async def create_data_dir() -> None:
    await core_config.data_dir.mkdir(exist_ok=True)


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

# Create the data directory
asyncio.run(create_data_dir())

# Update the logging level
logger.setLevel(core_config.logging_level)

JIRA_QUERY = 'project = "{project}" and status = "{status}" and assignee = "{assignee}" ORDER BY created ASC'

# Set the Large languageModel
LLM_MODEL = load_language_model(validated_config.llm.model)

# Load the Embeddings model
EMBEDDING_MODEL, EMBED_DIMENSION = load_embedding_model(validated_config)

# Define the path for the sentinel file
SENTINEL_PATH = validated_config.core.data_dir / "data_updated.flag"
