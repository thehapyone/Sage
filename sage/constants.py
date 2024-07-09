import asyncio
import os
import sys

import toml
from pydantic import ValidationError

from sage.utils.exceptions import ConfigException
from sage.utils.logger import CustomLogger
from sage.utils.supports import load_embedding_model, load_language_model
from sage.validators.config_toml import Config
from sage.validators.crew_ai import load_and_validate_agents_yaml
from sage.validators.starters import load_and_validate_starters_yaml

# Load the configuration file only once
config_path = os.getenv("SAGE_CONFIG_PATH", "config.toml")

# Initialize the logger
app_name = "codesage.ai"
logger = CustomLogger(name=app_name)


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
    chat_starters = load_and_validate_starters_yaml(core_config.starters_path)
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

# Set the Large languageModel and Embeddings
try:
    LLM_MODEL = load_language_model(logger, validated_config.llm.model)
    EMBEDDING_MODEL, EMBED_DIMENSION = load_embedding_model(logger, validated_config)
except Exception as error:
    logger.error(
        f"Error loading models: {str(error)}",
        exc_info=False,
    )
    sys.exit(2)

# Define the path for the sentinel file
SENTINEL_PATH = validated_config.core.data_dir / "data_updated.flag"

# Load any available agents
try:
    agents_crew = load_and_validate_agents_yaml(core_config.agents_dir, LLM_MODEL)
except (ValidationError, ConfigException) as error:
    logger.error(f"The configuration file is not valid - {str(error)}", exc_info=False)
    sys.exit(3)
