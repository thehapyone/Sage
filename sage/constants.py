import os
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
    raise ConfigException(f"The configuration file is not valid - {str(error)}")
except (FileNotFoundError, KeyError) as error:
    # Raise a ConfigException for both file not found and missing key errors
    raise ConfigException(
        f"The required configuration key or file is not found - {str(error)}"
    )

# Create the main data directory
Path(core_config.data_dir).mkdir(exist_ok=True)

# Update the logging level
logger.setLevel(core_config.logging_level)

JIRA_QUERY = 'project = "{project}" and status = "{status}" and assignee = "{assignee}" ORDER BY created ASC'

# Load the model
DEPLOYMENT_NAME = "gpt4-128k"
LLM_MODEL = AzureChatOpenAI(deployment_name=DEPLOYMENT_NAME)

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
else:
    raise ValidationError(
        f"Embedding type {validated_config.embedding.type} is not allowed"
    )

# llm = ChatOllama(model="llama2:13b",
#                  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
