from pydantic import (
    BaseModel,
    model_validator,
    root_validator,
    validator,
    SecretStr,
    field_serializer,
)
from typing import Literal, Optional, List
from pathlib import Path
import os
from logging import getLevelName

from utils.exceptions import ConfigException

sage_base = ".sage"


class Password(BaseModel):
    """
    Password Base Model.
    """

    password: Optional[SecretStr] = None

    @field_serializer("password", when_used="json")
    def dump_secret(self, v):
        return v.get_secret_value() if v else None


class SourceData(Password):
    """
    Source Data Model. Inherits Password.
    """

    username: str
    server: str

    @validator("server", pre=True, always=True)
    def add_https_to_server(cls, server):
        if not server.startswith("https://"):
            server = "https://" + server
        return server


class ConfluenceModel(SourceData):
    """
    Confluence Data Model. Inherits SourceData.
    """

    spaces: List[str]

    @validator("password", pre=True, always=True)
    def set_password(cls, v):
        password = v or os.getenv("CONFLUENCE_PASSWORD") or os.getenv("JIRA_PASSWORD")
        if password is None:
            raise ConfigException(
                "The Confluence password is missing. \
                    Please add it via an env variable or to the config - 'CONFLUENCE_PASSWORD'"
            )
        return password


class GitlabModel(SourceData):
    """
    Gitlab Data Model. Inherits SourceData.
    """

    username: Optional[str] = None
    groups: List[str] = []
    projects: List[str] = []

    @root_validator(skip_on_failure=True)
    def validate_projects_groups(values):
        """A validator that raises an error if both variable "projects" and "groups" are empty after initialization"""
        if not values.get("projects") and not values.get("groups"):
            raise ConfigException(
                "The Gitlab projects or groups are missing from the configuration file"
            )
        return values

    @validator("password", pre=True, always=True)
    def set_password(cls, v):
        password = v or os.getenv("GITLAB_PASSWORD")
        if password is None:
            raise ConfigException(
                "The Gitlab password is missing. Please add it via an env variable or to the config - 'GITLAB_PASSWORD'"
            )
        return password


class Web(BaseModel):
    """
    Web Model.
    """

    links: List[str]
    nested: bool
    ssl_verify: bool = True


class Files(BaseModel):
    """
    Files Model.
    """

    paths: List[str]


class Source(BaseModel):
    """
    Source Model.
    """

    top_k: Optional[int] = 20
    """The number of vector queries to return in the retriever"""
    confluence: Optional[ConfluenceModel] = None
    gitlab: Optional[GitlabModel] = None
    web: Optional[Web] = None
    files: Optional[Files] = None


class Jira_Config(Password):
    """
    Jira Model. Inherits Password.
    """

    url: str
    username: str
    polling_interval: int
    project: str
    status_todo: str

    @validator("password", pre=True, always=True)
    def set_password(cls, v):
        password = v or os.getenv("JIRA_PASSWORD")
        if password is None:
            raise ConfigException(
                "The JIRA password is missing. \
                    Please add it via an env variable or to the config - 'JIRA_PASSWORD'"
            )
        return password


class Core(BaseModel):
    """
    Core Model.
    """

    data_dir: Optional[str | Path] = Path.home() / sage_base
    logging_level: str | int = "INFO"
    user_agent: str = "codesage.ai"

    @validator("logging_level", pre=True, always=True)
    def set_logging_level(cls, v):
        return getLevelName(v)


class EmbeddingCore(BaseModel):
    """The Embedding Model schema"""

    name: str
    revision: str


class EmbeddingsConfig(BaseModel):
    openai: Optional[EmbeddingCore]
    jina: Optional[EmbeddingCore]
    type: Literal["jina", "openai"]


class CohereReRanker(Password):
    """The Cohere rerank schema"""

    name: str

    @validator("password", pre=True, always=True)
    def set_password(cls, v):
        password = v or os.getenv("COHERE_PASSWORD") or os.getenv("COHERE_API_KEY")
        if password is None:
            raise ConfigException(
                "The COHERE API KEY or password is missing. \
                    Please add it via an env variable or to the config - 'COHERE_PASSWORD'"
            )
        return password


class HuggingFaceReRanker(BaseModel):
    """The HuggingFace schema"""

    name: str
    revision: str


class ReRankerConfig(BaseModel):
    """Reranker config schema"""

    cohere: Optional[CohereReRanker]
    huggingface: Optional[HuggingFaceReRanker]
    type: Literal["cohere", "huggingface"]
    top_n: int = 5


class LLMCore(Password):
    """The LLM Core Model schema"""

    name: str
    endpoint: str
    revision: Optional[str] = None


class AzureLLM(LLMCore):
    """The LLM Core Model schema"""

    @validator("password", pre=True, always=True)
    def set_password(cls, v):
        password = v or os.getenv("AZURE_PASSWORD") or os.getenv("AZURE_OPENAI_API_KEY")
        if password is None:
            raise ConfigException(
                "The AZURE_OPENAI_API_KEY or password is missing. \
                    Please add it via an env variable or to the config password field - 'AZURE_OPENAI_API_KEY'"
            )
        return password


class LLMConfig(BaseModel):
    """The configuration for LLM models"""

    azure: Optional[AzureLLM]
    ollama: Optional[LLMCore]
    type: Literal["azure", "ollama"]


class UploadConfig(BaseModel):
    """The configuration for the Chat upload mode"""

    max_size_mb: Optional[int] = 10
    max_files: Optional[int] = 5
    timeout: Optional[int] = 300


class Config(BaseModel):
    """
    Config Model.
    """

    core: Core
    upload: Optional[UploadConfig] = UploadConfig()
    jira: Jira_Config
    source: Source
    reranker: Optional[ReRankerConfig] = None
    embedding: EmbeddingsConfig
    llm: LLMConfig
