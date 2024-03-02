from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
    SecretStr,
    field_serializer,
    Field,
    PositiveInt,
)
from typing import Literal, Optional, List
from pathlib import Path
import os
from logging import getLevelName

from sage.utils.exceptions import ConfigException

sage_base = ".sage"


class UploadConfig(BaseModel):
    """The configuration for the Chat upload mode"""

    max_size_mb: Optional[PositiveInt] = 10
    max_files: Optional[PositiveInt] = 5
    timeout: Optional[PositiveInt] = 300


class Password(BaseModel):
    """
    Password Base Model.
    """

    password: Optional[SecretStr] = None

    @field_serializer("password", when_used="json")
    def dump_secret(self, v):
        return v.get_secret_value() if v else None


class AzureConfig(Password):
    """Common Azure Configurations"""

    endpoint: str
    revision: str

    @field_validator("endpoint")
    @classmethod
    def add_https_to_endpoint(cls, v: str):
        if not v.startswith("https://"):
            return f"https://{v}"
        return v

    @model_validator(mode="after")
    def set_password(self) -> "Password":
        if self.password is None:
            if password_env := os.getenv("AZURE_PASSWORD") or os.getenv(
                "AZURE_OPENAI_API_KEY"
            ):
                self.password = SecretStr(password_env)
            else:
                raise ConfigException(
                    (
                        "The AZURE_OPENAI_API_KEY | AZURE_PASSWORD | config password is missing. "
                        "Please add it via an env variable or to the config password field."
                    )
                )
        return self


class OpenAIConfig(Password):
    """Common OpenAI Configurations"""

    organization: Optional[str] = None

    @model_validator(mode="after")
    def set_password(self) -> "Password":
        if self.password is None:
            if password_env := os.getenv("OPENAI_PASSWORD") or os.getenv(
                "OPENAI_API_KEY"
            ):
                self.password = SecretStr(password_env)
            else:
                raise ConfigException(
                    (
                        "The OPENAI_API_KEY | OPENAI_PASSWORD | config password is missing. "
                        "Please add it via an env variable or to the config password field."
                    )
                )
        return self


class ModelValidateType(BaseModel):
    """Base Model for validating the type data"""

    @model_validator(mode="before")
    @classmethod
    def validate_config_is_available_for_type(cls, data: dict) -> dict:
        """A validator that raises an error the config data for the type is missing"""
        model_type = data["type"]
        if not data.get(model_type):
            raise ConfigException(
                f"The Config data for type '{model_type}' is missing."
            )
        return data


class SourceData(Password):
    """
    Source Data Model. Inherits Password.
    """

    username: str
    server: str

    @field_validator("server")
    @classmethod
    def add_https_to_server(cls, v: str):
        if not v.startswith("https://"):
            return f"https://{v}"
        return v


class ConfluenceModel(SourceData):
    """
    Confluence Data Model. Inherits SourceData.
    """

    spaces: List[str]

    @model_validator(mode="after")
    def set_password(self) -> "Password":
        if self.password is None:
            if password_env := os.getenv("CONFLUENCE_PASSWORD") or os.getenv(
                "JIRA_PASSWORD"
            ):
                self.password = SecretStr(password_env)
            else:
                raise ConfigException(
                    (
                        "The CONFLUENCE_PASSWORD | JIRA_PASSWORD | config password is missing. "
                        "Please add it via an env variable or to the config password field."
                    )
                )
        return self


class GitlabModel(SourceData):
    """
    Gitlab Data Model. Inherits SourceData.
    """

    username: Optional[str] = None
    groups: List[str] = []
    projects: List[str] = []

    @model_validator(mode="before")
    @classmethod
    def validate_projects_groups(cls, values: dict) -> dict:
        """A validator that raises an error if both variable "projects" and "groups" are empty after initialization"""
        if not values.get("projects") and not values.get("groups"):
            raise ConfigException(
                "The Gitlab projects or groups are missing from the configuration file"
            )
        return values

    @model_validator(mode="after")
    def set_password(self) -> "Password":
        if self.password is None:
            if password_env := os.getenv("GITLAB_PASSWORD"):
                self.password = SecretStr(password_env)
            else:
                raise ConfigException(
                    (
                        "The GITLAB_PASSWORD | config password is missing. "
                        "Please add it via an env variable or to the config password field."
                    )
                )
        return self


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

    @model_validator(mode="after")
    def set_password(self) -> "Password":
        if self.password is None:
            if password_env := os.getenv("JIRA_PASSWORD"):
                self.password = SecretStr(password_env)
            else:
                raise ConfigException(
                    (
                        "The JIRA_PASSWORD | config password is missing. "
                        "Please add it via an env variable or to the config password field."
                    )
                )
        return self


class Core(BaseModel):
    """
    Core Model.
    """

    data_dir: Optional[str | Path] = Path.home() / sage_base
    logging_level: str | int = "INFO"
    user_agent: str = "codesage.ai"

    @field_validator("logging_level")
    @classmethod
    def set_logging_level(cls, v: str | int):
        return getLevelName(v)


class EmbeddingCore(BaseModel):
    """The Embedding Model schema"""

    name: str
    revision: Optional[str] = None


class EmbeddingsConfig(ModelValidateType):
    azure: Optional[EmbeddingCore] = None
    openai: Optional[EmbeddingCore] = None
    jina: Optional[EmbeddingCore] = None
    type: Literal["jina", "azure", "openai"]


class CohereReRanker(Password):
    """The Cohere rerank schema"""

    name: str

    @model_validator(mode="after")
    def set_password(self) -> "Password":
        if self.password is None:
            if password_env := os.getenv("COHERE_API_KEY"):
                self.password = SecretStr(password_env)
            else:
                raise ConfigException(
                    (
                        "The COHERE_PASSWORD | config password is missing. "
                        "Please add it via an env variable or to the config password field."
                    )
                )
        return self


class HuggingFaceReRanker(BaseModel):
    """The HuggingFace schema"""

    name: str
    revision: str


class ReRankerConfig(ModelValidateType):
    """Reranker config schema"""

    cohere: Optional[CohereReRanker] = None
    huggingface: Optional[HuggingFaceReRanker] = None
    type: Literal["cohere", "huggingface"]
    top_n: int = 5


class LLMCore(Password):
    """The LLM Core Model schema"""

    name: str
    endpoint: Optional[str] = None
    revision: Optional[str] = None


class LLMConfig(ModelValidateType):
    """The configuration for LLM models"""

    azure: Optional[LLMCore] = None
    openai: Optional[LLMCore] = None
    ollama: Optional[LLMCore] = None
    type: Literal["azure", "ollama", "openai"]


class Config(BaseModel):
    """
    Config Model.
    """

    core: Optional[Core] = Field(
        default=Core(), description="Sage's main configuration"
    )
    upload: Optional[UploadConfig] = UploadConfig()
    jira: Optional[Jira_Config] = Field(default=None, description="Jira configuration")
    azure: AzureConfig = Field(default=None, description="Shared Azure configuration")
    openai: OpenAIConfig = Field(
        default=None, description="Shared OpenAI configuration"
    )
    source: Source
    reranker: Optional[ReRankerConfig] = None
    embedding: EmbeddingsConfig
    llm: LLMConfig

    @model_validator(mode="before")
    @classmethod
    def check_provider_configs(cls, values: dict) -> dict:
        """Ensure the appropriate provider field is not empty when using specific providers"""
        embedding = values.get("embedding")
        llm = values.get("llm")

        # Define a list of providers to validate
        providers = [
            {"type": "azure", "config": "azure"},
            {"type": "openai", "config": "openai"},
        ]

        for provider in providers:
            provider_type = provider["type"]
            provider_config = provider["config"]

            # Check if embedding or llm type matches the provider type and ensure the corresponding config is provided
            if (
                (embedding and embedding["type"] == provider_type)
                or (llm and llm["type"] == provider_type)
            ) and not values.get(provider_config):
                raise ConfigException(
                    f"{provider_type.capitalize()} configuration must be provided when embedding or llm type is '{provider_type}'"
                )

        return values
