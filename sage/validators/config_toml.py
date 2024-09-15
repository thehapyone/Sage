import asyncio
import base64
import os
import logging
from typing import List, Literal, Optional

from anyio import Path
from croniter import croniter
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)

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

    username: Optional[str] = None
    server: Optional[str] = None
    max_concurrency: Optional[int] = 10
    """The concurrency limit for processing sources"""

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


class Web(SourceData):
    """
    Web Model.
    """

    links: List[str]
    nested: bool
    ssl_verify: bool = True
    headers: Optional[dict] = {}
    max_depth: Optional[int] = 10
    """Max depth to go for finding the child links"""

    @model_validator(mode="after")
    def generate_header(self) -> "Web":
        if not self.username and not self.password:
            return self
        elif (self.username and not self.password) or (
            self.password and not self.username
        ):
            raise ConfigException(
                "Both a Username and Password are required for the Web Source"
            )
        else:
            credentials = f"{self.username}:{self.password.get_secret_value()}"
            encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode(
                "utf-8"
            )
            self.headers.update({"Authorization": f"Basic {encoded_credentials}"})
        return self


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
    refresh_schedule: Optional[str] = None
    confluence: Optional[ConfluenceModel] = None
    gitlab: Optional[GitlabModel] = None
    web: Optional[Web] = None
    files: Optional[Files] = None

    @field_validator("refresh_schedule")
    @classmethod
    def validate_refresh_value(cls, v: str):
        if v is not None and not croniter.is_valid(v):
            raise ValueError("The value of refresh_schedule is not a valid cron syntax")
        return v


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_dir: Path
    logging_level: str | int = "INFO"
    user_agent: str = "codesage.ai"

    starters_path: Optional[str] = None
    agents_dir: Optional[str] = None

    disable_crewai_telemetry: Optional[bool] = True

    @field_validator("logging_level")
    @classmethod
    def set_logging_level(cls, v: str | int):
        return logging.getLevelName(v)

    @model_validator(mode="before")
    @classmethod
    def set_data_dir(cls, values: dict) -> dict:
        """Sets the default data dir if not set and convert to Path"""
        _data_dir = values.get("data_dir")
        if _data_dir is None:
            default_data_dir = asyncio.run(Path.home()) / sage_base
            values["data_dir"] = default_data_dir
            return values
        values["data_dir"] = Path(_data_dir)
        return values

    @model_validator(mode="after")
    def set_crewai_telemetry(self) -> "Core":
        """Enable or disable CrewAI telemetry by setting the correct environment variable."""
        os.environ.setdefault("OTEL_SDK_DISABLED", str(self.disable_crewai_telemetry).lower())
        return self

class EmbeddingsConfig(BaseModel):
    type: Literal["litellm", "huggingface"]
    model: str
    dimension: Optional[int] = None


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
                        "The COHERE_API_KEY | config password is missing. "
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


class LLMConfig(BaseModel):
    """The configuration for LLM models"""

    model: str


class Config(BaseModel):
    """
    Config Model.
    """

    core: Optional[Core] = Field(
        default=Core(), description="Sage's main configuration"
    )
    upload: Optional[UploadConfig] = UploadConfig()
    jira: Optional[Jira_Config] = Field(default=None, description="Jira configuration")
    source: Source
    reranker: Optional[ReRankerConfig] = None
    embedding: EmbeddingsConfig
    llm: LLMConfig
