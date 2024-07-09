# starters.py
from typing import List, Optional

import yaml
from chainlit import Starter
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
)

from sage.utils.exceptions import ConfigException


class StarterConfig(BaseModel, Starter):
    label: str
    message: str
    icon: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def combine_message_and_source(cls, values: dict) -> dict:
        """Combine the message and source to create a new message-source data"""
        message: str = values.get("message")
        if not message:
            raise ConfigException("The message field is missing")
        source: str = values.get("source", "none").strip()
        values["message"] = f"{message} %{source}%".strip()
        return values


class Starters(BaseModel):
    """Starters config model"""

    starters: List[StarterConfig] = Field(default=[])


def load_and_validate_starters_yaml(file_path: str | None) -> Starters:
    """Validates the chat starters yaml is valid"""
    if file_path is None:
        return []
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            if data is None:
                raise ConfigException("Starters content cannot be empty")
    except FileNotFoundError:
        raise ConfigException(f"Starters file not found at path: {file_path}")
    except yaml.YAMLError as exc:
        raise ConfigException(f"Error parsing YAML: {exc}")

    # Validate data with pydantic
    try:
        starters_config = Starters(**data).starters
    except Exception as ve:
        raise ConfigException(f"Validation error in starters YAML: {ve}")

    return starters_config
