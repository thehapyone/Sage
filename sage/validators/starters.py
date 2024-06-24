# starters.py
from typing import List, Optional

from chainlit import Starter
from pydantic import (
    BaseModel,
    Field,
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
