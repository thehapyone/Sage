
import pytest
from pydantic import ValidationError

from sage.utils.exceptions import ConfigException
from sage.validators.starters import (
    StarterConfig,
    Starters,
)

###########################################################################
################ Unit Tests for the StartersConfig ########################


@pytest.fixture
def starter_data():
    return {
        "label": "Daily Summary",
        "message": "Show me the daily summary",
        "icon": "https://example.com/icon.png",
    }


def test_valid_starter_config(starter_data):
    starter = StarterConfig(**starter_data)
    assert starter.label == starter_data["label"]
    assert starter.message == starter_data["message"] + " %none%"
    assert starter.icon == starter_data["icon"]


def test_starter_config_missing_fields():
    with pytest.raises(ConfigException):
        StarterConfig(label="Weather Report")


def test_starter_config_with_source():
    starter = StarterConfig(
        label="Weather Report", message="What is the weather?", source="yahoo"
    )
    assert starter.message == "What is the weather? %yahoo%"


def test_valid_starter_config_with_starters(starter_data):
    test_starter = StarterConfig(**starter_data)
    starter_config = Starters(starters=[test_starter])
    assert len(starter_config.starters) == 1
    assert starter_config.starters[0].label == starter_data["label"]
    assert starter_config.starters[0].message == starter_data["message"] + " %none%"
    assert starter_config.starters[0].icon == starter_data["icon"]


def test_starter_config_empty_starters_list():
    starter_config = Starters(starters=[])
    assert len(starter_config.starters) == 0


def test_starters_invalid_type():
    # Starters should be a list
    with pytest.raises(ValidationError):
        Starters(starters="not a list")
