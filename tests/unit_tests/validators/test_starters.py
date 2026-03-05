from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError

from sage.utils.exceptions import ConfigException
from sage.validators.starters import (
    StarterConfig,
    Starters,
    load_and_validate_starters_yaml,
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


@pytest.fixture
def valid_starters_yaml():
    return """
    starters:
      - label: "Daily Summary"
        message: "Show me the daily summary"
        icon: "https://example.com/icon.png"
    """


@pytest.fixture
def starters_yaml_with_validation_errors():
    return """
    starters:
      - label: "Daily Summary"
    """


def test_load_and_validate_starters_yaml_none():
    starters = load_and_validate_starters_yaml(None)
    assert starters == []


def test_load_and_validate_starters_yaml_file_not_found():
    with pytest.raises(
        ConfigException, match="Starters file not found at path: invalid_path.yaml"
    ):
        load_and_validate_starters_yaml("invalid_path.yaml")


def test_load_and_validate_starters_yaml_yaml_error():
    with patch("builtins.open", mock_open(read_data=": invalid yaml")):
        with pytest.raises(ConfigException) as excinfo:
            load_and_validate_starters_yaml("dummy_path.yaml")
    assert "Error parsing YAML" in str(excinfo.value)


def test_load_and_validate_starters_yaml_empty_content():
    empty_yaml = ""
    with patch("builtins.open", mock_open(read_data=empty_yaml)):
        with pytest.raises(ConfigException) as excinfo:
            load_and_validate_starters_yaml("dummy_path.yaml")
    assert "Starters content cannot be empty" in str(excinfo.value)


def test_load_and_validate_starters_yaml_valid_content(valid_starters_yaml):
    with patch("builtins.open", mock_open(read_data=valid_starters_yaml)):
        starters = load_and_validate_starters_yaml("dummy_path.yaml")
    assert len(starters) == 1
    assert starters[0].label == "Daily Summary"
    assert starters[0].message == "Show me the daily summary %none%"
    assert starters[0].icon == "https://example.com/icon.png"


def test_load_and_validate_starters_yaml_validation_errors(
    starters_yaml_with_validation_errors,
):
    with patch(
        "builtins.open", mock_open(read_data=starters_yaml_with_validation_errors)
    ):
        with pytest.raises(
            ConfigException,
            match="Validation error in starters YAML: The message field is missing",
        ):
            load_and_validate_starters_yaml("dummy_path.yaml")
