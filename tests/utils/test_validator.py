# test_validator.py

import pytest
from pydantic import ValidationError
from sage.utils.validator import AzureConfig, OpenAIConfig, ConfigException, Jira_Config
import os


# Fixture to set environment variables for tests
@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_PASSWORD", "azure_secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_secret")
    monkeypatch.setenv("JIRA_PASSWORD", "jira_secret")


# Test AzureConfig
def test_azure_config_with_explicit_password():
    config = AzureConfig(
        endpoint="example.com", revision="v1", password="explicit_secret"
    )
    assert config.endpoint == "https://example.com"
    assert config.password.get_secret_value() == "explicit_secret"


def test_azure_config_with_env_password():
    config = AzureConfig(endpoint="example.com", revision="v1")
    assert config.password.get_secret_value() == "azure_secret"


def test_azure_config_without_password():
    with pytest.raises(ConfigException):
        AzureConfig(endpoint="example.com", revision="v1", password=None)


# Test OpenAIConfig
def test_openai_config_with_explicit_password():
    config = OpenAIConfig(password="explicit_secret")
    assert config.password.get_secret_value() == "explicit_secret"


def test_openai_config_with_env_password():
    config = OpenAIConfig()
    assert config.password.get_secret_value() == "openai_secret"


def test_openai_config_without_password():
    with pytest.raises(ConfigException):
        OpenAIConfig(password=None)


# Test Jira_Config
def test_jira_config_with_explicit_password():
    config = Jira_Config(
        url="jira.example.com",
        username="user",
        polling_interval=30,
        project="PROJ",
        status_todo="To Do",
        password="explicit_secret",
    )
    assert config.password.get_secret_value() == "explicit_secret"


def test_jira_config_with_env_password():
    config = Jira_Config(
        url="jira.example.com",
        username="user",
        polling_interval=30,
        project="PROJ",
        status_todo="To Do",
    )
    assert config.password.get_secret_value() == "jira_secret"


def test_jira_config_without_password():
    with pytest.raises(ConfigException):
        Jira_Config(
            url="jira.example.com",
            username="user",
            polling_interval=30,
            project="PROJ",
            status_todo="To Do",
            password=None,
        )


# Additional tests should be written for other models and validators, such as:
# - UploadConfig
# - Password
# - LLMEmbeddingsValidateType
# - SourceData
# - ConfluenceModel
# - GitlabModel
# - Web
# - Files
# - Source
# - Core
# - EmbeddingCore
# - EmbeddingsConfig
# - CohereReRanker
# - HuggingFaceReRanker
# - ReRankerConfig
# - LLMCore
# - LLMConfig
# - Config

# Remember to test edge cases, such as invalid inputs, missing fields, and incorrect types.
