# test_constants.py

from importlib import reload
from logging import getLevelName
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from anyio import Path

from sage.utils.exceptions import ConfigException

# Sample configuration data as a dictionary
sample_config_data = {
    "core": {
        "data_dir": "/fake/path",
        "logging_level": "WARNING",
        "user_agent": "codesage.ai",
    },
    "upload": {
        "max_size_mb": 100,
    },
    "jira": {
        "url": "https://example.atlassian.net",
        "username": "jira_username",
        "password": "jira_password",
        "polling_interval": 100,
        "project": "PROJECT1",
        "status_todo": "To Do",
    },
    "openai": {"password": "openai_token", "oranization": "org1"},
    "source": {"top_k": 10, "files": {"paths": ["file1.pdf", "file2.pdf"]}},
    "embedding": {
        "type": "openai",
        "openai": {"name": "ada_embeddings"},
    },
    "llm": {"type": "openai", "openai": {"name": "gpt3.5"}},
}

mock_path_mkdir = AsyncMock(name="path.mkdir", return_value=True)
mock_openai_llm = Mock(name="ChatOpenAI")
mock_openai_embedding = Mock(name="OpenAIEmbeddings")
mock_openai_embedding.return_value.embed_query.return_value = [0] * 768
mock_logger_spec = Mock(name="logger")


@pytest.fixture
def mock_path(monkeypatch):
    # Mock the Path class to avoid actual filesystem operations
    monkeypatch.setattr("anyio.Path.mkdir", mock_path_mkdir)


@pytest.fixture
def mock_openai_models(monkeypatch):
    monkeypatch.setattr("langchain_openai.chat_models.ChatOpenAI", mock_openai_llm)
    monkeypatch.setattr(
        "langchain_openai.embeddings.OpenAIEmbeddings", mock_openai_embedding
    )


@pytest.fixture
def mock_common(monkeypatch):
    # Set the SAGE_CONFIG_PATH environment variable to point to the test config file
    monkeypatch.setenv("SAGE_CONFIG_PATH", "fake_config.toml")
    # Mock the JinaEmbeddings to avoid setting it up
    monkeypatch.setattr("sage.utils.supports.JinaAIEmbeddings", MagicMock())
    # Mock the sys.exit call to raise an exception for testing
    monkeypatch.setattr("sys.exit", Mock(name="SystemExit", side_effect=SystemExit))
    # Mock the toml.load function to return a sample configuration
    monkeypatch.setattr("toml.load", lambda path: sample_config_data)


@pytest.fixture
def mock_logger(monkeypatch):
    # Mock the CustomLogger to avoid actual logging during tests
    monkeypatch.setattr("sage.utils.logger.CustomLogger", mock_logger_spec)


def test_successful_config_load(
    mock_logger, mock_path, mock_openai_models, mock_common
):
    # Import the constants module to apply the mocks
    # reload(constants)
    import sage.constants as constants

    reload(constants)

    core_config = constants.core_config

    assert core_config.data_dir == Path("/fake/path")
    assert core_config.logging_level == getLevelName("WARNING")
    assert isinstance(constants.assets_dir, Path)

    assert mock_path_mkdir.assert_called_once
    # Test if the chat and embedding models got initialized
    assert mock_openai_llm.assert_called_once
    assert mock_openai_embedding.assert_called_once


@pytest.mark.parametrize(
    "exception, mock_patch_target",
    [
        (FileNotFoundError, "toml.load"),
        (ConfigException("validator error"), "sage.utils.validator.Config"),
        (KeyError("key is not found"), "sage.utils.validator.Config"),
    ],
)
def test_config_load_exceptions(
    exception, mock_patch_target, mock_logger, mock_path, mock_common
):
    with patch(mock_patch_target, side_effect=exception), pytest.raises(SystemExit):
        import sage.constants

        reload(sage.constants)
    mock_logger_spec.error.assert_called_once
