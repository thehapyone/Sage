# tests/conftest.py

import pytest
import os


@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    # Set up environment variables for all tests
    monkeypatch.setenv("AZURE_PASSWORD", "azure_secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_secret")
    monkeypatch.setenv("JIRA_PASSWORD", "jira_secret")

    yield
