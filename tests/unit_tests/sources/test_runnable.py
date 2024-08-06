from unittest.mock import MagicMock, patch

import pytest

from tests.unit_tests.extras import MockSession

mock_session = MockSession()

new_mock = MagicMock()

# Mock `sage.constants` entirely
with patch.dict(
    "sys.modules",
    {
        "sage.constants": MagicMock(),
    },
):
    from sage.sources.runnable import RunnableBase

# Define other mock instances
mock_retriever = MagicMock()
mock_runnable = MagicMock()
mock_chat_history_loader = MagicMock()


@pytest.fixture
def runnable_base_tool():
    return RunnableBase(mode="tool", user_session=mock_session)


@pytest.fixture
def runnable_base_chat():
    return RunnableBase(mode="chat", user_session=mock_session)


@pytest.fixture
def crew_mock(monkeypatch):
    crew_mock = MagicMock()
    crew_mock.return_value.runnable.return_value = {"crew1": MagicMock("crew1")}
    monkeypatch.setattr("sage.agent.crew.CrewAIRunnable", crew_mock)
    return crew_mock


def test_runnablebase_initialization_tool(runnable_base_tool):
    assert runnable_base_tool.mode == "tool"
    assert runnable_base_tool._runnable is None


def test_runnablebase_initialization_chat(runnable_base_chat):
    assert runnable_base_chat.mode == "chat"
    assert runnable_base_chat._runnable is None


def test_runnablebase_initialization_invalid_mode():
    with pytest.raises(ValueError) as excinfo:
        RunnableBase(mode="invalid_mode")
    assert "invalid_mode is not supported. Supported modes are: chat and tool" in str(
        excinfo.value
    )


def test_create_crew_runnable(runnable_base_tool):
    with patch("sage.agent.crew.CrewAIRunnable") as mock_crew:
        mock_crew.return_value.runnable.return_value = {"foo": "bar"}
        crew_runnable = runnable_base_tool.create_crew_runnable()
        assert crew_runnable == {"foo": "bar"}


def test_setup_runnable_with_custom_runnable_tool_mode(runnable_base_tool):
    mock_runnable = MagicMock()
    runnable_base_tool.setup_runnable(runnable=mock_runnable)
    assert runnable_base_tool._runnable == mock_runnable


def test_setup_runnable_with_custom_runnable_chat_mode(runnable_base_chat):
    mock_runnable = MagicMock()
    runnable_base_chat.setup_runnable(runnable=mock_runnable)
    assert runnable_base_chat._runnable == None
    assert mock_session.get("runnable") == mock_runnable
