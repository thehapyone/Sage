from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from sage.utils.exceptions import ConfigException
from sage.validators.crew_ai import (
    AgentConfig,
    TaskConfig,
    load_and_validate_agents_yaml,
)


# Fixtures for testing
@pytest.fixture
def valid_agent_yaml():
    return """
name: GameStartup
process: hierarchical
agents:
  - role: Game Designer
    goal: Design engaging and innovative game mechanics
    backstory: An expert with over a decade of experience in game design and is known for creating unique and popular game mechanics.
  
  - role: Marketing Strategist
    goal: Develop a marketing strategy to launch the game successfully
    backstory: You have worked on several successful game launches and excels at creating buzz and engaging the gaming community.
tasks:
  - description: "You help research and design the core mechanics of games. This is the game instructions: {input}"
    agent: Game Designer
    expected_output: A detailed report on the game mechanics including sketches and flowcharts
  - description: "Conduct a competitor analysis for similar games. Game details: {input}"
    agent: Marketing Strategist
    expected_output: A report on competitor strengths, weaknesses, and market positioning
    """


@pytest.fixture
def agent_yaml_with_validation_errors():
    return """
name: FaultyStartup
process: hierarchical
agents:
  - role: Game Designer
    goal: Design engaging and innovative game mechanics
    backstory: An expert with over a decade of experience in game design and is known for creating unique and popular game mechanics.
tasks:
  - description: "Invalid task without input placeholder."
    agent: Game Designer
    expected_output: A detailed report on the game mechanics including sketches and flowcharts
    """


mock_llm = Mock(name="ChatLiteLLM")


@pytest.fixture
def mock_path_is_dir(monkeypatch):
    monkeypatch.setattr(
        "pathlib.Path.is_dir", Mock(name="pathlib.Path.is_dir", return_value=True)
    )


@pytest.fixture
def mock_path_exists(monkeypatch):
    monkeypatch.setattr(
        "pathlib.Path.exists", Mock(name="pathlib.Path.exists", return_value=True)
    )


@pytest.fixture
def mock_path_glob(monkeypatch):
    mock_glob = MagicMock()
    mock_glob.side_effect = [
        [Path("valid.yaml")],  # first call returns .yaml file
        [],  # second call returns no .yml files
    ]
    monkeypatch.setattr(Path, "glob", mock_glob)


@pytest.fixture
def mock_path_is_dir_and_exists(monkeypatch, mock_path_is_dir, mock_path_exists):
    return


@pytest.fixture
def mock_standard_agent():
    return AgentConfig(
        role="SampleAgent", goal="MockAgent", backstory="SomeAgent", llm=mock_llm
    )


# Tests for TaskConfig
def test_task_config_valid(mock_standard_agent):
    task = TaskConfig(
        description="Task description: {input}",
        agent=mock_standard_agent,
        expected_output="Sample text",
    )
    assert task.description == "Task description: {input}"


def test_task_config_description_missing_input(mock_standard_agent):
    with pytest.raises(
        ValueError, match="Task description must include '{input}' placeholder."
    ):
        TaskConfig(
            description="Task description",
            agent=mock_standard_agent,
            expected_output="Sample text",
        )


# Tests for AgentConfig
def test_agent_config_valid():
    data = {
        "role": "Game Designer",
        "goal": "Design engaging and innovative game mechanics",
        "backstory": "Expert designer",
    }
    agent = AgentConfig(
        **data,
        llm=mock_llm,
    )
    assert agent.role == "Game Designer"
    assert agent.goal == "Design engaging and innovative game mechanics"
    assert agent.backstory == "Expert designer"


# Tests for CrewConfig
def test_load_and_validate_agents_wrong_dir():
    with pytest.raises(
        ConfigException, match="The agents dir 'dummy_path' does not exist"
    ):
        load_and_validate_agents_yaml("dummy_path", mock_llm)


def test_load_and_validate_agents_none_dir():
    crew = load_and_validate_agents_yaml(None, mock_llm)
    assert crew == []


def test_load_and_validate_agents_yaml_not_directory(mock_path_exists):
    with patch("pathlib.Path.is_dir", return_value=False):
        with pytest.raises(
            ConfigException, match="The agents dir 'not_a_dir' is not a directory"
        ):
            load_and_validate_agents_yaml("not_a_dir", mock_llm)


def test_load_and_validate_agents_yaml_no_yaml_files(mock_path_is_dir_and_exists):
    with patch("pathlib.Path.glob", return_value=[]):
        with pytest.raises(ConfigException) as excinfo:
            load_and_validate_agents_yaml("empty_dir", mock_llm)
        assert "The agents dir 'empty_dir' does not contain any YAML files" in str(
            excinfo.value
        )


def test_load_and_validate_agents_yaml_parsing_error(mock_path_is_dir_and_exists):
    with patch("pathlib.Path.glob", return_value=[Path("invalid.yaml")]), patch(
        "sage.validators.crew_ai.open", mock_open(read_data="invalid: yaml: content")
    ):
        with pytest.raises(ConfigException) as excinfo:
            load_and_validate_agents_yaml("some_dir", mock_llm)
        assert "Error parsing YAML:" in str(excinfo.value)


def test_load_and_validate_agents_yaml_validation_error(
    mock_path_is_dir_and_exists, mock_path_glob, agent_yaml_with_validation_errors
):
    with patch(
        "sage.validators.crew_ai.open",
        mock_open(read_data=agent_yaml_with_validation_errors),
    ):
        with pytest.raises(ConfigException) as excinfo:
            load_and_validate_agents_yaml("valid_but_invalid_crew.yaml", mock_llm)
        assert "Validation error in agent YAML:" in str(excinfo.value)
        assert "Task description must include '{input}' placeholder." in str(
            excinfo.value
        )


def test_load_and_validate_agents_yaml_wrong_agnent_matching(
    mock_path_is_dir_and_exists, mock_path_glob
):
    wrong_agent_matching = """
    name: GameStartup
    process: hierarchical
    agents:
      - role: Game Designer
        goal: Design engaging and innovative game mechanics
        backstory: An expert with over a decade of experience in game design and is known for creating unique and popular game mechanics.
      
      - role: Marketing Strategist
        goal: Develop a marketing strategy to launch the game successfully
        backstory: You have worked on several successful game launches and excels at creating buzz and engaging the gaming community.
    tasks:
      - description: "Build game - {input}"
        agent: Game Engineer
        expected_output: A detailed report on the game mechanics including sketches and flowcharts
      - description: "Conduct a competitor analysis for similar games. Game details: {input}"
        agent: Marketing Strategist
        expected_output: A report on competitor strengths, weaknesses, and market positioning
    """
    with patch(
        "sage.validators.crew_ai.open",
        mock_open(read_data=wrong_agent_matching),
    ):
        with pytest.raises(ConfigException) as excinfo:
            load_and_validate_agents_yaml("valid_but_invalid_crew.yaml", mock_llm)
        assert "Validation error in agent YAML:" in str(excinfo.value)
        assert (
            "Agent 'Game Engineer' assigned to task 'Build game - {input}' does not exist"
            in str(excinfo.value)
        )


def test_load_and_validate_agents_yaml_success(
    mock_path_is_dir_and_exists, mock_path_glob, valid_agent_yaml
):
    with patch("sage.validators.crew_ai.open", mock_open(read_data=valid_agent_yaml)):
        crew_models = load_and_validate_agents_yaml("some_dir", mock_llm)
        assert len(crew_models) == 1
        crew = crew_models[0]
        assert crew.name == "GameStartup"
        assert len(crew.agents) == 2
        assert crew.agents[0].role == "Game Designer"
        assert crew.agents[1].role == "Marketing Strategist"
        assert len(crew.tasks) == 2
        assert (
            crew.tasks[0].description
            == "You help research and design the core mechanics of games. This is the game instructions: {input}"
        )
        assert (
            crew.tasks[1].description
            == "Conduct a competitor analysis for similar games. Game details: {input}"
        )
