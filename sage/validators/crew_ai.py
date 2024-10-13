## Validator for the CrewAI framework interface
from pathlib import Path
from typing import Any, List, Optional

import yaml
from crewai import Agent, Crew, Task
from crewai.llm import LLM
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from langchain_core.runnables import RunnableConfig
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from sage.agent.memory import (
    CustomRAGStorage,
)
from sage.utils.exceptions import ConfigException
from sage.validators.config_toml import Core


class TaskConfig(Task):
    agent: Agent | str = Field(
        description="Agent responsible for execution the task.", default=None
    )

    @field_validator("description")
    @classmethod
    def task_description_should_include_input(cls, v: str) -> str:
        if "{input}" not in v:
            raise ValueError("Task description must include '{input}' placeholder.")
        return v


class AgentConfig(Agent):
    allow_delegation: bool = Field(
        default=False, description="Allow delegation of tasks to agents"
    )
    runnable_config: Optional[RunnableConfig] = Field(
        default=None,
        description="A runnable configuration to be used by the AgentExecutor",
    )


def configure_long_term_memory(db_storage_path: str, name: str) -> LongTermMemory:
    """Configure the long term memory storage for agents.

    This function initializes and configures an instance of `LongTermMemory`
    using a SQLite storage class. The SQLite database file is created in the specified
    directory, nested under the given name.

    Args:
        db_storage_path (str): The base path where the long term memory storage database should be created.
        name (str): The subdirectory name under the base path to store the long term memory database.

    Returns:
        LongTermMemory: An instance of `LongTermMemory` configured with a SQLite storage backend.
    """
    db_path = Path(db_storage_path) / name / "long_term_memory_storage.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return LongTermMemory(storage=LTMSQLiteStorage(db_path=str(db_path)))


class CrewConfig(Crew):
    """Generates a Crew profile from the loaded configuration"""

    name: str
    agents: List[AgentConfig] = Field(..., min_length=1)
    tasks: List[TaskConfig] = Field(..., min_length=1)
    verbose: int | bool = Field(default=True)
    memory: bool = Field(
        default=False,
        description="Whether the crew should use memory to store memories of it's execution",
    )
    db_storage_path: str = Field(
        description="Path for the memory storage database location"
    )

    @model_validator(mode="after")
    def create_crew_memory(self) -> "Crew":
        """Set private attributes."""
        if self.memory:
            self._long_term_memory = configure_long_term_memory(
                self.db_storage_path, self.name
            )
            crew_data_dir = Path(self.db_storage_path) / self.name
            self._short_term_memory = ShortTermMemory(
                crew=self,
                embedder_config=None,
                storage=CustomRAGStorage(
                    crew_name=self.name,
                    storage_type="short_term",
                    data_dir=crew_data_dir,
                    model=self.embedder["model"],
                    dimension=self.embedder["dimension"],
                ),
            )
            self._entity_memory = EntityMemory(
                crew=self,
                embedder_config=None,
                storage=CustomRAGStorage(
                    crew_name=self.name,
                    storage_type="entities",
                    data_dir=crew_data_dir,
                    model=self.embedder["model"],
                    dimension=self.embedder["dimension"],
                ),
            )
        return self

    @field_validator("agents")
    @classmethod
    def check_unique_agent_roles(cls, agents: List[AgentConfig]):
        roles = [agent.role for agent in agents]
        if len(roles) != len(set(roles)):
            raise ConfigException("Agent roles must be unique")
        return agents

    @model_validator(mode="before")
    @classmethod
    def validate_and_assign_agents_to_task(cls, values) -> dict:
        """
        Validates and assigns agents to their respective tasks within a crew configuration.

        This method performs the following actions:
        1. Checks if all agents provided in the configuration are properly instantiated.
        2. If agents are provided as dictionaries, it instantiates them as `AgentConfig` objects,
           using the manager's LLM if the agent's LLM is not specified.
        3. Creates a dictionary of agents keyed by their roles for quick lookup.
        4. For each task in the configuration, it matches the task's agent with a corresponding
           agent in the configuration:
           a. Raises a `ConfigException` if an agent assigned to a task does not exist.
           b. Updates the task to reference the instantiated agent object.

        Args:
            values (dict): The initial configuration values that include agents and tasks.

        Returns:
            dict: The updated configuration with tasks referencing instantiated agent objects.

        Raises:
            ConfigException: If a task references an agent that does not exist in the configuration.
        """
        agents = values.get("agents", [])

        # Ensure agents are properly instantiated
        if all(isinstance(agent, dict) for agent in agents):
            agents = [
                (
                    AgentConfig(
                        **{
                            **agent,
                            "llm": (
                                agent.get("llm")
                                if agent.get("llm")
                                else values.get("manager_llm")
                            ),
                        }
                    )
                    if isinstance(agent, dict)
                    else agent
                )
                for agent in agents
            ]
            values["agents"] = agents

        agents_dict = {agent.role: agent for agent in agents}

        for task in values.get("tasks", []):
            task_agent = task["agent"]
            task_description = task["description"]
            matched_agent = agents_dict.get(task_agent)
            if not matched_agent:
                raise ConfigException(
                    f"Agent '{task_agent}' assigned to task '{task_description}' does not exist."
                )
            # Update task with the initialized agent
            task["agent"] = matched_agent

        return values


def load_and_validate_agents_yaml(
    config: Core, llm_model: Any, embedding_model: Any, dimension: int
) -> list:
    """Validates and loads all available agents configuration files."""
    agent_dir = config.agents_dir
    if agent_dir is None:
        return []

    dir_path = Path(agent_dir)

    if not dir_path.exists():
        raise ConfigException(f"The agents dir '{agent_dir}' does not exist")

    if not dir_path.is_dir():
        raise ConfigException(f"The agents dir '{agent_dir}' is not a directory")

    # Check if the directory contains any .yaml or .yml files
    yaml_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))
    if not yaml_files:
        raise ConfigException(
            f"The agents dir '{agent_dir}' does not contain any YAML files"
        )

    crew_storage_path = config.data_dir / "crewai"
    crew_llm = LLM(model=llm_model.model_name)
    # Load respective agent files
    crew_list = []
    try:
        for file_path in yaml_files:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                if data is None:
                    raise ConfigException(f"The file '{file_path}' is empty")
                # Validate the data with Pydantic
                crew_model = CrewConfig(
                    **{
                        **data,
                        "manager_llm": (
                            data.get("manager_llm")
                            if data.get("manager_llm")
                            else crew_llm
                        ),
                        "embedder": {"model": embedding_model, "dimension": dimension},
                        "db_storage_path": str(crew_storage_path),
                    }
                )
                crew_list.append(crew_model)
    except yaml.YAMLError as exc:
        raise ConfigException(f"Error parsing YAML: {exc}")
    except Exception as ve:
        raise ConfigException(f"Validation error in agent YAML: {ve}")

    return crew_list
