## Validator for the CrewAI framework interface
from pathlib import Path
from typing import Any, List

import yaml
from chainlit import AsyncLangchainCallbackHandler
from crewai import Agent, Crew, Task
from crewai.process import Process
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from sage.utils.exceptions import ConfigException


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
    llm: Any
    allow_delegation: bool = Field(
        default=False, description="Allow delegation of tasks to agents"
    )
    # callbacks = ([AsyncLangchainCallbackHandler().handlers[0]],)


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
    manager_llm: Any

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
        agents = values.get("agents", [])

        # Ensure agents are properly instantiated
        if all(isinstance(agent, dict) for agent in agents):
            agents = [
                (
                    AgentConfig(**agent, llm=values.get("manager_llm"))
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


def load_and_validate_agents_yaml(agent_dir: str | None, llm_model: Any) -> list:
    """Validates and loads all available agents configuration files."""
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

    # Load respective agent files
    crew_list = []
    try:
        for file_path in yaml_files:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                if data is None:
                    raise ConfigException(f"The file '{file_path}' is empty")
                # Validate the data with Pydantic
                crew_model = CrewConfig(**data, manager_llm=llm_model)
                crew_list.append(crew_model)
    except yaml.YAMLError as exc:
        raise ConfigException(f"Error parsing YAML: {exc}")
    except Exception as ve:
        raise ConfigException(f"Validation error in agent YAML: {ve}")

    return crew_list
