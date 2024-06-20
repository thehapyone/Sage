from typing import Any, List, Optional

import yaml
from chainlit import AsyncLangchainCallbackHandler
from crewai import Agent, Task
from pydantic import BaseModel, Field, field_validator, model_validator

from sage.constants import LLM_MODEL
from sage.utils.exceptions import ConfigException


class TaskConfig(Task):
    agent: Agent | str = Field(
        description="Agent responsible for execution the task.", default=None
    )
    async_execution: Optional[bool] = Field(
        description="Whether the task should be executed asynchronously or not.",
        default=True,
    )


class AgentConfig(Agent):
    llm: Any = Field(
        default_factory=lambda: LLM_MODEL,
        description="Language model that will run the agent.",
    )
    allow_delegation: bool = Field(
        default=False, description="Allow delegation of tasks to agents"
    )
    # callbacks = ([AsyncLangchainCallbackHandler().handlers[0]],)


class CrewConfig(BaseModel):
    name: str
    agents: List[AgentConfig] = Field(..., min_length=1)
    tasks: List[TaskConfig] = Field(..., min_length=1)

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
                AgentConfig(**agent) if isinstance(agent, dict) else agent
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


# Example usage
yaml_config = """
name: GameStartup
agents:
  - role: Game Designer
    goal: Design engaging and innovative game mechanics
    backstory: Alex has a decade of experience in game design and is known for creating unique and popular game mechanics.

  - role: Marketing Strategist
    goal: Develop a marketing strategy to launch the game successfully
    backstory: Jamie has worked on several successful game launches and excels at creating buzz and engaging the gaming community.

tasks:
  - description: Research and design the core mechanics of the new game
    agent: Game Designer
    expected_output: A detailed report on the game mechanics including sketches and flowcharts

  - description: Create a comprehensive marketing plan for the game launch
    agent: Marketing Strategist
    expected_output: A complete marketing strategy document with timelines, channels, and key messages
"""

# Load the YAML configuration
config_dict = yaml.safe_load(yaml_config)

# Validate the configuration and set defaults
try:
    crew_config = CrewConfig(**config_dict)
    print("Configuration is valid")
    print(crew_config)
except Exception as e:
    print(f"Configuration validation error: {e}")
    raise e
