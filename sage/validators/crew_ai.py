## Validator for the CrewAI framework interface
from typing import Any, List, Optional, Tuple

import yaml
from chainlit import AsyncLangchainCallbackHandler
from crewai import Agent, Task, Crew
from crewai.process import Process
from pydantic import UUID4, BaseModel, Field, field_validator, model_validator

from sage.constants import LLM_MODEL
from sage.utils.exceptions import ConfigException


class TaskConfig(Task):
    agent: Agent | str = Field(
        description="Agent responsible for execution the task.", default=None
    )
    async_execution: Optional[bool] = Field(
        description="Whether the task should be executed asynchronously or not.",
        default=False,
    )

    @field_validator("description")
    @classmethod
    def task_description_should_include_input(cls, v: str) -> str:
        if "{input}" not in v:
            raise ValueError("Task description must include '{input}' placeholder.")
        return v


class AgentConfig(Agent):
    llm: Any = Field(
        default_factory=lambda: LLM_MODEL,
        description="Language model that will run the agent.",
    )
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

    @field_validator("agents")
    @classmethod
    def check_unique_agent_roles(cls, agents: List[AgentConfig]):
        roles = [agent.role for agent in agents]
        if len(roles) != len(set(roles)):
            raise ConfigException("Agent roles must be unique")
        return agents

    @model_validator(mode="before")
    @classmethod
    def set_manager_llm(cls, values) -> dict:
        values["manager_llm"] = LLM_MODEL
        return values

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



# Load the YAML configuration
config_dict = yaml.safe_load(yaml_config)

# Validate the configuration and set defaults
try:
    crew_model = CrewConfig(**config_dict)
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration validation error: {e}")
    raise e

game = "A tic tac toe game that can be playable by two players. The game should provide an instruction screen and playable in a simple UI. Also the game should support restarting the game "

#result = crew_model.kickoff({"input": game})
