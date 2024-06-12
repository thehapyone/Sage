from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
import yaml

from crewai import Agent, Task

from sage.utils.exceptions import ConfigException


class TaskConfig(Task):
    dependencies: List[str] = Field(default_factory=list)
    priority: Optional[str] = "Medium"
    callbacks: List[str] = Field(default_factory=lambda: ["default_callback"])


class AgentConfig(Agent):
    tools: List[str] = Field(default_factory=lambda: ["DefaultTool"])
    capabilities: Optional[List[str]] = []


class CrewConfig(BaseModel):
    name: str
    agents: List[AgentConfig] = Field(..., min_length=1)
    tasks: List[TaskConfig] = Field(..., min_length=1)

    @field_validator("agents")
    @classmethod
    def check_unique_agent_names(cls, agents: List[AgentConfig]):
        names = [agent.name for agent in agents]
        if len(names) != len(set(names)):
            raise ConfigException("Agent names must be unique")
        return agents

    @model_validator(mode="after")
    def check_valid_agent_assignment(self) -> "CrewConfig":
        agent_roles = [agent.role for agent in self.agents]
        for task in self.tasks:
            if task.agent not in agent_roles:
                raise ConfigException(
                    f"Agent '{task.agent}' assigned to task '{task.description}' does not exist"
                )
        return self


# Example usage
yaml_config = """
name: GameStartup
agents:
  - name: Alex
    role: Game Designer
    goal: Design engaging and innovative game mechanics
    backstory: Alex has a decade of experience in game design and is known for creating unique and popular game mechanics.

  - name: Jamie
    role: Marketing Strategist
    goal: Develop a marketing strategy to launch the game successfully
    backstory: Jamie has worked on several successful game launches and excels at creating buzz and engaging the gaming community.

tasks:
  - description: Research and design the core mechanics of the new game
    agent: Alex
    expected_output: A detailed report on the game mechanics including sketches and flowcharts

  - description: Create a comprehensive marketing plan for the game launch
    agent: Jamie
    expected_output: A complete marketing strategy document with timelines, channels, and key messages
"""

# Load the YAML configuration
config_dict = yaml.safe_load(yaml_config)

# Validate the configuration and set defaults
try:
    crew_config = CrewConfig(**config_dict)
    print("Configuration is valid")
    print(crew_config.json(indent=2))
except Exception as e:
    print(f"Configuration validation error: {e}")
