from typing import Any, List, Optional, Tuple

import yaml
from chainlit import AsyncLangchainCallbackHandler
from crewai import Agent, Task, Crew
from crewai.process import Process
from pydantic import BaseModel, Field, field_validator, model_validator

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


def create_crew(config: CrewConfig) -> Tuple[str, Crew]:
    """Generates a Crew profile from the loaded configuration"""

    crew = Crew(
        agents=config.agents,
        tasks=config.tasks,
        verbose=True,
        memory=False,
        process=Process.hierarchical,
        manager_llm=LLM_MODEL,
        share_crew=False,
    )

    return config.name, crew


# Example usage
yaml_config = """
name: GameStartup
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

  - description: "You develop the initial concept art and prototypes for the game. Game details: {input}"
    agent: Game Designer
    expected_output: Concept art and prototype sketches

  - description: "You wil create a comprehensive marketing plan for the game launch. Game details: {input}"
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
    _, crew_model = create_crew(crew_config)
    print(crew_model)
except Exception as e:
    print(f"Configuration validation error: {e}")
    raise e

game = "A tic tac toe game that can be playable by two players. The game should provide an instruction screen and playable in a simple UI. Also the game should support restarting the game "

result = crew_model.kickoff({"input": game})

# Print results
print("\n\n########################")
print("## Here is the result")
print("########################\n")
print("final code for the game:")
print(result)
