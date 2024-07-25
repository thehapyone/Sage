## Validator for the CrewAI framework interface
from pathlib import Path
from typing import Any, List, Optional

import yaml
from crewai import Agent, Crew, Task
from langchain_core.runnables import RunnableConfig
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
    runnable_config: Optional[RunnableConfig] = Field(
        default=None,
        description="A runnable configuration to be used by the AgentExecutor",
    )

    # Override the function until crewai supports this
    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        from crewai.memory.contextual.contextual_memory import ContextualMemory
        from langchain.tools.render import render_text_description

        if self.tools_handler:
            self.tools_handler.last_used_tool = {}  # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")

        task_prompt = task.prompt()

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            contextual_memory = ContextualMemory(
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        tools = tools or self.tools

        parsed_tools = self._parse_tools(tools or [])  # type: ignore # Argument 1 to "_parse_tools" of "Agent" has incompatible type "list[Any] | None"; expected "list[Any]"
        self.create_agent_executor(tools=tools)
        self.agent_executor.tools = parsed_tools
        self.agent_executor.task = task

        self.agent_executor.tools_description = render_text_description(parsed_tools)
        ## Overridden due to name mangling as __tools_names is a private method
        self.agent_executor.tools_names = self._Agent__tools_names(parsed_tools)

        if self.crew and self.crew._train:
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

        ## Overridden
        result = self.agent_executor.invoke(
            {
                "input": task_prompt,
                "tool_names": self.agent_executor.tools_names,
                "tools": self.agent_executor.tools_description,
            },
            config=self.runnable_config,
        )["output"]

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        # If there was any tool in self.tools_results that had result_as_answer
        # set to True, return the results of the last tool that had
        # result_as_answer set to True
        for tool_result in self.tools_results:  # type: ignore # Item "None" of "list[Any] | None" has no attribute "__iter__" (not iterable)
            if tool_result.get("result_as_answer", False):
                result = tool_result["result"]

        return result


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
