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

from sage.agent.memory import EnhanceLongTermMemory, LTMSQLiteStorage
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


def configure_long_term_memory(
    db_storage_path: str, name: str
) -> EnhanceLongTermMemory:
    """Configure the long term memory storage for agents.

    This function initializes and configures an instance of `EnhanceLongTermMemory`
    using a SQLite storage class. The SQLite database file is created in the specified
    directory, nested under the given name.

    Args:
        db_storage_path (str): The base path where the long term memory storage database should be created.
        name (str): The subdirectory name under the base path to store the long term memory database.

    Returns:
        EnhanceLongTermMemory: An instance of `EnhanceLongTermMemory` configured with a SQLite storage backend.
    """
    db_path = Path(db_storage_path) / name / "long_term_memory_storage.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return EnhanceLongTermMemory(storage=LTMSQLiteStorage(db_path=str(db_path)))


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
            # self._short_term_memory = ShortTermMemory(
            #     crew=self, embedder_config=self.embedder
            # )
            # self._entity_memory = EntityMemory(crew=self, embedder_config=self.embedder)
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


def load_and_validate_agents_yaml(config: Core, llm_model: Any) -> list:
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
                    **data,
                    manager_llm=llm_model,
                    db_storage_path=str(crew_storage_path),
                )
                crew_list.append(crew_model)
    except yaml.YAMLError as exc:
        raise ConfigException(f"Error parsing YAML: {exc}")
    except Exception as ve:
        raise ConfigException(f"Validation error in agent YAML: {ve}")

    return crew_list
