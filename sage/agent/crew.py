## CrewAIRunnable
from typing import Sequence

import chainlit as cl
from crewai.crews.crew_output import CrewOutput
from langchain.schema.runnable import (
    RunnableConfig,
    RunnableLambda,
)

from sage.validators.crew_ai import CrewConfig


def format_crew_results(output: CrewOutput) -> str:
    task = output.tasks_output[-1]
    header = "# 📊 Result Summary\n\n---\n"

    agent_heading = f"## 👨‍💼 Agent: **{task.agent}**\n"
    task_description = (
        f"**📝 Summary:** {task.summary if task.summary else 'No summary provided'}\n"
    )

    # Use consistent fenced code blocks for task.raw
    agent_output = f"### 🔍 Final Answer:\n{task.raw}\n"

    _raw = f"{agent_heading}{task_description}{agent_output}"

    final_output = f"{header}{_raw}"

    return final_output


class CrewAIRunnable:
    def __init__(self, crews: Sequence[CrewConfig]):
        self.available_crews = crews

    @staticmethod
    def _get_run_name(config: RunnableConfig) -> str | None:
        return config.get("metadata", {}).get("run_name") or config.get("run_name")

    def get_crew(self, config: RunnableConfig) -> CrewConfig:
        """Retrieve a crew by name or return the single available crew if only one exists."""
        crew_name = self._get_run_name(config)

        # If no crew name is provided, check if we have exactly one available crew
        if not crew_name:
            if len(self.available_crews) == 1:
                return self.available_crews[0]
            else:
                raise ValueError(
                    "Crew name is not provided and there is not exactly one available crew."
                )

        # Search for the crew by the given name
        for crew in self.available_crews:
            if crew.name == crew_name:
                return crew

        raise ValueError(f"Crew with name '{crew_name}' not found")

    def update_agents(self, crew: CrewConfig, config: RunnableConfig):
        """Update the agents attributes"""
        for agent in crew.agents:
            new_config = config.copy()
            extra_callbacks = new_config.get("callbacks").handlers
            new_config["run_name"] = agent.role
            if agent.callbacks:
                agent.callbacks.extend(extra_callbacks)
            else:
                agent.callbacks = extra_callbacks
            agent.runnable_config = new_config

    @staticmethod
    def _format_runnable_response(
        result: CrewOutput, profile: str = "Agent Mode"
    ) -> dict:
        return {
            "answer": (
                format_crew_results(result) if profile == "Agent Mode" else result.raw
            )
        }

    @staticmethod
    def _format_crew_input(request: dict) -> dict:
        return {"input": request["question"]}

    def _crew(self, x: dict, config: RunnableConfig) -> dict:
        """Synchronous crew execution"""
        crew = self.get_crew(config)
        # self.update_agents(crew, config)
        result = crew.kickoff(self._format_crew_input(x))
        return self._format_runnable_response(result)

    async def _acrew(self, x: dict, config: RunnableConfig) -> dict:
        """Asynchronous crew execution"""
        crew = self.get_crew(config)

        chat_profile: str = config.get("metadata", {}).get("chat_profile", "")
        if chat_profile == "Agent Mode":
            crew_inputs = self._format_crew_input(x)
        else:
            crew_inputs = x

        # self.update_agents(crew, config)
        async with cl.Step(name=crew.name, type="tool") as step:
            step.input = x
            result = await crew.kickoff_async(crew_inputs)

        return self._format_runnable_response(result, chat_profile)

    def runnable(self) -> dict[str, RunnableLambda]:
        """Create runnable instance for all available crews"""
        all_instances = {
            crew.name: RunnableLambda(self._crew, afunc=self._acrew).with_config(
                run_name=crew.name
            )
            for crew in self.available_crews
        }
        return all_instances
