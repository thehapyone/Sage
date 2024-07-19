## CrewAIRunnable
from typing import Sequence

import chainlit as cl
from asyncer import asyncify
from chainlit import AsyncLangchainCallbackHandler
from crewai import Crew
from langchain.schema.runnable import (
    RunnableConfig,
    RunnableLambda,
)

from sage.validators.crew_ai import CrewConfig


class CrewAIRunnable:
    def __init__(self, crews: Sequence[CrewConfig]):
        self.avaiable_crews = crews

    @staticmethod
    def _get_run_name(config: RunnableConfig) -> str | None:
        return config.get("metadata", {}).get("run_name") or config.get("run_name")

    def update_agents(self, crew: CrewConfig, config: RunnableConfig):
        """Update the agents attributes"""
        extra_callbacks = config.get("callbacks").handlers
        config["run_name"] = self._get_run_name(config)
        for agent in crew.agents:
            if agent.callbacks:
                agent.callbacks.extend(extra_callbacks)
            else:
                agent.callbacks = extra_callbacks
            agent.runnable_config = config

    def get_crew(self, config: RunnableConfig) -> CrewConfig:
        """Retrieve a crew by name"""
        crew_name = self._get_run_name(config)
        if not crew_name:
            raise ValueError(f"Crew name {crew_name} is not valid")
        for crew in self.avaiable_crews:
            if crew.name == crew_name:
                return crew
        raise ValueError(f"Crew with name {crew_name} not found")

    @staticmethod
    def _format_runnable_response(result: str) -> dict:
        return {"answer": result}

    @staticmethod
    def _format_crew_input(request: dict) -> dict:
        return {"input": request["question"]}

    def _crew(self, x: dict, config: RunnableConfig) -> dict:
        """Synchronous crew execution"""
        crew = self.get_crew(config)
        self.update_agents(crew, config)
        result = crew.kickoff(self._format_crew_input(x))
        return self._format_runnable_response(result)

    async def _acrew(self, x: dict, config: RunnableConfig) -> dict:
        """Asynchronous crew execution"""
        crew = self.get_crew(config)
        self.update_agents(crew, config)
        async with cl.Step(name=crew.name, type="tool") as step:
            step.input = x
            result = await asyncify(crew.kickoff)(self._format_crew_input(x))
        return self._format_runnable_response(result)

    def runnable(self) -> Sequence[RunnableLambda]:
        """Create runnable instance for all available crews"""
        return [
            RunnableLambda(self._crew, afunc=self._acrew).with_config(
                run_name=crew.name
            )
            for crew in self.avaiable_crews
        ]
