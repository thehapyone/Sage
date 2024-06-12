## CrewAIRunnable
from langchain.schema.runnable import (
    RunnableConfig,
    RunnableLambda,
)
from asyncer import asyncify
from chainlit import AsyncLangchainCallbackHandler

from crewai import Crew
import chainlit as cl


from sage.crewai_utils.gamer.tasks import GameTasks
from sage.crewai_utils.gamer.agents import GameAgents


tasks = GameTasks()
agents = GameAgents()


class CrewAIRunnable:
    

    def __init__(self, callback):
        self._callback = callback
        pass

    def _create_crew(self, callback) -> Crew:
        chainlit_callback_handler = callback.handlers[0]
        # Create Agents
        senior_engineer_agent = agents.senior_engineer_agent(chainlit_callback_handler)
        qa_engineer_agent = agents.qa_engineer_agent(chainlit_callback_handler)
        chief_qa_engineer_agent = agents.chief_qa_engineer_agent(
            chainlit_callback_handler
        )

        # Create Tasks
        code_game = tasks.code_task(senior_engineer_agent)
        review_game = tasks.review_task(qa_engineer_agent)
        approve_game = tasks.evaluate_task(chief_qa_engineer_agent)

        crew = Crew(
            agents=[senior_engineer_agent, chief_qa_engineer_agent],
            tasks=[code_game, approve_game],
            verbose=True,
            memory=True,
            # process=Process.hierarchical,
            # manager_llm=LLM_MODEL,
            share_crew=False,
            embedder={
                "provider": "azure_openai",
                "config": {
                    "model": "text-embedding-ada-002",
                    "deployment_name": "ada-embeddings",
                },
            },
        )
        return crew

    ## I need ainvoke, invoke,
    def _format_runnable_response(self, result: str) -> dict:
        return {"answer": result}

    def _format_crew_input(self, request: dict) -> dict:
        return {"game": request["question"]}

    def _crew(self, x: dict, config: RunnableConfig) -> dict:
        crew = self._create_crew(config.get("callbacks"))
        result = crew.kickoff(self._format_crew_input(x))
        return self._format_runnable_response(result)

    async def _acrew(self, x: dict, config: RunnableConfig) -> dict:
        crew = self._create_crew(config.get("callbacks"))
        result = await asyncify(crew.kickoff)(self._format_crew_input(x))
        return self._format_runnable_response(result)

    def mycrew(self) -> RunnableLambda:
        return RunnableLambda(self._crew, afunc=self._acrew).with_config(
            run_name="GameCrew"
        )
