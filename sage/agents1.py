from langchain.agents import AgentExecutor, XMLAgent, tool
from constants import LLM_MODEL as model
from langchain.tools import DuckDuckGoSearchRun, Tool


from langchain.schema.document import Document
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence, RunnableConfig, RunnableMap, RunnableLambda
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, AIMessagePromptTemplate


agent_instructions = """
You are a helpful assistant named Sage. Your goal is to assist the user in answering any questions or performing any actions they may have.

You have access to the following tools:

{tools}

To utilize a tool, you should use the following tags:

- To indicate the tool you are using: <tool>'tool name'</tool>
- To input commands or queries to the tool: <tool_input>'your query'</tool_input>
- To represent the tool's output or response: <observation>'tool output'</observation>

For example, if you have a tool called 'search' that can perform a Google search, and you need to find the weather in lagos, Nigeria; you would respond with:

<tool>search</tool><tool_input>weather in Lagos</tool_input>
<observation>It is currently 32 degrees.</observation>

Once you have gathered all necessary information and observations, provide the final answer using the <final_answer> tags. For instance:

<final_answer>The weather in Lagos, Nigeria is currently 32 degrees.</final_answer>

Ensure you sequentially process the tools as needed to fully address the question.
If a tool does not provide a satisfactory answer or additional clarification is needed, you may choose to use another tool or ask for further details.

Begin by addressing the user's question provided below:

Question: {question}"""

def agent_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        agent_instructions
    ) + AIMessagePromptTemplate.from_template("{intermediate_steps}")


duck_search = DuckDuckGoSearchRun()

toolsss = Tool(
    name="internet_search",
    description="Search the Internet using the duck duck search engine and return the first result. The input should a typical search query",
    func=duck_search.run,
)

# @tool
# def search(query: str) -> str:
#     """Search things about current events."""
#     return duck_search.run(query)


tool_list = [toolsss]

# Get prompt to use
prompt = XMLAgent.get_default_prompt()

def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

agent = (
    {
        "question": lambda x: x["question"],
        "intermediate_steps": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | prompt.partial(tools=convert_tools(tool_list))
    | model.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgent.get_default_output_parser()
)

agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)
for chunk in agent_executor.stream({"question": "whats the weather in New york?"}):
    print(chunk)

#print(agent_executor.invoke({"question": "whats the weather in New york?"}))



##
## Generate 3 extra random members to the current Anthea members list
##