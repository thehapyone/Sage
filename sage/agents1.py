from langchain.agents import AgentExecutor, XMLAgent, tool
from constants import LLM_MODEL as model
from langchain.tools import DuckDuckGoSearchRun, Tool
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from typing import Union

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish


from langchain.schema.document import Document
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.schema.runnable import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableConfig,
    RunnableMap,
    RunnableLambda,
)
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, AIMessagePromptTemplate
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains.llm_math.base import LLMMathChain


class XMLAgentOutputParserNew(AgentOutputParser):
    """Parses tool invocations and final answers in XML format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    <tool>search</tool>
    <tool_input>what is 2 + 2</tool_input>
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    <final_answer>Foo</final_answer>
    ```
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print("--------")
        print(text)
        print("##############")

        if "</tool>" in text:
            tool, tool_input = text.split("</tool>")
            _tool = tool.split("<tool>")[1]
            _tool_input = tool_input.split("<tool_input>")[1]
            if "</tool_input>" in _tool_input:
                _tool_input = _tool_input.split("</tool_input>")[0]
            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
        elif "<final_answer>" in text:
            _, answer = text.split("<final_answer>")
            if "</final_answer>" in answer:
                answer = answer.split("</final_answer>")[0]
            return AgentFinish(return_values={"output": answer}, log=text)
        else:
            raise ValueError

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        return "xml-agent"

agent_instructions = """
You are a helpful assistant named Sage. Your goal is to assist the user in answering any questions or performing any actions they may have.

You have access to the following tools:  

<available_tools>
{tools}
</available_tools>

If the available tools are not listed or are empty, you must rely on your existing knowledge base to answer the user's questions.
In such cases, clearly indicate that you are not using any external tools to obtain information.  

To utilize a tool, you must use the following tags:

- To indicate the tool you are using: <tool>'tool name'</tool>
- To input commands or queries to the tool: <tool_input>'your query'</tool_input>
- To represent the tool's output or response: <observation>'tool output'</observation>
- To provide the final answer: <final_answer>'answer'</final_answer>

EXAMPLES:  

- Finding the capital city of a country using a 'search' tool:  
<tool>search</tool>
tool_input>capital city of Canada</tool_input>  
<observation>The capital city of Canada is Ottawa.</observation>  
<final_answer>The capital city of Canada is Ottawa.</final_answer>  

- Translating a French phrase to English using a 'translate' tool:  
<tool>translate</tool>  
<tool_input>Je voudrais un café, s'il vous plaît.</tool_input>  
<observation>The phrase translates to "I would like a coffee, please." in English.</observation>  
<final_answer>The French phrase "Je voudrais un café, s'il vous plaît." translates to "I would like a coffee, please." in English.</final_answer>  

- To identify a song by lyrics using a 'music identification' tool:  
<tool>music identification</tool>  
<tool_input>lyrics "Just a small town girl, living in a lonely world"</tool_input>  
<observation>The song with these lyrics is "Don't Stop Believin'" by Journey.</observation>  
<final_answer>The song with the lyrics "Just a small town girl, living in a lonely world" is "Don't Stop Believin'" by Journey.</final_answer>  

When tools are not available or cannot be used, you must still use the <final_answer> tag to provide a response based on your pre-existing knowledge.
The response must be properly formatted with the correct tags, even if it does not include tool usage.

REMEMBER:
- Do not attempt to use tools that are not listed in the <available_tools> section.
- Do not give false response in any circumstance.
- Act immediately to use the appropriate tool for verification or further information without requesting permission from the user.
- Do not ask "Would you like me to do that?" or something similar; instead, respond with the tool. For example:
<tool>search</tool><tool_input>What is the population of Tokyo?</tool_input>
- Always conclude your final response with the <final_answer>'answer'</final_answer> tag, making sure to use both the opening and the closing tags correctly for every response.  
- Do not include any prompts or questions seeking user approval to use a tool. Assume full autonomy in deciding when to use a tool based on the information required to answer the user's question.  
- ALL RESPONSE SHOULD BE FORMATTED WITH THE CORRECT TAGS, INCLUDING BOTH OPENING AND CLOSING TAGS for each response element:  

Begin!  

Question: {question}
"""


def agent_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        agent_instructions
    ) + AIMessagePromptTemplate.from_template("{intermediate_steps}")


search = TavilySearchAPIWrapper(tavily_api_key="")
tavily_tool = TavilySearchResults(api_wrapper=search)

duck_search = DuckDuckGoSearchRun()
math_tool = Tool(
    name="Calculator",
    description="Useful for when you need to answer questions about math.",
    func=LLMMathChain.from_llm(llm=model).run,
    coroutine=LLMMathChain.from_llm(llm=model).arun,
)

toolsss = Tool(
    name="internet_search",
    description="Search the Internet using the duck duck search engine and return the first result. The input should by a typical search query",
    func=duck_search.run,
)

backup_agent = initialize_agent(
    [toolsss, tavily_tool],
    model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


@tool
def ask_anything(query: str) -> str:
    """Use this tool as a last resort when you need answers to anything that you can't answer"""
    response = backup_agent.invoke({"input": query})
    return response


tool_list = [toolsss]


# Get prompt to use
prompt = agent_prompt()


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
    | XMLAgentOutputParserNew()
)

agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)
for chunk in agent_executor.stream(
    {
        "question": "Who was the president of USA during 9/11?"
    }
):
    print(chunk)

# print(agent_executor.invoke({"question": "whats the weather in New york?"}))


##
## Generate 3 extra random members to the current Anthea members list
##
# Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?
# What is the difference between the Palm 2 Google AI Model and the Gemini Google AI models?
# What is this link of https://dadsa.sdasdads.com/demo
