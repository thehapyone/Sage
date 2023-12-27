from pathlib import Path
from typing import List
from typing import Union

from transformers import AutoModel
from langchain.schema.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, AIMessagePromptTemplate
from langchain.tools import Tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.schema import AgentAction, AgentFinish

from markdown import markdown
from html2text import HTML2Text
from unstructured.partition.auto import partition_md

text_maker = HTML2Text()
text_maker.ignore_links = False
text_maker.ignore_images = True
text_maker.ignore_emphasis = True


class JinaAIEmebeddings(Embeddings):
    """Am embedding class powered by hugging face jinaAI"""

    def __init__(
        self,
        cache_dir: str,
        revision: str = "7302ac470bed880590f9344bfeee32ff8722d0e5",
        jina_model: str = "jinaai/jina-embeddings-v2-base-en",
    ):
        """Initialize the Jina Embeddings"""
        Path(cache_dir).mkdir(exist_ok=True)

        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=jina_model,
            trust_remote_code=True,
            cache_dir=cache_dir,
            resume_download=True,
            revision=revision,
        )

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)


def markdown_to_text_using_html2text(markdown_text: str) -> str:
    """Convert the markdown docs into plaintext using the html2text plugin

    Args:
        markdown_text (str): Markdown text

    Returns:
        str: Plain text
    """
    html = markdown(markdown_text)
    return text_maker.handle(html).replace("\\", "")

def convert_intermediate_steps(intermediate_steps: dict):
    """
    Convert intermediate steps from agents into string outputs
    """
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool>"
            f"<tool_input>{action.tool_input}</tool_input>"
            f"<observation>{observation}</observation>"
        )
    return log

def agent_prompt(instructions: str) -> ChatPromptTemplate:
    """Generate a prompt template for XML agents"""
    return ChatPromptTemplate.from_template(
        instructions
    ) + AIMessagePromptTemplate.from_template("{intermediate_steps}")

def convert_tools(tools: List[Tool]):
    """Logic for converting tools to string to for usage in prompt"""
    result = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    return result

class CustomXMLAgentOutputParser(XMLAgentOutputParser):
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

    def parse(self, text: str) -> AgentAction | AgentFinish:
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
            return AgentFinish(return_values={"output": text}, log=text)
