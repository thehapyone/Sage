from functools import lru_cache

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from sage.constants import LLM_MODEL

prompt_instruction = """
Your task is to generate succinct labels for data sources, formatted as '[Category]: [Descriptor]'.

Requirements:
- Use the data source type (e.g., 'Web', 'Confluence', etc) as the Category (one word category).
- Create a Descriptor with a maximum of four words, capturing the source's essence.
- For web sources, emphasize the root domain and relevant URL path details, especially the final segment.

Example:
Input: Web: https://docs.gitlab.com/ee/ci/
Label: Web: GitLab CI Docs

Now, label this input:
Input: {input}
Response format:
[Category]: [Descriptor]
"""

label_prompt = PromptTemplate.from_template(prompt_instruction)


@lru_cache(maxsize=10)
async def generate_source_label(source_metadata: str) -> str:
    """Generate a short label representation for the given source metadata"""
    label_chain = label_prompt | LLM_MODEL | StrOutputParser()
    return await label_chain.ainvoke({"input": source_metadata})
