from langchain.chains.llm import LLMChain
from typing import Any
from jira import Issue, JIRAError
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI, ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

issue_template = """
**Title**: {title}

**Key**: {key}

**Status**: {status}

**Description**: {description}

**Reporter**: {reporter}

**Comments**: 
{comments}

"""

issue_templateb_bak = """
**Title**: {title}

**Key**: {key}

**Status**: {status}

**Description**: {description}

**Reporter**: {reporter}

**Parent Issue**: <parent_summary>

**Comments**: 
{comments}

Linked Issues:
    <linked_issues>
"""


linked_issue_template = """
{issue_key}: {issue_summary}
"""

summary_template = """
You are an AI assistant that helps in summarization of Jira issues.
Please provide a comprehensive summary and analysis of the Jira issue, taking into account all relevant details, comments, and context.
This summary will be used by the agent assigned to work on the issue for a more efficient resolution.
The summary should be concise and to the point, highlighting the key information and issues related to the task.
"""


class SummaryChain:

    def __init__(self) -> None:

        DEPLOYMENT_NAME = "gpt4-8k"
        llm1 = AzureChatOpenAI(
            deployment_name=DEPLOYMENT_NAME)

        llm = ChatOllama(model="llama2",
                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", summary_template),
            ("human", "{issue}"),
        ])

        self.chain = LLMChain(llm=llm, prompt=chat_prompt)

    def summarize(self, issue: str) -> Any:
        # Summarize the given issue
        return self.chain.invoke({
            "issue": issue
        }).get("text")


class IssueAgent:

    def __init__(self, issue: Issue) -> None:

        self.template = PromptTemplate.from_template(issue_template)
        self._issue = issue

    def generate_issue_template(self) -> str:
        # Given an issue, it returns a formatted issue template

        issue_formatted = self.template.format(
            title=self.get_field("summary"),
            key=self._issue.key,
            status=self._issue.fields.status.name,
            description=self.get_field("description"),
            reporter=self._issue.fields.reporter.displayName,
            comments=self._get_comments()
        )

        return issue_formatted

    def get_field(self, field: str) -> Any:
        return self._issue.get_field(field)

    def _get_comments(self) -> str:
        """Extract out the comments attached to an issue"""
        comments = self.get_field("comment")
        if comments is None:
            return "None"

        comment_template = """
        {author}: {content}
        """
        comment_list = []
        for comment in comments.comments:
            message = comment_template.format(
                author=comment.author.displayName,
                content=comment.body
            ).strip()
            comment_list.append(message)

        if not comment_list:
            return "None"

        return "\n".join(comment_list)
