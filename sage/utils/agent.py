from langchain.chains.llm import LLMChain
from typing import Any
from jira import Issue, JIRAError

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI, ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from utils.jira_helper import Jira

issue_template = """
**Title**: {title}

**Key**: {key}

**Status**: {status}

**Description**: {description}

**Reporter**: {reporter}

**Comments**: 
{comments}

**Parent Issue Summary**: 
{parent}

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

SUMMARY_TEMPLATE = """
"As an AI assistant, your task is to concisely summarize Jira issues, incorporating all vital details, comments, and contexts. Your analysis will assist the assignee in effectively resolving the issue.

Your summary should:
 - Highlight key task-related information and issues.
 - Be detailed, accurate, and tailored for a technical audience, while staying clear and concise.
 - Use a professional, yet casual tone.
 - Use emojis sparingly and only if they add value.
 - Be well-structured and easy to comprehend.

Additional guidelines:
 - Go straight to the summary without introducing or acknowledging yourself.
 - If a parent issue summary is present, use it for context and additional information for the main task, but don't include parent tasks in the Definition of Done (DoD) for this task.
 - Write in the second person and use appropriate prepositions, as your summary will be added to the issue's comments.
"""

PARENT_SUMMARY_TEMPLATE = SUMMARY_TEMPLATE + """
Please note, as this is a parent issue, your summary should be short, concise and contain max of three paragraphs. Exclude any summary headers, as they'll be added to the child summary field. Don't forget to include the issue key in the summary.
"""


class SummaryChain:

    def __init__(self, system_template: str = SUMMARY_TEMPLATE) -> None:

        DEPLOYMENT_NAME = "gpt4-8k"
        llm = AzureChatOpenAI(
            deployment_name=DEPLOYMENT_NAME)

        # llm = ChatOllama(model="llama2:13b",
        #                  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{issue}"),
        ])

        self.chain = LLMChain(llm=llm, prompt=chat_prompt)

    def summarize(self, issue: str) -> Any:
        # Summarize the given issue
        return self.chain.invoke({
            "issue": issue
        }).get("text")


class IssueAgent:

    def __init__(self) -> None:

        self.template = PromptTemplate.from_template(issue_template)

    def generate_issue_template(self, issue: Issue) -> str:
        # Given an issue, it returns a formatted issue template

        issue_formatted = self.template.format(
            title=self.get_field("summary", issue),
            key=issue.key,
            status=issue.fields.status.name,
            description=self.get_field("description", issue),
            reporter=issue.fields.reporter.displayName,
            comments=self._get_comments(issue),
            parent=self._get_parent(issue)
        )

        return issue_formatted

    @staticmethod
    def get_field(field: str, issue: Issue) -> Any:
        try:
            return issue.get_field(field)
        except AttributeError:
            return "None"

    def _get_parent(self, issue) -> str:
        """Returns a summarized version of the parent Issue"""
        parent_issue = self.get_field("parent", issue)     # type: Issue
        if parent_issue == "None":
            return "No parent issue"

        # Check if the parent is an EPIC issue
        issue_type = parent_issue.get_field("issuetype")

        if issue_type.name == "Epic":
            return "No parent issue"

        # Summarize this parent issue
        issue_formatted = self.generate_issue_template(
            Jira().get_issue(parent_issue.key))

        parent_summary = SummaryChain(
            PARENT_SUMMARY_TEMPLATE).summarize(issue_formatted)
        return parent_summary

    def _get_comments(self, issue) -> str:
        """Extract out the comments attached to an issue"""
        comments = self.get_field("comment", issue)
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

    def summarize(self, issue: Issue) -> None:
        """
        Given an Issue, this method helps to provide a detail summary of a Jira issue and then publish the issue
        in the commnents field of the Jira ticket

        Args:
            issue (Issue): A Jira Issue object
        """

        # Get the issue format out
        issue_formatted = self.generate_issue_template(issue)

        chain = SummaryChain()
        summary_text = chain.summarize(issue_formatted)

    def publish_summary(self, summary: str, issue_key: str) -> None:
        """
        Function helps to publish the summary

        Args:
            summary (str): The issue summary
            issue_key (str): The issue key
        """
        Jira().add_comment(issue_key=issue_key, body=summary)
