from functools import cached_property
from queue import Queue
from time import sleep
from typing import List

from constants import JIRA_QUERY, jira_config
from jira import JIRA, Issue, JIRAError
from jira.resources import Comment, Resource

issue_fields = [
    "summary",
    "status",
    "parent",
    "description",
    "attachment",
    "comment",
    "issuelinks",
    "reporter",
    "issuetype",
]


class Jira:
    _instance = None
    _run_session = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self._jira = JIRA(
            server=jira_config.url,
            basic_auth=(jira_config.username, jira_config.password.get_secret_value()),
        )
        self._tracked_issues = list()  # type: List[Issue]

        self._polling_interval = jira_config.polling_interval

        self._queue = Queue()

    @property
    def queue(self) -> Queue:
        return self._queue

    @property
    def _tracked_ids(self) -> List[str]:
        # A list of tracked Issue Index ID
        ids = [int(issue.id) for issue in self._tracked_issues]
        return ids if ids else [0]

    @cached_property
    def query(self) -> str:
        return JIRA_QUERY.format(
            project=jira_config.project,
            status=jira_config.status_todo,
            assignee=jira_config.username,
        )

    def get_issue(self, issue_key: str) -> Issue:
        """Returns a issue"""
        return self._jira.issue(id=issue_key)

    def add_comment(self, issue_key: str, body: str) -> None:
        """Add a comment to the Issue"""
        response = self._jira.add_comment(issue=issue_key, body=body)

        if response:
            return

        raise Exception(f"Error adding comment. key: {issue_key} and body: {body}")

    def search_issues(self):
        """
        Search issues matching the configured query and adds them to the queue.
        """
        issues = self._jira.search_issues(
            self.query, startAt=self._tracked_ids[-1], fields=issue_fields
        )

        for issue in issues:
            comments = issue.get_field("comment")
            if isinstance(comments, Comment):
                print(True)
            # check if issue not yet tracked
            if issue.id not in self._tracked_ids:
                self._tracked_issues.append(issue)
                # add to queue
                self.queue.put(issue)

    def run(self) -> None:
        # Commence the whole Jira issues tracking and saving it.
        """
        -- Query Jira for all Issues that matches a given Query
        -- Check if found issues are not already tracked
        -- Add issue to the Issue Queue
        -- Wait till the polling interval is over

        """

        if self._run_session:
            print("There is already an active session")
            return

        self._run_session = True

        while True:
            try:
                self.search_issues()
                sleep(self._polling_interval)
            except KeyboardInterrupt:
                print("keyboard interrupt")
                break
            except Exception as error:
                print("AN error has occurred")
                raise error
