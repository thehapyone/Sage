from typing import List
from functools import cached_property
from jira import JIRA, Issue, JIRAError

from constants import jira_config, JIRA_QUERY


class Jira:

    def __init__(self) -> None:
        self._jira = JIRA(
            server=jira_config["url"],
            basic_auth=(jira_config["username"], jira_config["password"])
        )
        self._tracked_issues = list()  # type: List[Issue]

        self._polling_interval = jira_config["polling_interval"]

    @property
    def _tracked_ids(self) -> List[str]:
        # A list of tracked Issue Index ID
        ids = [int(issue.id) for issue in self._tracked_issues]
        return ids if ids else [0]

    @cached_property
    def query(self) -> str:
        return JIRA_QUERY.format(
            project=jira_config["project"],
            status=jira_config["status_todo"],
            assignee=jira_config["username"]
        )

    def search_issues(self):
        # Search Issues matching the given query
        fields = [
            "summary",
            "description",
            "attachment",
            "comment",
            "issuelinks",
            "reporter",
        ]
        issues = self._jira.search_issues(self.query,
                                          startAt=self._tracked_ids[-1],
                                          fields=fields)

        for issue in issues:
            # check if issue not yet tracked
            if issue.id not in self._tracked_ids:
                self._tracked_issues.append(issue)

    def run() -> None:
        # Commence the whole Jira issues tracking and saving it.
        """
        -- Query Jira for all Issues that matches a given Query
        -- Check if found issues are not already tracked
        -- Add issue to the Issue Queue
        -- Wait till the polling interval is over        

        """
