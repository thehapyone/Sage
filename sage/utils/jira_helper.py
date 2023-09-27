from jira import JIRA, JIRAError
from jira.client import ResultList

from constants import jira_config


class Jira:

    def __init__(self) -> None:
        self.jira = JIRA(
            server=jira_config["url"],
            basic_auth=(jira_config["username"], jira_config["password"])
        )
