from utils.jira_helper import Jira
from utils.agent import IssueAgent, SummaryChain

myjira = Jira()

key = "ANTHEA-906"
issue = myjira.get_issue(key)

agent = IssueAgent()

agent.planner(issue)
