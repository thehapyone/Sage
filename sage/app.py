from utils.jira_helper import Jira
from utils.jira_agent import IssueAgent, SummaryChain
from utils.sources import Source
from constants import sources_config

myjira = Jira()

key = "ANTHEA-906"
issue = myjira.get_issue(key)

agent = IssueAgent()

print(agent.generate_issue_template(issue))

# agent.planner(issue)
#source_list = Source()
#source_list.run()
