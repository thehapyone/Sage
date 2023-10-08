from utils.jira_helper import Jira
from utils.agent import IssueAgent, SummaryChain

myjira = Jira()

key = "ANTHEA-943"
issue = myjira.get_issue(key)

agent = IssueAgent()
issue_template = agent.generate_issue_template(issue)
print(issue_template)

# chain = SummaryChain()

# result = chain.summarize(issue_template)
# print(result)