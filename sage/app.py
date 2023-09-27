from utils.jira_helper import Jira

myjira = Jira().jira

jql = f'project = "ANTHEA" and status = "To Do" and assignee = "jira-user.act@getinge.com" ORDER BY created ASC'

issues = myjira.search_issues(jql)

print(issues[0].fields.description)

print(issues)