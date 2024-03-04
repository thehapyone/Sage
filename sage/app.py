import asyncio

from sage.utils.sources import Source

# myjira = Jira()

# key = "ANTHEA-906"
# issue = myjira.get_issue(key)

# agent = IssueAgent()

# print(agent.generate_issue_template(issue))

# # agent.planner(issue)
source_list = Source()

# qa_tool = SourceQAService(mode="tool")
# res = qa_tool._run("Hello, how are you?")
# print(res)

asyncio.run(source_list.run())
