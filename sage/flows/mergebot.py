import random
from crewai.flow.flow import Flow, listen, router, start, and_
from crewai import Crew
from pydantic import BaseModel
from sage.constants import agents_crew
import re


# Get al Crew Instances
all_instances = {crew.name: crew for crew in agents_crew}

# Get the MR Processing Crew
mr_processing_crew: Crew = all_instances.get("MR Input Processing Crew")

# Get all Impact Assessment Crews
code_analysis_crew: Crew = all_instances.get("Code Analysis Assessment Crew")
complexity_assessment_crew: Crew = all_instances.get("Complexity Assessment Crew")
risk_assessment_crew: Crew = all_instances.get("Risk Assessment Crew")
test_coverage_crew: Crew = all_instances.get("Test Coverage Assessment Crew")

# Get the Impact Evaluator Crew
impact_evaluator_crew: Crew = all_instances.get("Impact Evaluator Crew")

# Get the MR Decision and Execution Crew
mr_decision_crew: Crew = all_instances.get("MR Publication Crew")


# Sample Input
bot_input = "MR https://gitlab.its.getingecloud.net/dts/lestrade/aide-recorder/-/merge_requests/53"


def extract_merge_request_id(output_string):
    pattern = r"https://.+/merge_requests/(\d+)"

    # Search the output string for the pattern
    match = re.search(pattern, output_string)

    # Extract and return the Merge Request ID if found
    if match:
        return int(match.group(1))
    else:
        return None


class MergeBotState(BaseModel):
    """Defines the bot data state"""

    mr_details: str = ""
    mr_id: int = None
    code_analysis_assessment: str = ""
    complexity_assessment: str = ""
    test_coverage_assessment: str = ""
    risk_assessment: str = ""
    impact_assessment: str = ""


class MergeBotFlow(Flow[MergeBotState]):
    initial_state = MergeBotState

    @start()
    def begin(self):
        print("Commencing and starting the MergeBot")

    @listen(begin)
    def mr_retriever(self):
        """Runs a Crew to extract Merge Request Details"""
        mr_details = mr_processing_crew.kickoff(inputs={"input": bot_input}).raw
        self.state.mr_details = mr_details
        self.state.mr_id = extract_merge_request_id(mr_details)

    @listen(mr_retriever)
    def code_analysis_assessment(self):
        """Runs the Code Analysis Assessment on the MR details"""

        self.state.code_analysis_assessment = code_analysis_crew.kickoff(
            inputs={"input": self.state.mr_details}
        ).raw

    @listen(mr_retriever)
    def complexity_assessment(self):
        """Runs the complexity_assessment on the MR details"""

        self.state.complexity_assessment = complexity_assessment_crew.kickoff(
            inputs={"input": self.state.mr_details}
        ).raw

    @listen(mr_retriever)
    def test_coverage_assessment(self):
        """Runs the Test Coverage Assessement on the MR details"""

        self.state.test_coverage_assessment = test_coverage_crew.kickoff(
            inputs={"input": self.state.mr_details}
        ).raw

    @listen(mr_retriever)
    def risk_assessment(self):
        """Runs the Risk Analysis Assessement on the MR details"""

        self.state.risk_assessment = risk_assessment_crew.kickoff(
            inputs={"input": self.state.mr_details}
        ).raw

    @listen(
        and_(
            code_analysis_assessment,
            complexity_assessment,
            test_coverage_assessment,
            risk_assessment,
        )
    )
    def impact_evaluator(self):
        """Runs the Impact Evaluator Analysis Assessement on the MR details"""

        self.state.impact_assessment = impact_evaluator_crew.kickoff(
            inputs={
                "input": self.state.mr_id,
                "code_analysis_assessment": self.state.code_analysis_assessment,
                "complexity_assessment": self.state.complexity_assessment,
                "test_coverage": self.state.test_coverage_assessment,
                "risk_assessment": self.state.risk_assessment,
            }
        ).raw
        print("\nFinal Impact Assessment Report:")
        print(self.state.impact_assessment)

    @listen(impact_evaluator)
    def mr_decision(self):
        """Runs the MR decision crew on the impact assessment report"""

        result = mr_decision_crew.kickoff(
            inputs={
                "input": self.state.mr_id,
                "impact_assessment_report": self.state.impact_assessment,
            }
        ).raw

        print("\nFinal Response:")
        print(result)

    @listen(mr_decision)
    def finish(self):
        print("MergeBot Processing Completed")

mergebot = MergeBotFlow()

if __name__ == "__main__":
    mergebot.kickoff()
    mergebot.plot("mergebot_flow")
