import re
from typing import Any, Dict, List, Type
import json

from langchain_community.utilities.gitlab import GitLabAPIWrapper
from crewai_tools import BaseTool
from pydantic import BaseModel, Field


class GitLabAPIWrapperExtra(GitLabAPIWrapper):
    """
    Extended GitLab API Wrapper with additional merge request functionalities.
    """

    def strip_ansi_codes(self, text: str) -> str:
        """
        Removes ANSI escape sequences from the text.

        Parameters:
            text (str): The text containing ANSI codes.

        Returns:
            str: The cleaned text without ANSI codes.
        """
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", text)

    def extract_step_script_section(self, log: str) -> str:
        """
        Extracts the 'step_script' section from the job log.

        Parameters:
            log (str): The full job log.

        Returns:
            str: The 'step_script' section of the log.
        """
        # Patterns to identify section starts and ends
        section_start_pattern = r"section_start:\d+:(\w+)"
        section_end_pattern = r"section_end:\d+:(\w+)"

        lines = log.split("\n")
        in_step_script = False
        step_script_log = []

        for line in lines:
            # Check for section start
            start_match = re.match(section_start_pattern, line)
            if start_match:
                section_name = start_match.group(1)
                if section_name == "step_script":
                    in_step_script = True
                    continue  # Skip the section start line

            # Check for section end
            end_match = re.match(section_end_pattern, line)
            if end_match:
                section_name = end_match.group(1)
                if section_name == "step_script":
                    in_step_script = False
                    continue  # Skip the section end line

            # Collect lines within 'step_script' section
            if in_step_script:
                step_script_log.append(line)

        return "\n".join(step_script_log)

    def parse_job_log(self, log: str, job_status: str) -> Dict[str, Any]:
        """
        Parses the job log to extract warnings and errors with context,
        focusing on the 'step_script' section.

        Parameters:
            log (str): The job log string.
            job_status (str): Status of the job (e.g., 'success', 'failed').

        Returns:
            Dict[str, Any]: A dictionary containing lists of warnings and errors with context.
        """
        # First, extract the 'step_script' section
        step_script_lines = log.split("\n")

        warnings = []
        errors = []

        # Remove ANSI codes from lines
        lines = [self.strip_ansi_codes(line) for line in step_script_lines]

        context_size = 5
        # If the job failed, capture the tail of the step_script
        if job_status == "failed":
            error_context = "\n".join(lines[-30:])
            errors.append(error_context)
        else:
            # For successful jobs, look for warnings and errors with context
            i = 0
            while i < len(lines):
                line = lines[i]
                if "WARNING" in line or "WARN" in line:
                    # Collect context lines before and after
                    start = max(i - context_size, 0)
                    end = min(i + context_size, len(lines))
                    context = "\n".join(lines[start:end])
                    warnings.append(context)
                    i = end  # Skip lines already processed
                else:
                    i += 1

        return {"warnings": warnings, "errors": errors}

    def get_pipeline_job(self, job_id: int) -> dict:
        """Gets the job information"""
        job = self.gitlab_repo_instance.jobs.get(job_id)

        # Get the job logs
        job_log = job.trace()

        # Parse the log to extract warnings and errors
        parsed_log = self.parse_job_log(job_log.decode("utf-8"), job.status)
        num_warnings = len(parsed_log["warnings"])
        num_errors = len(parsed_log["errors"])

        # Prepare job details
        job_detail = {
            "id": job.id,
            "name": job.name,
            "status": job.status,
            "stage": job.stage,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "duration": job.duration,
            "web_url": job.web_url,
            "warnings_count": num_warnings,
            "errors_count": num_errors,
        }

        # Include sample of warnings and errors if any
        if num_warnings > 0:
            job_detail["warnings_sample"] = parsed_log["warnings"][:5]
        if num_errors > 0:
            job_detail["errors_sample"] = parsed_log["errors"][:5]

        return job_detail

    def get_pipeline_details(self, pipeline_id: int) -> str:
        """
        Retrieves detailed information of a pipeline, including job logs and summaries.

        Parameters:
            pipeline_id (int): The ID of the pipeline.

        Returns:
            str: A formatted string containing pipeline details and job summaries.
        """
        try:
            # Fetch the pipeline
            pipeline = self.gitlab_repo_instance.pipelines.get(pipeline_id)

            # Basic pipeline info
            pipeline_info = {
                "id": pipeline.id,
                "status": pipeline.status,
                "ref": pipeline.ref,
                "sha": pipeline.sha,
                "created_at": pipeline.created_at,
                "updated_at": pipeline.updated_at,
                "web_url": pipeline.web_url,
            }

            # Fetch jobs associated with the pipeline
            jobs = pipeline.jobs.list(get_all=True)
            jobs_details = []
            total_warnings = 0
            total_errors = 0

            for job in jobs:
                # Fetch job log
                job_detail = self.get_pipeline_job(job.id)

                num_warnings = job_detail["warnings_count"]
                num_errors = job_detail["errors_count"]
                total_warnings += num_warnings
                total_errors += num_errors
                jobs_details.append(job_detail)

            # Summarize pipeline results
            summary = {
                "Pipeline Info": pipeline_info,
                "Total Jobs": len(jobs_details),
                "Total Warnings": total_warnings,
                "Total Errors": total_errors,
                "Jobs Details": jobs_details,
            }

            # Format the output for better readability
            formatted_output = json.dumps(summary, indent=2, default=str)
            return formatted_output

        except Exception as e:
            return f"Failed to retrieve pipeline details for Pipeline ID {pipeline_id}: {str(e)}"

    def parse_diff_content(self, diff: str):
        """
        Parses the diff content to calculate lines added and removed.

        Parameters:
            diff (str): The diff string.

        Returns:
            Tuple[int, int]: Number of lines added and removed.
        """
        additions = len(re.findall(r"^(\+[^+])", diff, re.MULTILINE))
        deletions = len(re.findall(r"^-(?!-)", diff, re.MULTILINE))
        return additions, deletions

    def pretty_print_merge_request(self, mr_details: dict):

        # Prepare MR Metadata string
        mr_metadata = [
            "Merge Request Details:",
            f"Title: {mr_details['mr']['title']}",
            f"Author: {mr_details['mr']['author']['name']} ({mr_details['mr']['author']['username']})",
            f"State: {mr_details['mr']['state']}",
            f"Created At: {mr_details['mr']['created_at']}",
            f"Updated At: {mr_details['mr']['updated_at']}",
            f"Source Branch: {mr_details['mr']['source_branch']}",
            f"Target Branch: {mr_details['mr']['target_branch']}",
            f"Labels: {', '.join(mr_details['mr']['labels'])}",
            f"Pipeline Status: {mr_details['mr']['pipeline_status']}",
            f"Approvals Required: {mr_details['mr']['approvals_required']}",
            f"Approvals Received: {mr_details['mr']['approvals_received']}",
            f"Web URL: {mr_details['mr']['web_url']}",
            f"Merge Status: {mr_details['mr']['merge_status']}",
            f"First Contribution: {mr_details['mr']['first_contribution']}",
        ]

        # Append Assignee Details
        assignee_info = [
            f"- {assignee['name']} ({assignee['username']})"
            for assignee in mr_details["mr"]["assignees"]
        ]
        mr_metadata.append(
            f"Assignees:\n{chr(10).join(assignee_info) if assignee_info else 'None'}"
        )

        # Prepare Changes and Statistics strings
        changes_info = ["\nChanges:"]
        for change in mr_details["changes"]:
            changes_info.extend(
                [
                    f"File: {change['new_path']}",
                    f"  - Lines Added: {change['lines_added']}",
                    f"  - Lines Removed: {change['lines_removed']}",
                    f"  - Change Type: {'New File' if change['new_file'] else 'Renamed File' if change['renamed_file'] else 'Deleted File' if change['deleted_file'] else 'Modified'}",
                    f"  - Generated File: {'Yes' if change['generated_file'] else 'No'}",
                    f"  - Diff:\n{change['diff']}\n",
                ]
            )

        stats_info = [
            "Statistics:",
            f"Total Files Changed: {mr_details['stats']['total_files_changed']}",
            f"Total Lines Added: {mr_details['stats']['total_lines_added']}",
            f"Total Lines Removed: {mr_details['stats']['total_lines_removed']}",
        ]

        # Compile all information into a single string for output
        full_output = "\n".join(mr_metadata + changes_info + stats_info)

        # Return the formatted string
        return full_output

    def get_merge_request(self, mr_iid: int) -> str:
        """
        Retrieves comprehensive details of a merge request, including diffs and statistics.

        Parameters:
            mr_iid (int): The internal ID (iid) of the merge request.

        Returns:
            Dict[str, Any]: A dictionary containing detailed MR information.
        """
        try:
            # Fetch the merge request
            mr = self.gitlab_repo_instance.mergerequests.get(mr_iid)

            # Fetch author details
            author = mr.author
            author_details = {
                "id": author["id"],
                "username": author["username"],
                "name": author["name"],
            }

            # Fetch assignee details
            assignees_details = []
            for assignee in mr.assignees:
                assignees_details.append(
                    {
                        "id": assignee["id"],
                        "username": assignee["username"],
                        "name": assignee["name"],
                    }
                )

            # Fetch MR changes/diffs
            changes = []
            total_lines_added = 0
            total_lines_removed = 0

            mr_changes = mr.changes(unidiff=True, access_raw_diffs=True)
            for change in mr_changes["changes"]:
                # Calculate lines added and removed from the diff content
                lines_added, lines_removed = self.parse_diff_content(
                    change.get("diff", "")
                )
                total_lines_added += lines_added
                total_lines_removed += lines_removed

                changes.append(
                    {
                        "old_path": change.get("old_path"),
                        "new_path": change.get("new_path"),
                        "a_mode": change.get("a_mode"),
                        "b_mode": change.get("b_mode"),
                        "new_file": change.get("new_file", False),
                        "renamed_file": change.get("renamed_file", False),
                        "deleted_file": change.get("deleted_file", False),
                        "generated_file": change.get("generated_file", False),
                        "diff": change.get("diff"),
                        "lines_added": lines_added,
                        "lines_removed": lines_removed,
                    }
                )

            # Calculate statistics
            stats = {
                "total_files_changed": len(changes),
                "total_lines_added": total_lines_added,
                "total_lines_removed": total_lines_removed,
            }

            # Get approvals
            approvals = mr.approvals.get()

            # # MR Dicussions
            # discussion_notes = []

            # discussions = mr.discussions.list(all=True)
            # for discussion in discussions:
            #     discussion_id = discussion.attributes.get("id")
            #     notes = discussion.attributes.get("notes")
            #     for note in notes:
            #         data = {"discussion_id": discussion_id, "note": note}
            #         discussion_notes.append(data)

            # Compile the final MR details
            mr_details = {
                "mr": {
                    # "id": mr.id,
                    "iid": mr.iid,
                    "project_id": mr.project_id,
                    "title": mr.title,
                    "description": mr.description,
                    "state": mr.state,
                    "created_at": mr.created_at,
                    "updated_at": mr.updated_at,
                    "author": author_details,
                    "assignees": assignees_details,
                    "source_branch": mr.source_branch,
                    "target_branch": mr.target_branch,
                    "labels": mr.labels,
                    "web_url": mr.web_url,
                    "merge_status": mr.detailed_merge_status,
                    "first_contribution": mr.first_contribution,
                    "pipeline_status": (
                        mr.head_pipeline.get("status") if mr.head_pipeline else None
                    ),
                    "approvals_required": approvals.approvals_required,
                    "approvals_received": len(approvals.approved_by),
                    # "approvals_received": mr.approvals_received,
                    # "discussion_unresolved_count": mr.discussion_unresolved_count,
                },
                "changes": changes,
                "stats": stats,
            }

            return self.pretty_print_merge_request(mr_details)

        except Exception as e:
            return {
                "error": f"Failed to retrieve merge request details for MR IID {mr_iid}: {str(e)}"
            }

    def post_merge_request_comment(self, mr_iid: int, comment: str) -> str:
        """
        Posts a comment to a merge request.

        Parameters:
            mr_iid (int): The internal ID (iid) of the merge request.
            comment (str): The comment text to post.

        Returns:
            str: Success or failure message.
        """
        try:
            mr = self.gitlab_repo_instance.mergerequests.get(mr_iid)
            mr.notes.create({"body": comment.strip()})
            return f"Successfully posted comment to Merge Request {mr_iid}."
        except Exception as e:
            return f"Failed to post comment to Merge Request {mr_iid}: {str(e)}"

    def post_merge_request_thread_comment(
        self, mr_iid: int, note_id: int, comment: str
    ) -> str:
        """
        Posts a reply to an existing comment thread in a merge request.

        Parameters:
            mr_iid (int): The internal ID (iid) of the merge request.
            note_id (int): The ID of the note (comment) to reply to.
            comment (str): The reply comment text.

        Returns:
            str: Success or failure message.
        """
        try:
            mr = self.gitlab_repo_instance.mergerequests.get(mr_iid)
            note = mr.notes.get(note_id)
            note.notes.create({"body": comment})
            return f"Successfully posted reply to comment {note_id} in Merge Request {mr_iid}."
        except Exception as e:
            return f"Failed to post reply to comment {note_id} in Merge Request {mr_iid}: {str(e)}"

    def get_merge_request_comments(self, mr_iid: int) -> List[Dict[str, Any]]:
        """
        Retrieves all comments for a specific merge request.

        Parameters:
            mr_iid (int): The internal ID (iid) of the merge request.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing comment details.
        """
        try:
            mr = self.gitlab_repo_instance.mergerequests.get(mr_iid)
            notes = mr.notes.list(all=True)
            comments = []
            for note in notes:
                comments.append(
                    {
                        "id": note.id,
                        "body": note.body,
                        "author": note.author["username"],
                        "created_at": note.created_at,
                        "updated_at": note.updated_at,
                    }
                )
            return comments
        except Exception as e:
            return [
                {
                    "error": f"Failed to get comments for Merge Request {mr_iid}: {str(e)}"
                }
            ]

    def approve_merge_request(self, mr_iid: int) -> str:
        """
        Approves a merge request.

        Parameters:
            mr_iid (int): The internal ID (iid) of the merge request.

        Returns:
            str: Success or failure message.
        """
        try:
            # Retrieve the merge request
            mr = self.gitlab_repo_instance.mergerequests.get(mr_iid)
            # Approve the merge request
            mr.approve()
            return f"Successfully approved Merge Request {mr_iid}."
        except Exception as e:
            return f"Failed to approve Merge Request {mr_iid}: {str(e)}"

    def run(self, mode: str, query: str = "", body: dict = {}) -> str:
        # Parent class handling
        original_modes = {
            "get_issues",
            "get_issue",
            "comment_on_issue",
            "create_file",
            "create_pull_request",
            "read_file",
            "update_file",
            "delete_file",
        }

        if mode in original_modes:
            return super().run(mode, query)

        elif mode == "get_merge_request":
            try:
                mr_iid = int(query.strip())
                mr_details = self.get_merge_request(mr_iid)
                return mr_details
            except ValueError:
                return "Invalid input. Please provide the Merge Request number as an integer."
        elif mode == "post_merge_request_comment":
            try:
                return self.post_merge_request_comment(**body)
            except ValueError:
                return "Invalid input format. Expected:\n<mr_iid>\n\n<comment>"
        elif mode == "approve_merge_request":
            try:
                return self.approve_merge_request(**body)
            except ValueError:
                return "Invalid input format."
        elif mode == "get_pipeline_details":
            try:
                pipeline_id = int(query.strip())
                pipeline_details = self.get_pipeline_details(pipeline_id)
                return pipeline_details
            except ValueError:
                return "Invalid input. Please provide the Pipeline ID as an integer."
        elif mode == "post_merge_request_thread_comment":
            try:
                parts = query.split("\n\n", 2)
                if len(parts) != 3:
                    raise ValueError
                mr_iid = int(parts[0].strip())
                note_id = int(parts[1].strip())
                comment = parts[2].strip()
                return self.post_merge_request_thread_comment(mr_iid, note_id, comment)
            except ValueError:
                return "Invalid input format. Expected:\n<mr_iid>\n\n<note_id>\n\n<comment>"
        elif mode == "get_merge_request_comments":
            try:
                mr_iid = int(query.strip())
                comments = self.get_merge_request_comments(mr_iid)
                return json.dumps(comments, indent=2)
            except ValueError:
                return "Invalid input. Please provide the Merge Request number as an integer."
        else:
            raise ValueError(f"Invalid mode: {mode}")


# prompt.py (continued)


GET_MERGE_REQUEST_PROMPT = """
This tool will fetch the complete details of a specific merge request.
Example input: {\"merge_request_iid\": \"10\"}
"""

POST_MERGE_REQUEST_COMMENT_PROMPT = """
This tool is useful when you need to post a comment to a merge request in the GitLab repository.
    
For example, to post a comment "Looks good to me!" to merge request number 10:

Example input: {\"merge_request_iid\": \"10\", \"comment\": \"Looks good to me!\"}

"""

APPROVE_MERGE_REQUEST_PROMPT = """
This tool is useful when you need to approve a merge request in a GitLab repository.
    
For example, to approve a merge request number 10:

Example input: {\"merge_request_iid\": \"10\"}

"""

POST_MERGE_REQUEST_THREAD_COMMENT_PROMPT = """
This tool is useful when you need to post a reply to an existing comment thread in a merge request. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:
    
- First, you must specify the merge request number (IID) as an integer.
- Then, you must place two newlines.
- Then, you must specify the existing comment ID as an integer.
- Then, you must place two newlines.
- Then, you must specify your reply comment.
    
For example, to reply "I agree with your assessment." to comment ID 5 in merge request number 10, you would pass in the following string:

10

5

I agree with your assessment.
"""


FETCH_PIPELINE_DETAILS_PROMPT = """  
This tool will fetch detailed information of a specific pipeline, including job logs and summarized relevant information.  
Example input: {"pipeline_id": "12345"}  
"""


class GitlabPipelineToolSchema(BaseModel):
    """Input for GitlabPipelineTool."""

    pipeline_id: str = Field(..., description="The ID of the pipeline")


class GitlabPipelineTool(BaseTool):
    """Tool for fetching and summarizing pipeline details from GitLab."""

    mode: str = "get_pipeline_details"
    name: str = "Get Pipeline Details"
    description: str = FETCH_PIPELINE_DETAILS_PROMPT
    args_schema: Type[BaseModel] = GitlabPipelineToolSchema
    api_wrapper: GitLabAPIWrapperExtra = GitLabAPIWrapperExtra(
        gitlab_branch="main",
    )

    def _run(
        self,
        **kwargs: Any,
    ) -> str:
        """Use the GitLab API to fetch pipeline details."""
        pipeline_id = kwargs.get("pipeline_id")

        if not pipeline_id:
            return "The Pipeline ID is required."

        return self.api_wrapper.run(self.mode, str(pipeline_id))


class GitlabToolSchema(BaseModel):
    """Input for GitlabTools."""

    merge_request_iid: str = Field(
        ..., description="The project level ID of the merge request"
    )


class GitlabMergeRequestTool(BaseTool):
    """Tool for getting merge requests from the GitLab API."""

    mode: str = "get_merge_request"
    name: str = "Get Merge Requests"
    description: str = GET_MERGE_REQUEST_PROMPT
    args_schema: Type[BaseModel] = GitlabToolSchema
    api_wrapper: GitLabAPIWrapperExtra = GitLabAPIWrapperExtra(
        gitlab_branch="main",
    )

    def _run(
        self,
        **kwargs: Any,
    ) -> str:
        """Use the GitLab API to run an operation."""
        mr_iid = kwargs.get("merge_request_iid")

        if not mr_iid:
            return "The Merge Request IID is required."

        return self.api_wrapper.run(self.mode, str(mr_iid))


class GitlabCommentToolSchema(BaseModel):
    """Input for GitlabTools."""

    merge_request_iid: str = Field(
        ..., description="The project level ID of the merge request"
    )
    comment: str = Field(
        ..., description="The merge request comment to be posted to the MR"
    )


class GitlabMergeCommentTool(BaseTool):
    """Tool for posting merge comments using the Gitlab API."""

    mode: str = "post_merge_request_comment"
    name: str = "Post Merge Requests Comment"
    description: str = POST_MERGE_REQUEST_COMMENT_PROMPT
    args_schema: Type[BaseModel] = GitlabCommentToolSchema
    api_wrapper: GitLabAPIWrapperExtra = GitLabAPIWrapperExtra(
        gitlab_branch="main",
    )

    def _run(
        self,
        **kwargs: Any,
    ) -> str:
        """Use the GitLab API to run an operation."""
        mr_iid = kwargs.get("merge_request_iid")
        mr_comment = kwargs.get("comment")

        if not (mr_iid or mr_comment):
            return "The Merge Request IID and message are required."

        return self.api_wrapper.run(
            self.mode, body={"mr_iid": mr_iid, "comment": mr_comment}
        )


class GitlabMergeApprovalTool(BaseTool):
    """Tool for approving Gitlab merge requests."""

    mode: str = "approve_merge_request"
    name: str = "Approve Merge Requests Comment"
    description: str = APPROVE_MERGE_REQUEST_PROMPT
    args_schema: Type[BaseModel] = GitlabToolSchema
    api_wrapper: GitLabAPIWrapperExtra = GitLabAPIWrapperExtra(
        gitlab_branch="main",
    )

    def _run(
        self,
        **kwargs: Any,
    ) -> str:
        """Use the GitLab API to run an operation."""
        mr_iid = kwargs.get("merge_request_iid")

        if not mr_iid:
            return "The Merge Request IID are required."

        return self.api_wrapper.run(self.mode, body={"mr_iid": int(mr_iid)})
