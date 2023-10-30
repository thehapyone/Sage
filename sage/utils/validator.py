from pydantic import BaseModel
from typing import Optional, List


class Password(BaseModel):
    """  
    Password Base Model.  
    """
    password: Optional[str] = None


class SourceData(Password):
    """  
    Source Data Model. Inherits Password.  
    """
    username: str
    server: str
    spaces: List[str]


class Web(BaseModel):
    """  
    Web Model.  
    """
    links: List[str]
    nested: bool


class Source(BaseModel):
    """  
    Source Model.  
    """
    confluence: SourceData
    gitlab: SourceData
    # Web is optional, and defaults to None if not present
    web: Optional[Web] = None


class Jira_Config(Password):
    """  
    Jira Model. Inherits Password.  
    """
    url: str
    username: str
    polling_interval: int
    project: str
    status_todo: str


class Config(BaseModel):
    """  
    Config Model.  
    """
    jira: Jira_Config
    source: Source
