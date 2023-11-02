from pydantic import BaseModel, validator
from typing import Optional, List
from pathlib import Path
import os
from utils.exceptions import ConfigException

sage_base = ".sage"


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


class Confluence(SourceData):
    """  
    Confluence Data Model. Inherits SourceData.  
    """
    @validator('password', pre=True, always=True)
    def set_password(cls, v):
        password = v or os.getenv(
            'CONFLUENCE_PASSWORD') or os.getenv('JIRA_PASSWORD')
        if password is None:
            raise ConfigException(
                "The Confluence password is missing. \
                    Please add it via an env variable or to the config - 'CONFLUENCE_PASSWORD'")
        return password


class Gitlab(SourceData):
    """  
    Gitlab Data Model. Inherits SourceData.  
    """
    @validator('password', pre=True, always=True)
    def set_password(cls, v):
        password = v or os.getenv('GITLAB_PASSWORD')
        if password is None:
            raise ConfigException(
                "The Gitlab password is missing. \
                    Please add it via an env variable or to the config - 'GITLAB_PASSWORD'")
        return password


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
    confluence: Optional[Confluence] = None
    gitlab: Optional[Gitlab] = None
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

    @validator('password', pre=True, always=True)
    def set_password(cls, v):
        password = v or os.getenv('JIRA_PASSWORD')
        if password is None:
            raise ConfigException(
                "The JIRA password is missing. \
                    Please add it via an env variable or to the config - 'JIRA_PASSWORD'")
        return password


class Core(BaseModel):
    """
    Core Model.
    """
    data_dir: Optional[str | Path] = Path.home() / sage_base
    sources_dir: Optional[str | Path] = Path(data_dir) / "sources"


class Config(BaseModel):
    """  
    Config Model.  
    """
    core: Core
    jira: Jira_Config
    source: Source
