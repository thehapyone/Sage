# test_validator.py

import base64
from logging import getLevelName
from pathlib import Path
import os

import pytest
from pydantic import SecretStr, ValidationError

from sage.utils.exceptions import ConfigException
from sage.validators.config_toml import (
    CohereReRanker,
    Config,
    ConfluenceModel,
    Core,
    EmbeddingsConfig,
    Files,
    GitlabModel,
    HuggingFaceReRanker,
    Jira_Config,
    LLMConfig,
    ModelValidateType,
    Password,
    ReRankerConfig,
    Source,
    SourceData,
    UploadConfig,
    Web,
)


################## Unit Tests for the UploadConfig ########################
def test_upload_config_default_values():
    """
    Test that the UploadConfig class correctly uses default values.
    """
    config = UploadConfig()
    assert config.max_size_mb == 10
    assert config.max_files == 5
    assert config.timeout == 300


def test_upload_config_custom_values():
    """
    Test that the UploadConfig class correctly accepts and sets custom values.
    """
    config = UploadConfig(max_size_mb=20, max_files=10, timeout=600)
    assert config.max_size_mb == 20
    assert config.max_files == 10
    assert config.timeout == 600


@pytest.mark.parametrize(
    "field,invalid_values",
    [
        ("max_size_mb", [0, -10, -100]),
        ("max_files", [0, -10, -100]),
        ("timeout", [0, -10, -100]),
    ],
)
def test_upload_config_non_positive_values(field, invalid_values):
    """
    Test that the UploadConfig class raises a validation error for non positive values.
    """
    for invalid_value in invalid_values:
        with pytest.raises(ValidationError):
            UploadConfig(**{field: invalid_value})


############################################################################
###################### Unit Tests for the Password #########################


def test_password_is_optional():
    """Password should be optional"""
    model = Password()
    assert model.password is None


def test_password_secret_str_parsed():
    """Password should be parsed via SecretStr"""
    secret_password = SecretStr("my_secret_password")
    model = Password(password=secret_password)
    assert model.password == secret_password


def test_password_dump_secret():
    """Password dump_secret should return secret value"""
    secret_password = SecretStr("my_secret_password")
    model = Password(password=secret_password)
    assert model.dump_secret(model.password) == "my_secret_password"


def test_dump_secret_none():
    """Password dump_secret should handle None password"""
    model = Password()
    assert model.dump_secret(None) is None


def test_password_available_only_via_secret_parser():
    """Password can be fetch via secret parser function"""
    secret_password = SecretStr("my_secret_password")
    model = Password(password=secret_password)
    assert model.password != "my_secret_password"
    assert model.password.get_secret_value() == "my_secret_password"


##############################################################################
##################### Unit Tests for the ModelValidateType ###################


def test_llm_embeddings_validate_type_config_available():
    class TestModel(ModelValidateType):
        type: str
        some_type: dict

    values = {"type": "some_type", "some_type": {"key": "value"}}
    validated_values = TestModel.validate_config_is_available_for_type(values)
    assert validated_values == values


def test_llm_embeddings_validate_type_config_missing_raises_exception():
    class TestModel(ModelValidateType):
        type: str
        some_type: dict

    values = {"type": "some_type"}
    with pytest.raises(ConfigException) as exc_info:
        TestModel.validate_config_is_available_for_type(values)
    assert "The Config data for type 'some_type' is missing." in str(exc_info.value)


##############################################################################
##################### Unit Tests for the SourceData ##########################


def test_source_data_add_https_to_server():
    source_data = SourceData(username="user", server="example.com")
    assert source_data.server == "https://example.com"

    # Test username is expected value
    assert source_data.username == "user"


def test_source_data_server_already_has_https():
    source_data = SourceData(username="user", server="https://example.com")
    assert source_data.server == "https://example.com"


##############################################################################
##################### Unit Tests for the ConfluenceModel #####################


def test_confluence_model_password_validation(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_PASSWORD", "test_confluence_password")
    config = ConfluenceModel(username="user", server="example.com", spaces=["DEV"])
    assert config.password.get_secret_value() == "test_confluence_password"

    # Test username is expected value
    assert config.username == "user"

    monkeypatch.delenv("CONFLUENCE_PASSWORD", raising=False)
    with pytest.raises(ConfigException) as exc_info:
        ConfluenceModel(username="user", server="example.com", spaces=["DEV"])
    assert (
        "The CONFLUENCE_PASSWORD | JIRA_PASSWORD | config password is missing"
        in str(exc_info.value)
    )


################################################################################
##################### Unit Tests for the GitlabModel ##########################


def test_gitlab_model_validate_projects_groups(monkeypatch):
    # Test the group value is the expected value
    monkeypatch.setenv("GITLAB_PASSWORD", "test_gitlab_password")
    config = GitlabModel(server="gitlab.com", groups=["group1"])
    assert config.groups == ["group1"]
    assert config.projects == []

    # Test the project value is the expected value
    config = GitlabModel(server="gitlab.com", projects=["project1"])
    assert config.projects == ["project1"]
    assert config.groups == []

    # Test an error is raised when no groups or projects is added
    with pytest.raises(ConfigException) as exc_info:
        GitlabModel(server="gitlab.com")
    assert (
        "The Gitlab projects or groups are missing from the configuration file"
        in str(exc_info.value)
    )


def test_gitlab_model_password_validation(monkeypatch):
    monkeypatch.setenv("GITLAB_PASSWORD", "test_gitlab_password")
    config = GitlabModel(server="gitlab.com", groups=["group1"])
    assert config.password.get_secret_value() == "test_gitlab_password"

    monkeypatch.delenv("GITLAB_PASSWORD", raising=False)
    with pytest.raises(ConfigException) as exc_info:
        GitlabModel(server="gitlab.com", groups=["group1"])
    assert "The GITLAB_PASSWORD | config password is missing." in str(exc_info.value)


################################################################################
##################### Unit Tests for the Web Model ############################


def test_web_model_creation_with_credentials():
    username = "user"
    password = "pass"
    encoded_credentials = base64.b64encode(
        f"{username}:{password}".encode("utf-8")
    ).decode("utf-8")
    web = Web(
        links=["https://example.com"],
        nested=True,
        username=username,
        password=SecretStr(password),
    )
    assert web.links == ["https://example.com"]
    assert web.nested is True
    assert web.ssl_verify is True
    assert web.headers == {"Authorization": f"Basic {encoded_credentials}"}


def test_web_model_creation_without_credentials():
    web = Web(links=["https://example.com"], nested=True)
    assert web.links == ["https://example.com"]
    assert web.nested is True
    assert web.ssl_verify is True
    assert web.headers == {}
    assert web.username is None
    assert web.password is None


def test_web_model_creation_with_username_only():
    with pytest.raises(ConfigException) as excinfo:
        Web(links=["https://example.com"], nested=True, username="user")
    assert "Both a Username and Password are required for the Web Source" in str(
        excinfo.value
    )


def test_web_model_creation_with_password_only():
    with pytest.raises(ConfigException) as excinfo:
        Web(links=["https://example.com"], nested=True, password=SecretStr("pass"))
    assert "Both a Username and Password are required for the Web Source" in str(
        excinfo.value
    )


def test_web_model_ssl_verify():
    web = Web(links=["https://example.com"], nested=False, ssl_verify=False)
    assert web.ssl_verify is False


################################################################################
##################### Unit Tests for the Files Model ###########################


def test_files_model_creation():
    files = Files(paths=["/path/to/file1", "/path/to/file2"])
    assert files.paths == ["/path/to/file1", "/path/to/file2"]


################################################################################
##################### Unit Tests for the Source Model ##########################


def test_source_model_default_top_k():
    source = Source()
    assert source.top_k == 20


def test_source_model_custom_top_k():
    source = Source(top_k=10)
    assert source.top_k == 10


def test_source_model_optional_fields(monkeypatch):
    monkeypatch.setenv("GITLAB_PASSWORD", "gitlab_secret")
    monkeypatch.setenv("JIRA_PASSWORD", "jira_secret")
    source = Source(
        top_k=5,
        confluence=ConfluenceModel(
            username="user", server="https://example.com", spaces=["SPACE"]
        ),
        gitlab=GitlabModel(
            username="user", server="https://example.com", projects=["project1"]
        ),
        web=Web(links=["https://example.com"], nested=True),
        files=Files(paths=["/path/to/file1"]),
        refresh_schedule="1 * * * *",
    )
    assert source.confluence is not None
    assert source.gitlab is not None
    assert source.web is not None
    assert source.files is not None
    assert source.refresh_schedule == "1 * * * *"


def test_source_model_missing_optional_fields():
    source = Source(top_k=5)
    assert source.confluence is None
    assert source.gitlab is None
    assert source.web is None
    assert source.files is None
    assert source.refresh_schedule is None


def test_source_model_invalid_cron_syntax():
    with pytest.raises(ValueError) as excinfo:
        Source(refresh_schedule="invalid_cron_syntax")
    assert "The value of refresh_schedule is not a valid cron syntax" in str(
        excinfo.value
    )


################################################################################
##################### Unit Tests for the Jira_Config ###########################


def test_jira_config_password_validation(monkeypatch):
    monkeypatch.setenv("JIRA_PASSWORD", "test_jira_password")
    config = Jira_Config(
        url="https://jira.example.com",
        username="user",
        polling_interval=30,
        project="TEST",
        status_todo="To Do",
    )
    assert config.password.get_secret_value() == "test_jira_password"

    monkeypatch.delenv("JIRA_PASSWORD", raising=False)
    with pytest.raises(ConfigException) as exc_info:
        Jira_Config(
            url="https://jira.example.com",
            username="user",
            polling_interval=30,
            project="TEST",
            status_todo="To Do",
        )
    assert "The JIRA_PASSWORD | config password is missing." in str(exc_info.value)


def test_jira_config_model_creation(monkeypatch):
    monkeypatch.setenv("JIRA_PASSWORD", "test_jira_password")
    config = Jira_Config(
        url="https://jira.example.com",
        username="user",
        polling_interval=30,
        project="TEST",
        status_todo="To Do",
    )
    assert config.url == "https://jira.example.com"
    assert config.username == "user"
    assert config.polling_interval == 30
    assert config.project == "TEST"
    assert config.status_todo == "To Do"


###############################################################################
##################### Unit Tests for the Core #################################


def test_core_default_values():
    core = Core()
    assert core.data_dir == Path.home() / ".sage"
    assert core.logging_level == "INFO"
    assert core.user_agent == "codesage.ai"
    assert core.starters_path is None
    assert core.agents_dir is None
    assert core.disable_crewai_telemetry == True


def test_core_custom_values():
    custom_data_dir = "/custom/path"
    core = Core(
        data_dir=custom_data_dir,
        logging_level="DEBUG",
        user_agent="custom_agent",
        starters_path="/fake/starters.yaml",
        agents_dir="/fake/agents",
    )
    assert str(core.data_dir) == custom_data_dir
    assert core.logging_level == getLevelName("DEBUG")
    assert core.user_agent == "custom_agent"
    assert core.starters_path == "/fake/starters.yaml"
    assert core.agents_dir == "/fake/agents"


def test_core_logging_level_validation():
    core = Core(logging_level="WARNING")
    assert core.logging_level == getLevelName("WARNING")

    # Test invalid level
    core = Core(logging_level="INVALID_LEVEL")
    assert core.logging_level == "Level INVALID_LEVEL"


def test_core_crewai_telemetry():
    # Test OTEL_SDK_DISABLED is disabled by default
    os.environ.pop("OTEL_SDK_DISABLED", None)
    core = Core()
    assert os.environ.get("OTEL_SDK_DISABLED") == "true"

    # Test OTEL_SDK_DISABLED is enabled
    os.environ.pop("OTEL_SDK_DISABLED", None)
    core = Core(disable_crewai_telemetry=False)
    assert os.environ.get("OTEL_SDK_DISABLED") == "false"
    os.environ.pop("OTEL_SDK_DISABLED", None)


###############################################################################
##################### Unit Tests for the EmbeddingsConfig #####################


def test_embedding_config_seting_values():
    embeddings_config = EmbeddingsConfig(
        type="litellm", model="text-embedding-ada-002", dimension=1024
    )
    assert embeddings_config.type == "litellm"
    assert embeddings_config.model == "text-embedding-ada-002"
    assert embeddings_config.dimension == 1024


def test_embedding_config_no_default_values():
    with pytest.raises(ValidationError) as _:
        EmbeddingsConfig(type="huggingface")


###############################################################################
##################### Unit Tests for the CohereReRanker #######################


def test_cohere_reranker_password_validation(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_password")
    reranker = CohereReRanker(name="cohere_reranker")
    assert reranker.password.get_secret_value() == "test_cohere_password"

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ConfigException) as exc_info:
        CohereReRanker(name="cohere_reranker")
    assert "The COHERE_API_KEY | config password is missing. " in str(exc_info.value)


###############################################################################
##################### Unit Tests for the HuggingFaceReRanker ##################


def test_huggingface_reranker_creation():
    reranker = HuggingFaceReRanker(name="hf_reranker", revision="v1")
    assert reranker.name == "hf_reranker"
    assert reranker.revision == "v1"


###############################################################################
##################### Unit Tests for the ReRankerConfig #######################


def test_reranker_config_default_top_n(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_password")
    reranker_config = ReRankerConfig(
        type="cohere", cohere=CohereReRanker(name="cohere_reranker")
    )
    assert reranker_config.top_n == 5


def test_reranker_config_custom_top_n(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_password")
    reranker_config = ReRankerConfig(
        type="cohere", cohere=CohereReRanker(name="cohere_reranker"), top_n=10
    )
    assert reranker_config.top_n == 10


def test_reranker_config_type_validation(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_password")
    # Test that the type field accepts valid literals
    reranker_config_cohere = ReRankerConfig(
        type="cohere", cohere=CohereReRanker(name="cohere_reranker")
    )
    assert reranker_config_cohere.type == "cohere"

    reranker_config_hf = ReRankerConfig(
        type="huggingface",
        huggingface=HuggingFaceReRanker(name="hf_reranker", revision="v1"),
    )
    assert reranker_config_hf.type == "huggingface"

    # Test that an invalid type raises a ValueError
    with pytest.raises(ConfigException):
        ReRankerConfig(type="invalid_type")


def test_reranker_config_provided_model_validation(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_password")
    # Test that the correct reranker model is provided based on the type
    reranker_config_cohere = ReRankerConfig(
        type="cohere", cohere=CohereReRanker(name="cohere_reranker")
    )
    assert reranker_config_cohere.cohere is not None
    assert reranker_config_cohere.huggingface is None

    reranker_config_hf = ReRankerConfig(
        type="huggingface",
        huggingface=HuggingFaceReRanker(name="hf_reranker", revision="v1"),
    )
    assert reranker_config_hf.huggingface is not None
    assert reranker_config_hf.cohere is None


def test_reranker_config_missing_model_raises_exception():
    # Test that a missing reranker model raises a ConfigException
    with pytest.raises(ConfigException) as exc_info:
        ReRankerConfig(type="cohere")
    assert "The Config data for type 'cohere' is missing." in str(exc_info.value)

    with pytest.raises(ConfigException) as exc_info:
        ReRankerConfig(type="huggingface")
    assert "The Config data for type 'huggingface' is missing." in str(exc_info.value)


###################################################################################
######################### Unit Tests for the LLMConfig ############################


def test_llm_model_validation():
    llm_config = LLMConfig(model="azure/gpt4-128k")
    assert llm_config.model == "azure/gpt4-128k"


def test_llm_model_value_required():
    with pytest.raises(ValidationError):
        LLMConfig()


###########################################################################
######################## Unit Tests for the Config ########################


@pytest.fixture
def setup_env_vars(monkeypatch):
    # Set up environment variables for all tests
    monkeypatch.setenv("AZURE_PASSWORD", "azure_secret")
    monkeypatch.setenv("OPENAI_PASSWORD", "openai_secret")
    monkeypatch.setenv("GITLAB_PASSWORD", "gitlab_secret")
    monkeypatch.setenv("JIRA_PASSWORD", "jira_secret")
    monkeypatch.setenv("COHERE_API_KEY", "cohere_secret")

    yield


def test_config_creation_with_all_fields(setup_env_vars):
    config = Config(
        core=Core(),
        upload=UploadConfig(),
        jira=Jira_Config(
            url="https://jira.example.com",
            username="user",
            polling_interval=30,
            project="TEST",
            status_todo="To Do",
        ),
        source=Source(top_k=10),
        reranker=ReRankerConfig(
            type="cohere", cohere=CohereReRanker(name="cohere_reranker")
        ),
        embedding={
            "type": "litellm",
            "model": "openai/ada_embedding",
        },
        llm={
            "model": "gpt3.5",
        },
    )
    assert config.core is not None
    assert config.upload is not None
    assert config.jira is not None
    assert config.source is not None
    assert config.reranker is not None
    assert config.embedding is not None
    assert config.llm is not None


def test_config_default_optional_fields(setup_env_vars):
    config = Config(
        jira=Jira_Config(
            url="https://jira.example.com",
            username="user",
            polling_interval=30,
            project="TEST",
            status_todo="To Do",
        ),
        source=Source(top_k=10),
        embedding={
            "type": "huggingface",
            "model": "jina/model",
        },
        llm={
            "model": "gpt3.5",
        },
    )
    assert isinstance(config.core, Core)
    assert isinstance(config.upload, UploadConfig)
    assert config.reranker is None
