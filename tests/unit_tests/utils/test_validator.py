# test_validator.py

import base64
import os
from logging import getLevelName
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from sage.utils.exceptions import ConfigException
from sage.utils.validator import (
    AzureConfig,
    CohereReRanker,
    Config,
    ConfluenceModel,
    Core,
    EmbeddingCore,
    EmbeddingsConfig,
    Files,
    GitlabModel,
    HuggingFaceReRanker,
    Jira_Config,
    LLMConfig,
    LLMCore,
    ModelValidateType,
    OpenAIConfig,
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


############################################################################
##################### Unit Tests for the AzureConfig #######################


def test_azure_config_creation():
    """Test creating an AzureConfig instance with all required fields"""
    config = AzureConfig(
        endpoint="example.com", revision="1", password="custom_password"
    )
    assert config.endpoint == "https://example.com"
    assert config.revision == "1"
    assert config.password.get_secret_value() == "custom_password"


def test_azure_config_endpoint_validation(monkeypatch):
    """Test that the endpoint is correctly prefixed with 'https://'"""
    monkeypatch.setenv("AZURE_PASSWORD", "test_azure_password")

    config = AzureConfig(endpoint="example.com", revision="1")
    assert config.endpoint == "https://example.com"

    # Test that an already correct endpoint remains unchanged
    config = AzureConfig(endpoint="https://example.com", revision="1")
    assert config.endpoint == "https://example.com"


def test_azure_config_password_validation():
    """Test that the password is set from the constructor"""
    config = AzureConfig(
        endpoint="example.com", revision="1", password="custom_password"
    )
    assert config.password.get_secret_value() == "custom_password"

    # Test that the password is set from the AZURE_PASSWORD environment variable
    os.environ["AZURE_PASSWORD"] = "test_azure_password"
    config = AzureConfig(endpoint="example.com", revision="1")
    assert config.password.get_secret_value() == "test_azure_password"
    del os.environ["AZURE_PASSWORD"]

    # Test that the password is set from the AZURE_OPENAI_API_KEY environment variable
    os.environ["AZURE_OPENAI_API_KEY"] = "test_azure_openai_api_key"
    config = AzureConfig(endpoint="example.com", revision="1")
    assert config.password.get_secret_value() == "test_azure_openai_api_key"
    del os.environ["AZURE_OPENAI_API_KEY"]

    # Test that a missing password raises a ConfigException
    error_mgs = (
        "The AZURE_OPENAI_API_KEY | AZURE_PASSWORD | config password is missing. "
        "Please add it via an env variable or to the config password field."
    )
    with pytest.raises(ConfigException, match=error_mgs) as _:
        AzureConfig(endpoint="example.com", revision="1")


############################################################################
##################### Unit Tests for the OpenAIConfig #######################


def test_openai_config_password_validation(monkeypatch):
    """Test OpenAIConfig password validation"""
    # Test that the password is set from the constructor
    config = OpenAIConfig(password="custom_password")
    assert config.password.get_secret_value() == "custom_password"

    # Test that the password is set from the OPENAI_PASSWORD environment variable
    monkeypatch.setenv("OPENAI_PASSWORD", "test_openai_password")
    config = OpenAIConfig()
    assert config.password.get_secret_value() == "test_openai_password"
    monkeypatch.delenv("OPENAI_PASSWORD", raising=False)

    # Test that the password is set from the OPENAI_API_KEY environment variable
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_api_key")
    config = OpenAIConfig()
    assert config.password.get_secret_value() == "test_openai_api_key"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Test that a missing password raises a ConfigException
    error_mgs = (
        "The OPENAI_API_KEY | OPENAI_PASSWORD | config password is missing. "
        "Please add it via an env variable or to the config password field."
    )
    with pytest.raises(ConfigException, match=error_mgs) as _:
        OpenAIConfig()


def test_openai_config_organization_optional():
    """Test OpenAI config organization is optional"""
    config = OpenAIConfig(password="custom_password")
    assert config.organization is None

    # Test organization value can be set
    config = OpenAIConfig(organization="cool_org", password="custom_password")
    assert config.organization == "cool_org"


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
    )
    assert source.confluence is not None
    assert source.gitlab is not None
    assert source.web is not None
    assert source.files is not None


def test_source_model_missing_optional_fields():
    source = Source(top_k=5)
    assert source.confluence is None
    assert source.gitlab is None
    assert source.web is None
    assert source.files is None


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


def test_core_custom_values():
    custom_data_dir = "/custom/path"
    core = Core(
        data_dir=custom_data_dir, logging_level="DEBUG", user_agent="custom_agent"
    )
    assert core.data_dir == custom_data_dir
    assert core.logging_level == getLevelName("DEBUG")
    assert core.user_agent == "custom_agent"


def test_core_logging_level_validation():
    core = Core(logging_level="WARNING")
    assert core.logging_level == getLevelName("WARNING")

    # Test invalid level
    core = Core(logging_level="INVALID_LEVEL")
    assert core.logging_level == "Level INVALID_LEVEL"


###############################################################################
##################### Unit Tests for the EmbeddingsConfig #####################


def test_embeddings_config_validation():
    embeddings_config = EmbeddingsConfig(
        type="azure", azure=EmbeddingCore(name="azure_embedding")
    )
    assert embeddings_config.type == "azure"
    assert embeddings_config.azure.name == "azure_embedding"

    with pytest.raises(ConfigException) as exc_info:
        EmbeddingsConfig(type="azure")
    assert "The Config data for type 'azure' is missing." in str(exc_info.value)


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


###############################################################################
##################### Unit Tests for the LLMCore ##############################


def test_llm_core_creation():
    llm_core = LLMCore(
        name="llm_service",
        password="secret",
        endpoint="https://llm.example.com",
        revision="v1",
    )
    assert llm_core.name == "llm_service"
    assert llm_core.password.get_secret_value() == "secret"
    assert llm_core.endpoint == "https://llm.example.com"
    assert llm_core.revision == "v1"


def test_llm_core_optional_fields():
    llm_core = LLMCore(name="llm_service")
    assert llm_core.endpoint is None
    assert llm_core.revision is None


###################################################################################
######################### Unit Tests for the LLMConfig ############################


def test_llm_config_type_validation():
    llm_config = LLMConfig(type="openai", openai=LLMCore(name="openai_llm"))
    assert llm_config.type == "openai"
    assert llm_config.openai is not None

    with pytest.raises(ConfigException):
        LLMConfig(type="invalid_type")


def test_llm_config_provided_model_validation():
    llm_config = LLMConfig(type="ollama", ollama=LLMCore(name="ollama_llm"))
    assert llm_config.ollama is not None
    assert llm_config.azure is None
    assert llm_config.openai is None


def test_llm_config_missing_model_raises_exception():
    with pytest.raises(ConfigException) as exc_info:
        LLMConfig(type="azure")
    assert "The Config data for type 'azure' is missing." in str(exc_info.value)


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


def test_config_provider_configs_validation(setup_env_vars):
    # Test that the validator raises an exception when required provider configs are missing
    with pytest.raises(ConfigException) as exc_info:
        Config(
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
            embedding={
                "type": "azure",
                "azure": EmbeddingCore(name="azure_embed", revision="v1"),
            },
            llm={
                "type": "azure",
                "azure": LLMCore(name="azure_llm", password="secret"),
            },
        )
    assert (
        "Azure configuration must be provided when embedding or llm type is 'azure'"
        in str(exc_info.value)
    )


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
        azure=AzureConfig(endpoint="example.com", revision="1"),
        openai=OpenAIConfig(organization="org"),
        source=Source(top_k=10),
        reranker=ReRankerConfig(
            type="cohere", cohere=CohereReRanker(name="cohere_reranker")
        ),
        embedding={
            "type": "azure",
            "azure": EmbeddingCore(name="azure_embed", revision="v1"),
        },
        llm={"type": "azure", "azure": LLMCore(name="azure_llm", password="secret")},
    )
    assert config.core is not None
    assert config.upload is not None
    assert config.jira is not None
    assert config.azure is not None
    assert config.openai is not None
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
        azure=AzureConfig(endpoint="example.com", revision="1"),
        embedding={
            "type": "azure",
            "azure": EmbeddingCore(name="azure_embed", revision="v1"),
        },
        llm={"type": "azure", "azure": LLMCore(name="azure_llm", password="secret")},
    )
    assert isinstance(config.core, Core)
    assert isinstance(config.upload, UploadConfig)
    assert config.openai is None
    assert config.reranker is None
