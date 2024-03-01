# test_validator.py

import pytest
from pydantic import ValidationError, SecretStr
from sage.utils.validator import UploadConfig, Password, AzureConfig
from sage.utils.exceptions import ConfigException
import os


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


def test_azure_config_endpoint_validation():
    """Test that the endpoint is correctly prefixed with 'https://'"""
    config = AzureConfig(endpoint="example.com", revision="1")
    assert config.endpoint == "https://example.com"

    # Test that an already correct endpoint remains unchanged
    config = AzureConfig(endpoint="https://example.com", revision="1")
    assert config.endpoint == "https://example.com"


def test_azure_config_password_validation():
    # Test that the password is set from the constructor
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
    with pytest.raises(ConfigException, match=error_mgs) as exc_info:
        AzureConfig(endpoint="example.com", revision="1")
