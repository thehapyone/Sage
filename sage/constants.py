import os
import toml
from utils.exceptions import ConfigException

# Load the configuration file only once
config_path = os.getenv("SAGE_CONFIG_PATH", "config.toml")

try:
    config = toml.load(config_path)
    jira_config = config["JIRA"]
except (FileNotFoundError, KeyError) as error:
    # Raise a ConfigException for both file not found and missing key errors
    raise ConfigException(f"The required configuration key or file is not found - {str(error)}")

# Get the JIRA password from environment variables
jira_password = os.getenv("JIRA_PASSWORD")

if not jira_password:
    raise ConfigException(f"The JIRA password is missing. Please add it via an env variable - 'JIRA_PASSWORD'")

jira_config["password"] = jira_password