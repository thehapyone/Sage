# Installation Guide for Sage

## Prerequisites

Before installing Sage, ensure you have the following:

- Docker installed on your system
- An environment file or the necessary environment variables set up

## Installation Steps

1. **Create the container directory**: `mkdir -p codesage`

2. **Set Environment Variables**: Create your `sage.env` file in the root directory and fill in the necessary environment variables: For example:
   ```shell
    AZURE_OPENAI_API_KEY="sample token"
    GITLAB_PASSWORD="sample token"
    CHAINLIT_AUTH_SECRET="sample secret"
    COHERE_API_KEY="sample key"
    JIRA_PASSWORD="sample token"
    SAGE_HOME="path to your docker compose directory"
    HF_TOKEN="Hugging face hub API token"
   ```

3. **Prepare Configuration**: Edit the `config.toml` file to set up your Sage configuration according to your needs.

4. **Run Docker Compose**: Use the provided `docker-compose.yml` file to start Sage: 
    
    `docker-compose --env-file sage.env up -d`

5. **Verify Installation**: Once the container is running, navigate to `http://localhost:8080` in your web browser to access Sage.

## Updating Sage

To update Sage to the latest version, pull the new image and restart the service:

`docker-compose pull docker-compose up -d`
