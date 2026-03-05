# Data Sources Overview for Sage

Sage can connect to various data sources to retrieve and manipulate data. This document outlines the data sources currently supported and how to configure them.

## Sources

The Sources module is the entry point for configuring access to external data sources, allowing CodeSage to interact with and index data.

Supported sources include:

- Confluence
- GitLab
- Websites (via nested web crawling functionality)
- Local files

### Confluence

CodeSage can index content from Atlassian Confluence spaces or individual pages.

### GitLab

CodeSage can connect to GitLab to index content from specified projects or groups.

### Websites / Web Links

CodeSage can crawl and index content from external web links, including nested page structures.

### Local files

CodeSage can crawl and index content from files on the host. E.g, Index all the files in your docs directory.

## Configuring Sources

To configure a source in Sage:

1. Add the data source details to your `config.toml` file, including connection strings, authentication credentials, and any other necessary parameters.
2. Restart Sage to apply the new configuration.

## Using Sources

Once configured, you can ask Sage to perform actions related to the data source, such as:

- Querying a database for specific records.
- Fetching data from an API endpoint.
- Reading or writing files to a file system.
