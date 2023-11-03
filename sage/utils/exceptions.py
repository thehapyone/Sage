"""
Custom exception modules.

Classes:  
    ConfigException: An exception class for handling schema/config validation exceptions or error
    SourceException: An exception class for handling apps related validation errors
    TransformerException: An exception class for handling trasnsformer related validation errors
"""


class ConfigException(Exception):
    """
    An exception for configuration related errors
    """
    pass

class SourceException(Exception):
    """
    An exception handler for sources
    """