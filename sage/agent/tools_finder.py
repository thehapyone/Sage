import importlib
import pkgutil
import inspect
from typing import Dict, Type, List
import sys
import os
from crewai_tools import Tool, BaseTool
from sage.utils.exceptions import ToolDiscoveryException
from sage.utils.logger import CustomLogger


def discover_tools(additional_paths: List[str] = None) -> Dict[str, Type[Tool]]:
    """
    Dynamically discover and load tool classes from specified packages and directories.

    Args:
        additional_paths (List[str], optional): List of directory paths to scan for user-defined tools.

    Returns:
        Dict[str, Type[Tool]]: Mapping from tool name to tool class.
    """
    tools = {}

    packages = ["crewai_tools"]

    logger = CustomLogger()

    # Scan specified packages
    for package in packages:
        try:
            pkg = importlib.import_module(package)
        except ImportError as e:
            raise ToolDiscoveryException(f"Error importing package '{package}': {e}")

        for finder, name, ispkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            try:
                module = importlib.import_module(name)
            except ImportError as e:
                raise ToolDiscoveryException(f"Error importing module '{name}': {e}")

            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if (
                    inspect.isclass(attribute)
                    and attribute not in (Tool, BaseTool)
                    and (issubclass(attribute, Tool) or issubclass(attribute, BaseTool))
                ):
                    tool_name = attribute.__name__
                    tools[tool_name] = attribute

    # Scan additional directories for user-defined tools
    if additional_paths:
        for path in additional_paths:
            if not os.path.isdir(path):
                logger.warning(f"Tool path '{path}' is not a directory. Skipping.")
                continue

            # Add the path to sys.path to allow module imports
            sys.path.insert(0, path)
            for finder, name, ispkg in pkgutil.iter_modules([path]):
                try:
                    module = importlib.import_module(name)
                except ImportError as e:
                    logger.warning(
                        f"Error importing module '{name}' from '{path}': {e}"
                    )
                    continue

                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if (
                        inspect.isclass(attribute)
                        and attribute not in (Tool, BaseTool)
                        and (
                            issubclass(attribute, Tool)
                            or issubclass(attribute, BaseTool)
                        )
                    ):
                        tool_name = attribute.__name__
                        tools[tool_name] = attribute
            # Remove the path from sys.path to prevent side effects
            sys.path.pop(0)

    if not tools:
        raise ToolDiscoveryException(
            "No tools were discovered. Ensure tools inherit from 'Tool' and are in the scanned packages or directories."
        )

    return tools
