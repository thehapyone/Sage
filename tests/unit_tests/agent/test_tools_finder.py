# tests/unit_tests/agent/test_tools_finder.py

import os
import pkgutil
import sys
from unittest.mock import MagicMock, patch

import pytest
from crewai_tools import BaseTool, Tool

# Import the module and classes to be tested
from sage.agent.tools_finder import discover_tools
from sage.utils.exceptions import ToolDiscoveryException


# Sample Tool Classes for Testing
class SampleTool(Tool):
    pass


class AnotherTool(BaseTool):
    pass


class NotATool:
    pass


@pytest.fixture
def mock_crewai_tools_package():
    """
    Fixture to mock the 'crewai_tools' package with sample modules and classes.
    """
    with patch.dict("sys.modules", {}):
        # Mock the 'crewai_tools' package
        crewai_tools = MagicMock()
        crewai_tools.__path__ = ["crewai_tools"]
        crewai_tools.__name__ = "crewai_tools"  # Set __name__
        sys.modules["crewai_tools"] = crewai_tools

        # Mock submodules
        module_one = MagicMock()
        module_one.__name__ = "crewai_tools.module_one"  # Set __name__
        setattr(module_one, "SampleTool", SampleTool)
        setattr(module_one, "NotATool", NotATool)
        sys.modules["crewai_tools.module_one"] = module_one

        module_two = MagicMock()
        module_two.__name__ = "crewai_tools.module_two"  # Set __name__
        setattr(module_two, "AnotherTool", AnotherTool)
        sys.modules["crewai_tools.module_two"] = module_two

        # Mock pkgutil.walk_packages to iterate over the mocked modules
        with patch("pkgutil.walk_packages") as mock_walk:
            mock_walk.return_value = [
                (None, "crewai_tools.module_one", False),
                (None, "crewai_tools.module_two", False),
            ]
            yield


@pytest.fixture
def mock_additional_path(tmp_path):
    """
    Fixture to create a temporary directory with user-defined tool modules.
    """
    user_tools_dir = tmp_path / "user_tools"
    user_tools_dir.mkdir()

    # Create a valid tool module
    tool_module = user_tools_dir / "user_tool.py"
    tool_module.write_text(
        """
from crewai_tools import Tool

class UserDefinedTool(Tool):
    pass

class NotATool:
    pass
"""
    )

    # Create an invalid tool module
    invalid_module = user_tools_dir / "invalid_tool.py"
    invalid_module.write_text(
        """
# This module does not define any Tool subclasses
class SomeClass:
    pass
"""
    )

    return str(user_tools_dir)


def test_discover_tools_from_crewai_tools(mock_crewai_tools_package):
    tools = discover_tools()
    assert "SampleTool" in tools
    assert "AnotherTool" in tools
    assert not any(tool_name == "NotATool" for tool_name in tools)


def test_discover_tools_from_additional_paths(mock_additional_path):
    tools = discover_tools(additional_paths=[mock_additional_path])
    assert "UserDefinedTool" in tools
    assert not any(tool_name == "NotATool" for tool_name in tools)
    assert not any(tool_name == "SomeClass" for tool_name in tools)


def test_discover_tools_with_import_error_in_package(mock_crewai_tools_package):
    def import_module_side_effect(name):
        if name == "crewai_tools.module_two":
            raise ImportError("Failed to import")
        return sys.modules.get(name)

    with patch("importlib.import_module", side_effect=import_module_side_effect):
        with pytest.raises(ToolDiscoveryException) as exc_info:
            discover_tools()
        assert (
            "Error importing module 'crewai_tools.module_two': Failed to import"
            in str(exc_info.value)
        )


def test_discover_tools_with_invalid_additional_paths(
    mock_crewai_tools_package, mock_additional_path, caplog
):
    invalid_path = "/invalid/path"
    with patch("os.path.isdir", side_effect=lambda x: x == mock_additional_path):
        with patch("pkgutil.iter_modules") as mock_iter_modules:
            # Mock iter_modules for the valid path
            def iter_modules_side_effect(paths):
                if paths == [mock_additional_path]:
                    return [
                        (0, "user_tools.user_tool", False),
                        (0, "user_tools.invalid_tool", False),
                    ]
                return []

            mock_iter_modules.side_effect = iter_modules_side_effect

            discover_tools(additional_paths=[invalid_path, mock_additional_path])
            # Should log a warning for the invalid path and still discover from the valid path
            assert (
                f"Tool path '{invalid_path}' is not a directory. Skipping."
                in caplog.text
            )


def test_discover_tools_no_tools_found():
    with patch("pkgutil.walk_packages", return_value=[]):
        with patch("os.path.isdir", return_value=False):
            with pytest.raises(ToolDiscoveryException) as exc_info:
                discover_tools(additional_paths=[])
            assert "No tools were discovered." in str(exc_info.value)


def test_discover_tools_with_invalid_tool_classes(mock_crewai_tools_package):
    # Add a module with classes that do not inherit from Tool or BaseTool
    module_invalid = MagicMock()
    module_invalid.__name__ = "crewai_tools.module_invalid"  # Set __name__
    setattr(module_invalid, "SomeClass", NotATool)
    sys.modules["crewai_tools.module_invalid"] = module_invalid

    with patch("pkgutil.walk_packages") as mock_walk:
        mock_walk.return_value = [
            (None, "crewai_tools.module_one", False),
            (None, "crewai_tools.module_two", False),
            (None, "crewai_tools.module_invalid", False),
        ]
        tools = discover_tools()
        assert "SampleTool" in tools
        assert "AnotherTool" in tools
        assert "SomeClass" not in tools


def test_discover_tools_handles_sys_path_cleanly(
    mock_crewai_tools_package, mock_additional_path
):
    original_sys_path = sys.path.copy()
    try:
        with patch("pkgutil.walk_packages") as mock_walk:
            # Mock walking the additional path
            mock_walk.side_effect = lambda path, prefix: pkgutil.iter_modules(
                [mock_additional_path], prefix="user_tools."
            )

            # Mock importlib.import_module for additional modules
            def import_module_side_effect(name):
                if name == "user_tools.user_tool":
                    user_tool_module = MagicMock()
                    user_tool_module.__name__ = "user_tools.user_tool"
                    setattr(user_tool_module, "UserDefinedTool", MagicMock(spec=Tool))
                    setattr(user_tool_module, "NotATool", NotATool)
                    sys.modules["user_tools.user_tool"] = user_tool_module
                    return user_tool_module
                elif name == "user_tools.invalid_tool":
                    invalid_tool_module = MagicMock()
                    invalid_tool_module.__name__ = "user_tools.invalid_tool"
                    setattr(invalid_tool_module, "SomeClass", NotATool)
                    sys.modules["user_tools.invalid_tool"] = invalid_tool_module
                    return invalid_tool_module
                else:
                    return sys.modules.get(name)

            with patch(
                "importlib.import_module", side_effect=import_module_side_effect
            ):
                discover_tools(additional_paths=[mock_additional_path])
                # Ensure sys.path is restored
                assert sys.path == original_sys_path
    finally:
        sys.path = original_sys_path


def test_discover_tools_with_duplicate_tool_names(
    mock_crewai_tools_package, mock_additional_path
):
    # Create a duplicate tool in additional paths
    duplicate_tool_path = mock_additional_path
    # Create a duplicate_tool.py
    with open(os.path.join(duplicate_tool_path, "duplicate_tool.py"), "w") as f:
        f.write(
            """
from crewai_tools import Tool

class SampleTool(Tool):
    pass
"""
        )
    with patch("pkgutil.walk_packages") as mock_walk:
        mock_walk.return_value = [
            (None, "crewai_tools.module_one", False),
            (None, "crewai_tools.module_two", False),
            (None, "user_tools.duplicate_tool", False),
        ]

        # Mock import_module to handle 'user_tools.duplicate_tool'
        def import_module_side_effect(name):
            if name == "user_tools.duplicate_tool":
                duplicate_module = MagicMock()
                duplicate_module.__name__ = "user_tools.duplicate_tool"
                setattr(duplicate_module, "SampleTool", SampleTool)
                sys.modules["user_tools.duplicate_tool"] = duplicate_module
                return duplicate_module
            elif name == "crewai_tools.module_one":
                return sys.modules["crewai_tools.module_one"]
            elif name == "crewai_tools.module_two":
                return sys.modules["crewai_tools.module_two"]
            else:
                return sys.modules.get(name)

        with patch("importlib.import_module", side_effect=import_module_side_effect):
            tools = discover_tools(additional_paths=[duplicate_tool_path])
            # The duplicate should overwrite the original 'SampleTool'
            assert "SampleTool" in tools
            # Ensure that 'SampleTool' refers to the duplicated class
            assert (
                tools["SampleTool"] is SampleTool
            )  # Since both are the same class in this test


def test_discover_tools_with_empty_additional_paths(mock_crewai_tools_package):
    tools = discover_tools(additional_paths=[])
    assert "SampleTool" in tools
    assert "AnotherTool" in tools
    assert "UserDefinedTool" not in tools  # Since no additional paths provided


def test_discover_tools_with_none_additional_paths(mock_crewai_tools_package):
    tools = discover_tools(additional_paths=None)
    assert "SampleTool" in tools
    assert "AnotherTool" in tools
    assert "UserDefinedTool" not in tools  # Since no additional paths provided
