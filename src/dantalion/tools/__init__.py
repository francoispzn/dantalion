"""The tool framework: typed callables the agent can invoke."""

from __future__ import annotations

from dantalion.tools.base import Tool, ToolResult
from dantalion.tools.decorator import function_to_tool, tool
from dantalion.tools.registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "function_to_tool",
    "tool",
]
