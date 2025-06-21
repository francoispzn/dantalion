"""A collection of tools the agent is allowed to use.

The registry is the boundary between "every tool that exists" and "the tools
offered for this run". A domain assembles one, the agent reads ``specs`` off it
to tell the model what is available, and looks tools up by name when the model
asks to call one.
"""

from __future__ import annotations

from collections.abc import Iterator

from dantalion.errors import ToolError, ToolNotFound
from dantalion.tools.base import Tool
from dantalion.types import ToolSpec


class ToolRegistry:
    """An ordered, name-indexed set of tools."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for entry in tools or []:
            self.register(entry)

    def register(self, entry: Tool) -> Tool:
        if entry.name in self._tools:
            raise ToolError(f"tool already registered: {entry.name!r}")
        self._tools[entry.name] = entry
        return entry

    def add(self, *tools: Tool) -> None:
        for entry in tools:
            self.register(entry)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(f"no such tool: {name!r}") from None

    def specs(self) -> list[ToolSpec]:
        return [entry.spec() for entry in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
