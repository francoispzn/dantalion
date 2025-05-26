"""Exception hierarchy for Dantalion.

Everything raised on purpose inherits from :class:`DantalionError`, so callers
can catch the whole family with a single ``except`` while still being able to
react to specific failures (a budget overrun is very different from a provider
timeout). Nothing here carries behaviour beyond a message and, where useful, a
little structured context.
"""

from __future__ import annotations


class DantalionError(Exception):
    """Base class for every error raised deliberately by this package."""


class ConfigError(DantalionError):
    """Configuration is missing, contradictory, or otherwise unusable."""


class ProviderError(DantalionError):
    """A model provider failed to produce a usable response."""


class ProviderTimeout(ProviderError):
    """The provider did not respond within the configured deadline."""


class CapabilityError(DantalionError):
    """A capability was requested that the selected model cannot satisfy."""


class StructuredOutputError(DantalionError):
    """The model could not be coaxed into the requested structured shape."""


class SchemaValidationError(StructuredOutputError):
    """Decoded JSON did not validate against the target schema.

    The raw text is kept around so callers (and traces) can see exactly what
    the model produced before it was rejected.
    """

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class ToolError(DantalionError):
    """Base class for tool-related failures."""


class ToolNotFound(ToolError):
    """A tool was requested by name but is not registered."""


class ToolValidationError(ToolError):
    """Arguments supplied for a tool call did not match its schema."""


class ToolExecutionError(ToolError):
    """A tool raised while running. The original cause is chained."""


class BudgetExceeded(DantalionError):
    """A run exhausted its token, time, or step budget."""


class Cancelled(DantalionError):
    """A run was cancelled cooperatively before it finished."""


class AgentError(DantalionError):
    """The agent loop reached a state it could not recover from."""
