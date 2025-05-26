"""Dantalion: a model-agnostic, local-first autonomous investigation agent.

The public surface is intentionally small. Most users either drive the agent
through the command line (``dantalion ...``) or assemble the pieces directly:

    from dantalion.providers import build_provider
    from dantalion.agent import Agent

See ``DESIGN.md`` for the architecture and the rationale behind it.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
