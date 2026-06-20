"""helicoils-mcp -- expose the helicoil detector over the Model Context Protocol.

A thin stdio MCP server wrapping :mod:`helicoils.inference`. The detection logic lives
in :mod:`helicoils_mcp.core` (importable and testable on its own); :mod:`helicoils_mcp.server`
registers the FastMCP tools around it.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
