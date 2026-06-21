"""tensor-factory-mcp -- expose a tensor-factory detector over the Model Context Protocol.

A thin stdio MCP server wrapping :mod:`tensor_factory.inference`. The detection logic lives
in :mod:`tensor_factory_mcp.core` (importable and testable on its own); the
:mod:`tensor_factory_mcp.server` module registers the FastMCP tools around it.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
