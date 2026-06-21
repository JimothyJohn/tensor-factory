"""FastMCP server exposing a tensor-factory detector as MCP tools (stdio transport).

Tools are thin wrappers over :mod:`tensor_factory_mcp.core`; all are read-only, local, and
return JSON strings. Run with ``tensor-factory-mcp`` (stdio) or register in an MCP client.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, core

mcp = FastMCP("tensor_factory_mcp")


def _read_only(title: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=title,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


class _ModelParams(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    model_path: str | None = Field(
        default=None,
        description="Path to an ONNX model. Omit to use the bundled demo model.",
    )
    input_size: int = Field(
        default=core.DEFAULT_INPUT_SIZE,
        description="Square model input size in pixels.",
        ge=16,
        le=4096,
    )


class DetectInput(_ModelParams):
    image_path: str = Field(
        ...,
        description="Path to the image to run detection on (e.g. '/data/frame_001.png').",
        min_length=1,
    )


class BenchmarkInput(_ModelParams):
    n: int = Field(
        default=100, description="Number of inference iterations to time.", ge=1, le=10000
    )


def _ok(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2)


def _err(exc: Exception) -> str:
    return f"Error: {type(exc).__name__}: {exc}"


@mcp.tool(name="tensor_factory_detect", annotations=_read_only("Detect"))
def tensor_factory_detect(params: DetectInput) -> str:
    """Detect the target object in a single image and return its bounding box.

    Args:
        params (DetectInput):
            - image_path (str): image file to analyze.
            - model_path (str | None): ONNX model; default is the bundled demo model.
            - input_size (int): square model input size (default 480).

    Returns:
        str: JSON with schema:
        {
          "box_norm":   {"x1": float, "y1": float, "x2": float, "y2": float},  # [0,1] xyxy
          "box_pixels": {"x1": int, "y1": int, "x2": int, "y2": int},          # absolute
          "uint8":      [int, int, int, int],                                   # 8-bit xyxy
          "image_size": {"width": int, "height": int},
          "model":      str
        }
        On failure: "Error: <type>: <message>".
    """
    try:
        return _ok(core.detect(params.image_path, params.model_path, params.input_size))
    except Exception as exc:  # noqa: BLE001 -- surfaced as an actionable tool message
        return _err(exc)


@mcp.tool(name="tensor_factory_model_info", annotations=_read_only("Model Info"))
def tensor_factory_model_info(params: _ModelParams) -> str:
    """Report the resolved model path, input size, IO name, and ORT providers.

    Returns:
        str: JSON {"model": str, "input_size": int, "input_name": str, "providers": [str]}
        or "Error: <type>: <message>".
    """
    try:
        return _ok(core.model_info(params.model_path, params.input_size))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool(
    name="tensor_factory_benchmark",
    annotations=_read_only("Benchmark"),
)
def tensor_factory_benchmark(params: BenchmarkInput) -> str:
    """Measure CPU inference throughput (fps) on a synthetic frame.

    Returns:
        str: JSON {"model": str, "fps": float, "iterations": int, "input_size": int}
        or "Error: <type>: <message>".
    """
    try:
        return _ok(core.benchmark(params.model_path, params.input_size, params.n))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tensor-factory-mcp", description="MCP server for tensor-factory detection (stdio)."
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
