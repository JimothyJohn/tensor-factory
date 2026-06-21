import asyncio
import json

import pytest
from PIL import Image

from tensor_factory_mcp import server


@pytest.mark.unit
def test_server_name():
    assert server.mcp.name == "tensor_factory_mcp"


@pytest.mark.unit
def test_tools_registered():
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    expected = {"tensor_factory_detect", "tensor_factory_model_info", "tensor_factory_benchmark"}
    assert expected <= names


@pytest.mark.unit
def test_detect_tool_returns_json(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(p)
    out = server.tensor_factory_detect(server.DetectInput(image_path=str(p)))
    data = json.loads(out)
    assert "box_norm" in data and "uint8" in data


@pytest.mark.unit
def test_detect_tool_missing_image_is_actionable():
    out = server.tensor_factory_detect(server.DetectInput(image_path="/no/such.png"))
    assert out.startswith("Error:")
