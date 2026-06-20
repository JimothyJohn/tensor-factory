import asyncio
import json

import pytest
from PIL import Image

from helicoils_mcp import server


@pytest.mark.unit
def test_server_name():
    assert server.mcp.name == "helicoils_mcp"


@pytest.mark.unit
def test_tools_registered():
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"helicoils_detect", "helicoils_model_info", "helicoils_benchmark"} <= names


@pytest.mark.unit
def test_detect_tool_returns_json(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(p)
    out = server.helicoils_detect(server.DetectInput(image_path=str(p)))
    data = json.loads(out)
    assert "box_norm" in data and "uint8" in data


@pytest.mark.unit
def test_detect_tool_missing_image_is_actionable():
    out = server.helicoils_detect(server.DetectInput(image_path="/no/such.png"))
    assert out.startswith("Error:")
