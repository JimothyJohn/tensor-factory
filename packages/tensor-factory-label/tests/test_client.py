import json

import httpx
import pytest

from tensor_factory_label import LabelStudioClient


def _transport():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Token tok"
        path = request.url.path
        if request.method == "POST" and path == "/api/projects/":
            body = json.loads(request.content)
            assert body == {"title": "t", "label_config": "<View/>"}
            return httpx.Response(201, json={"id": 7})
        if request.method == "POST" and path == "/api/projects/7/import":
            tasks = json.loads(request.content)
            assert isinstance(tasks, list)
            return httpx.Response(201, json={"task_count": len(tasks)})
        if request.method == "GET" and path == "/api/projects/7/export":
            assert request.url.params.get("exportType") == "JSON"
            return httpx.Response(200, json=[{"data": {}, "annotations": []}])
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


@pytest.fixture
def client():
    return LabelStudioClient("http://ls.local", "tok", transport=_transport())


@pytest.mark.unit
def test_create_project_returns_id(client):
    assert client.create_project("t", "<View/>") == 7


@pytest.mark.unit
def test_import_tasks_posts_list(client):
    out = client.import_tasks(7, [{"data": {"image": "x"}}, {"data": {"image": "y"}}])
    assert out["task_count"] == 2


@pytest.mark.unit
def test_export_json_returns_tasks(client):
    assert client.export_json(7) == [{"data": {}, "annotations": []}]


@pytest.mark.unit
def test_http_error_raises(client):
    with pytest.raises(httpx.HTTPStatusError):
        client.export_json(999)
