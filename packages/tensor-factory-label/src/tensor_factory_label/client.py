"""Minimal Label Studio REST client (httpx).

Targets the stable open-source endpoints (verified against api.labelstud.io):
``POST /api/projects/``, ``POST /api/projects/{id}/import``, and the community
``GET /api/projects/{id}/export?exportType=JSON``. Using the REST API directly avoids the
churn of the Label Studio Python SDK across versions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx


class LabelStudioClient:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = url.rstrip("/")
        self._headers = {"Authorization": f"Token {token}"}
        self._timeout = timeout
        self._transport = transport  # injected in tests via httpx.MockTransport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base,
            headers=self._headers,
            timeout=self._timeout,
            transport=self._transport,
        )

    def create_project(self, title: str, label_config: str) -> int:
        """Create a project; return its id."""
        with self._client() as client:
            resp = client.post(
                "/api/projects/", json={"title": title, "label_config": label_config}
            )
            resp.raise_for_status()
            return int(resp.json()["id"])

    def import_tasks(self, project_id: int, tasks: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Bulk-import tasks (with optional predictions) into a project."""
        with self._client() as client:
            resp = client.post(f"/api/projects/{project_id}/import", json=list(tasks))
            resp.raise_for_status()
            return resp.json()

    def export_json(self, project_id: int) -> list[dict[str, Any]]:
        """Export a project's annotated tasks as a JSON list (community endpoint)."""
        with self._client() as client:
            resp = client.get(f"/api/projects/{project_id}/export", params={"exportType": "JSON"})
            resp.raise_for_status()
            return resp.json()
