"""Contract tests for the single-file detection app (docs/app.html).

The page is the binary: one HTML file with the model, fonts and code inside it, so it must
keep working with the network unplugged. These tests pin that it loads nothing from
anywhere else and that the landing page links to it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"


@pytest.fixture(scope="module")
def html() -> str:
    return (DOCS / "app.html").read_text(encoding="utf-8")


@pytest.mark.unit
def test_app_loads_no_external_resource(html):
    assert not re.search(r"<script[^>]+\bsrc=", html)
    assert not re.search(r"<link[^>]+\bhref=", html)
    assert not re.search(r"<(?:img|video|audio|source|iframe)[^>]+\bsrc=[\"'](?!data:|blob:)", html)
    assert not re.search(
        r"\b(?:fetch|importScripts|XMLHttpRequest|WebSocket)\s*\(\s*[\"']https?:", html
    )
    assert "@import" not in html
    assert not re.search(r"url\(\s*[\"']?https?:", html)


@pytest.mark.unit
def test_app_only_links_out_to_the_repo(html):
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", html))
    assert urls <= {"https://github.com/JimothyJohn/tensor-factory"}, urls


@pytest.mark.unit
def test_app_local_links_resolve(html):
    for target in set(re.findall(r'href="([a-z-]+\.html)"', html)):
        assert (DOCS / target).is_file(), target


@pytest.mark.unit
def test_app_is_linked_from_the_landing_page():
    assert 'href="app.html"' in (DOCS / "index.html").read_text(encoding="utf-8")
