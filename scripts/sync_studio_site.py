#!/usr/bin/env python3
"""Copy the Studio's page into the site, so it is hosted at <site>/studio/.

GitHub Pages serves ``docs/`` as is; the Studio lives in ``studio/`` (the backend serves
it from there). This copies the files a browser needs (the page, its styles, ``js/`` and
the ONNX templates in ``models/``) to ``docs/studio/`` and removes anything there that no
longer exists in the source. ``--check`` changes nothing and exits 1 when the two differ;
the test suite runs it, so a change to ``studio/`` cannot merge without the copy.

    python3 scripts/sync_studio_site.py          # copy
    python3 scripts/sync_studio_site.py --check  # verify
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "studio"
DST = REPO / "docs" / "studio"
TOP = ("index.html", "studio.css")
TREES = ("js", "models")


def wanted() -> dict[str, bytes]:
    """Site-relative path -> bytes for every file the hosted page needs."""
    files = {name: (SRC / name).read_bytes() for name in TOP}
    for tree in TREES:
        for path in sorted((SRC / tree).rglob("*")):
            if path.is_file():
                files[path.relative_to(SRC).as_posix()] = path.read_bytes()
    return files


def present() -> dict[str, bytes]:
    if not DST.is_dir():
        return {}
    return {p.relative_to(DST).as_posix(): p.read_bytes() for p in DST.rglob("*") if p.is_file()}


def main(argv: list[str]) -> int:
    want, have = wanted(), present()
    stale = sorted(set(have) - set(want))
    changed = sorted(k for k, v in want.items() if have.get(k) != v)
    if "--check" in argv:
        for name in changed:
            print(f"out of date: docs/studio/{name}")
        for name in stale:
            print(f"no longer in studio/: docs/studio/{name}")
        if changed or stale:
            print("run: python3 scripts/sync_studio_site.py")
            return 1
        return 0
    for name in stale:
        (DST / name).unlink()
    for name in changed:
        out = DST / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(want[name])
    print(f"docs/studio: {len(changed)} written, {len(stale)} removed, {len(want)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
