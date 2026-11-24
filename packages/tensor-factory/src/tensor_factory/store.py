"""S3 store for models and datasets -- pure stdlib, so the edge can pull with no packages.

Everything is content-addressed: a file's bytes live once at ``blobs/sha256/<hex>``, and a
small JSON document names them.

    <prefix>/blobs/sha256/<hex>              the bytes of one file
    <prefix>/models/<name>.json              {"sha256", "size", "file"}: a named model
    <prefix>/datasets/<name>/manifest.json   {"name", "files": [{"path", "sha256", "size"}]}

A model is pulled by its sha256 (or by name, which resolves to one), verified, and moved
into place atomically, which is exactly what the serving layer's hot reload watches for.

Requests are signed with AWS Signature Version 4 using ``hmac`` and ``hashlib``; there is
no boto3 and no ``aws`` CLI on the path. Configuration comes from the environment:

    TENSOR_FACTORY_STORE            s3://bucket[/prefix]
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN (optional)
    AWS_REGION or AWS_DEFAULT_REGION   (default us-east-1)
    TENSOR_FACTORY_STORE_ENDPOINT   optional: an S3-compatible endpoint, path-style
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

__all__ = ["Credentials", "S3Store", "StoreError", "sha256_file", "sign"]

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA = re.compile(r"[0-9a-f]{64}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]")
_TIMEOUT_S = 60
_ATTEMPTS = 3


class StoreError(RuntimeError):
    """A store operation failed; the message never contains credentials."""


@dataclass(frozen=True)
class Credentials:
    access_key: str
    secret_key: str
    session_token: str | None = None

    def __repr__(self) -> str:  # a traceback or log line must not carry the secret
        return f"Credentials(access_key=...{self.access_key[-4:]})"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Credentials:
        env = os.environ if env is None else env
        access, secret = env.get("AWS_ACCESS_KEY_ID"), env.get("AWS_SECRET_ACCESS_KEY")
        if not access or not secret:
            raise StoreError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")
        return cls(access, secret, env.get("AWS_SESSION_TOKEN") or None)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uri_encode(text: str, *, slash_safe: bool) -> str:
    return quote(text, safe="-_.~/" if slash_safe else "-_.~")


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def sign(
    method: str,
    host: str,
    path: str,
    query: Mapping[str, str],
    headers: Mapping[str, str],
    payload_sha256: str,
    credentials: Credentials,
    region: str,
    now: dt.datetime,
    service: str = "s3",
) -> dict[str, str]:
    """Return ``headers`` plus the SigV4 ``Authorization`` (and date / hash / token) headers.

    ``path`` is the raw, unencoded request path. Every header passed in is signed.
    """
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    day = amz_date[:8]
    out = {k.lower(): " ".join(str(v).split()) for k, v in headers.items()}
    out["host"] = host
    out["x-amz-content-sha256"] = payload_sha256
    out["x-amz-date"] = amz_date
    if credentials.session_token:
        out["x-amz-security-token"] = credentials.session_token

    signed = ";".join(sorted(out))
    canonical = "\n".join(
        [
            method,
            _uri_encode(path, slash_safe=True),
            "&".join(
                f"{_uri_encode(k, slash_safe=False)}={_uri_encode(v, slash_safe=False)}"
                for k, v in sorted(query.items())
            ),
            "".join(f"{k}:{out[k]}\n" for k in sorted(out)),
            signed,
            payload_sha256,
        ]
    )
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = _hmac(("AWS4" + credentials.secret_key).encode(), day)
    for part in (region, service, "aws4_request"):
        key = _hmac(key, part)
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    out["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={credentials.access_key}/{scope}, "
        f"SignedHeaders={signed}, Signature={signature}"
    )
    return out


def _check_name(name: str, what: str) -> str:
    if not _NAME.fullmatch(name) or ".." in name:
        raise StoreError(f"invalid {what} name {name!r}: use letters, digits, '.', '_' and '-'")
    return name


def _safe_relpath(text: str) -> PurePosixPath:
    """A manifest path is data from the bucket: it must stay inside the output directory."""
    # Judge the raw segments: PurePosixPath would quietly normalise "./x" and "a//b".
    if "\\" in text or "\x00" in text or any(p in ("", ".", "..") for p in text.split("/")):
        raise StoreError(f"unsafe path in manifest: {text!r}")
    return PurePosixPath(text)


class S3Store:
    """A bucket (and optional key prefix) holding content-addressed models and datasets."""

    def __init__(
        self,
        url: str,
        *,
        credentials: Credentials | None = None,
        region: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        parts = urlsplit(url)
        if parts.scheme != "s3" or not _BUCKET.fullmatch(parts.netloc):
            raise StoreError(f"store must look like s3://bucket[/prefix], got {url!r}")
        self.bucket = parts.netloc
        self.prefix = parts.path.strip("/")
        self.region = region or "us-east-1"
        self._credentials = credentials if credentials is not None else Credentials.from_env()
        if endpoint:
            ep = urlsplit(endpoint)
            if ep.scheme not in ("http", "https") or not ep.netloc:
                raise StoreError(f"endpoint must be an http(s) URL, got {endpoint!r}")
            self._base, self._host = f"{ep.scheme}://{ep.netloc}", ep.netloc
            self._root = f"/{self.bucket}"  # S3-compatible endpoints are path-style
        else:
            self._host = f"{self.bucket}.s3.{self.region}.amazonaws.com"
            self._base, self._root = f"https://{self._host}", ""

    @classmethod
    def from_env(cls, url: str | None = None, env: Mapping[str, str] | None = None) -> S3Store:
        env = os.environ if env is None else env
        url = url or env.get("TENSOR_FACTORY_STORE")
        if not url:
            raise StoreError("no store given: pass --store or set TENSOR_FACTORY_STORE")
        return cls(
            url,
            credentials=Credentials.from_env(env),
            region=env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION"),
            endpoint=env.get("TENSOR_FACTORY_STORE_ENDPOINT") or None,
        )

    # -- raw object operations ------------------------------------------------------------

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _request(
        self,
        method: str,
        key: str | None,
        *,
        query: Mapping[str, str] | None = None,
        body: bytes | None = None,
        body_sha256: str | None = None,
    ) -> tuple[int, bytes]:
        """One signed request. Returns (status, body); 404 is returned, not raised."""
        query = dict(query or {})
        path = self._root + ("/" + self._key(key) if key is not None else "/")
        payload_hash = body_sha256 or (hashlib.sha256(body).hexdigest() if body else _EMPTY_SHA256)
        url = self._base + _uri_encode(path, slash_safe=True)
        if query:
            url += "?" + "&".join(
                f"{_uri_encode(k, slash_safe=False)}={_uri_encode(v, slash_safe=False)}"
                for k, v in sorted(query.items())
            )
        last = "no attempt made"
        for attempt in range(_ATTEMPTS):
            now = dt.datetime.now(dt.UTC)
            headers = sign(
                method, self._host, path, query, {}, payload_hash, self._credentials,
                self.region, now,
            )  # fmt: skip
            req = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
                    return resp.status, resp.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read()
                if exc.code == 404:
                    return 404, detail
                last = f"HTTP {exc.code} {_error_code(detail)}".strip()
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = f"network error: {getattr(exc, 'reason', exc)}"
            if attempt + 1 < _ATTEMPTS:
                time.sleep(0.5 * (attempt + 1))
        raise StoreError(f"{method} s3://{self.bucket}/{self._key(key or '')} failed: {last}")

    def put_bytes(self, key: str, data: bytes) -> None:
        self._request("PUT", key, body=data)

    def get_bytes(self, key: str) -> bytes:
        status, body = self._request("GET", key)
        if status == 404:
            raise StoreError(f"not in the store: s3://{self.bucket}/{self._key(key)}")
        return body

    def exists(self, key: str) -> bool:
        return self._request("HEAD", key)[0] != 404

    def list(self, prefix: str = "") -> Iterator[tuple[str, int]]:
        """Yield (key, size) for every object under ``prefix``, relative to the store."""
        full = self._key(prefix)
        strip = len(self.prefix) + 1 if self.prefix else 0
        token: str | None = None
        while True:
            query = {"list-type": "2", "prefix": full}
            if token:
                query["continuation-token"] = token
            status, body = self._request("GET", None, query=query)
            if status == 404:
                raise StoreError(f"bucket not found: {self.bucket}")
            root = ET.fromstring(body)  # noqa: S314 -- the response of a signed S3 request
            for item in root.iter():
                if _local(item.tag) == "Contents":
                    fields = {_local(c.tag): c.text or "" for c in item}
                    yield fields["Key"][strip:], int(fields.get("Size") or 0)
            fields = {_local(c.tag): c.text or "" for c in root}
            token = fields.get("NextContinuationToken")
            if fields.get("IsTruncated") != "true" or not token:
                return

    # -- blobs ----------------------------------------------------------------------------

    def push_file(self, path: str | Path) -> tuple[str, int]:
        """Upload one file under its sha256 (skipped if already there). Returns (sha, size)."""
        path = Path(path)
        sha, size = sha256_file(path), path.stat().st_size
        key = f"blobs/sha256/{sha}"
        if not self.exists(key):
            self._request("PUT", key, body=path.read_bytes(), body_sha256=sha)
        return sha, size

    def pull_blob(self, sha: str, out: str | Path) -> Path:
        """Download a blob, verify its sha256, then move it into place atomically.

        A file that fails verification is never left at ``out``; whatever was there stays.
        """
        if not _SHA.fullmatch(sha):
            raise StoreError(f"not a sha256: {sha!r}")
        out = Path(out)
        if out.is_file() and sha256_file(out) == sha:
            return out
        data = self.get_bytes(f"blobs/sha256/{sha}")
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise StoreError(f"sha256 mismatch for {sha}: the store returned {got}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, out)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return out

    # -- models ---------------------------------------------------------------------------

    def push_model(self, path: str | Path, name: str | None = None) -> str:
        path = Path(path)
        name = _check_name(name or path.stem, "model")
        sha, size = self.push_file(path)
        doc = {"sha256": sha, "size": size, "file": path.name}
        self.put_bytes(f"models/{name}.json", json.dumps(doc, sort_keys=True).encode())
        return sha

    def resolve_model(self, ref: str) -> str:
        """A 64-hex sha256 is itself; anything else is a model name looked up in the store."""
        if _SHA.fullmatch(ref):
            return ref
        doc = _load_json(self.get_bytes(f"models/{_check_name(ref, 'model')}.json"))
        sha = doc.get("sha256") if isinstance(doc, dict) else None
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise StoreError(f"model record {ref!r} is malformed")
        return sha

    def pull_model(self, ref: str, out: str | Path) -> str:
        sha = self.resolve_model(ref)
        self.pull_blob(sha, out)
        return sha

    # -- datasets -------------------------------------------------------------------------

    def push_dataset(self, directory: str | Path, name: str | None = None) -> dict:
        root = Path(directory)
        if not root.is_dir():
            raise StoreError(f"not a directory: {root}")
        name = _check_name(name or root.resolve().name, "dataset")
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue  # never follow a link out of the dataset
            sha, size = self.push_file(path)
            files.append({"path": path.relative_to(root).as_posix(), "sha256": sha, "size": size})
        if not files:
            raise StoreError(f"no files under {root}")
        manifest = {"name": name, "files": files}
        self.put_bytes(
            f"datasets/{name}/manifest.json", json.dumps(manifest, sort_keys=True).encode()
        )
        return manifest

    def pull_dataset(self, name: str, out_dir: str | Path) -> dict:
        manifest = _load_json(
            self.get_bytes(f"datasets/{_check_name(name, 'dataset')}/manifest.json")
        )
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
            raise StoreError(f"dataset manifest {name!r} is malformed")
        entries = manifest["files"]
        out_root = Path(out_dir)
        plan: list[tuple[Path, str]] = []
        for entry in entries:  # validate every path before writing anything
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise StoreError(f"dataset manifest {name!r} is malformed")
            sha = entry.get("sha256")
            if not isinstance(sha, str) or not _SHA.fullmatch(sha):
                raise StoreError(f"dataset manifest {name!r} is malformed")
            plan.append((out_root.joinpath(*_safe_relpath(entry["path"]).parts), sha))
        for target, sha in plan:
            self.pull_blob(sha, target)
        return manifest


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _error_code(body: bytes) -> str:
    """The <Code> of an S3 error document, e.g. AccessDenied; '' if there is none."""
    try:
        root = ET.fromstring(body)  # noqa: S314
    except ET.ParseError:
        return ""
    for item in root.iter():
        if _local(item.tag) == "Code":
            return (item.text or "")[:64]
    return ""


def _load_json(data: bytes) -> object:
    try:
        return json.loads(data)
    except ValueError as exc:
        raise StoreError("the store returned a document that is not JSON") from exc
