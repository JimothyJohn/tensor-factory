"""The S3 store: signing pinned to AWS's published examples, and the push/pull contract.

The contract tests run the real client over real HTTP against a small in-process
S3-compatible server (path-style, as the endpoint override expects). The server is on the
adversary's side where it matters: it can hand back wrong bytes and hostile manifests.
The integration tests at the bottom talk to the real bucket.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
from xml.sax.saxutils import escape

import pytest

from tensor_factory import cli
from tensor_factory.store import Credentials, S3Store, StoreError, sha256_file, sign

# -- SigV4: the worked examples from the Amazon S3 API reference --------------------------
# "Signature Calculations for the Authorization Header: Transferring Payload in a Single
# Chunk". Same key, bucket and timestamp throughout.
DOC_CREDS = Credentials("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
DOC_HOST = "examplebucket.s3.amazonaws.com"
DOC_NOW = dt.datetime(2013, 5, 24, tzinfo=dt.UTC)
EMPTY = hashlib.sha256(b"").hexdigest()


def _signature(headers: dict[str, str]) -> str:
    return headers["authorization"].rsplit("Signature=", 1)[1]


@pytest.mark.unit
def test_sign_matches_aws_get_object_example():
    out = sign(
        "GET", DOC_HOST, "/test.txt", {}, {"Range": "bytes=0-9"}, EMPTY, DOC_CREDS,
        "us-east-1", DOC_NOW,
    )  # fmt: skip
    assert _signature(out) == "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    assert "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date" in out["authorization"]


@pytest.mark.unit
def test_sign_matches_aws_put_object_example():
    body = b"Welcome to Amazon S3."
    out = sign(
        "PUT",
        DOC_HOST,
        "/test$file.text",
        {},
        {"Date": "Fri, 24 May 2013 00:00:00 GMT", "x-amz-storage-class": "REDUCED_REDUNDANCY"},
        hashlib.sha256(body).hexdigest(),
        DOC_CREDS,
        "us-east-1",
        DOC_NOW,
    )
    assert _signature(out) == "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"


@pytest.mark.unit
def test_sign_matches_aws_list_objects_example():
    out = sign(
        "GET", DOC_HOST, "/", {"max-keys": "2", "prefix": "J"}, {}, EMPTY, DOC_CREDS,
        "us-east-1", DOC_NOW,
    )  # fmt: skip
    assert _signature(out) == "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"


@pytest.mark.unit
def test_sign_includes_session_token_in_signed_headers():
    creds = Credentials(DOC_CREDS.access_key, DOC_CREDS.secret_key, "tok")
    out = sign("GET", DOC_HOST, "/k", {}, {}, EMPTY, creds, "us-east-1", DOC_NOW)
    assert out["x-amz-security-token"] == "tok"
    assert "x-amz-security-token" in out["authorization"]


@pytest.mark.unit
def test_credentials_never_show_the_secret():
    creds = Credentials("AKIAEXAMPLE1234", "super-secret-value", "session-secret")
    assert "super-secret-value" not in repr(creds) + str(creds)
    assert "session-secret" not in repr(creds)


@pytest.mark.unit
@pytest.mark.parametrize("env", [{}, {"AWS_ACCESS_KEY_ID": "a"}, {"AWS_SECRET_ACCESS_KEY": "s"}])
def test_credentials_missing_is_a_clear_error(env):
    with pytest.raises(StoreError, match="AWS_ACCESS_KEY_ID"):
        Credentials.from_env(env)


@pytest.mark.unit
@pytest.mark.parametrize(
    "url", ["", "bucket", "https://bucket", "s3://", "s3://Has_Caps", "s3://a.b.c/x", "s3://ab"]
)
def test_bad_store_url_is_rejected(url):
    with pytest.raises(StoreError):
        S3Store(url, credentials=DOC_CREDS)


# -- an S3-compatible server, in process ---------------------------------------------------


class _Bucket(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.objects: dict[str, bytes] = {}
        self.fail_next = 0  # answer this many requests with 503 first
        self.requests: list[tuple[str, str, dict[str, str]]] = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    @property
    def bucket(self) -> _Bucket:
        assert isinstance(self.server, _Bucket)
        return self.server

    def _reply(self, status: int, body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _handle(self) -> None:
        parts = urlsplit(self.path)
        key = unquote(parts.path).lstrip("/")
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.bucket.requests.append((self.command, key, dict(self.headers.items())))
        if self.bucket.fail_next > 0:
            self.bucket.fail_next -= 1
            return self._reply(503, b"<Error><Code>SlowDown</Code></Error>")
        if not self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 "):
            return self._reply(403, b"<Error><Code>AccessDenied</Code></Error>")
        objects = self.bucket.objects
        if self.command == "PUT":
            if self.headers.get("x-amz-content-sha256") != hashlib.sha256(body).hexdigest():
                return self._reply(400, b"<Error><Code>XAmzContentSHA256Mismatch</Code></Error>")
            objects[key] = body
            return self._reply(200)
        query = parse_qs(parts.query)
        if "list-type" in query:
            prefix = key.rstrip("/") + "/" + query.get("prefix", [""])[0]
            start = int(query.get("continuation-token", ["0"])[0])
            keys = sorted(k for k in objects if k.startswith(prefix))
            page, more = keys[start : start + 2], start + 2 < len(keys)  # 2 per page
            xml = "<ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>"
            xml += f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
            if more:
                xml += f"<NextContinuationToken>{start + 2}</NextContinuationToken>"
            for k in page:
                short = escape(k.split("/", 1)[1])
                xml += f"<Contents><Key>{short}</Key><Size>{len(objects[k])}</Size></Contents>"
            return self._reply(200, (xml + "</ListBucketResult>").encode())
        if key not in objects:
            return self._reply(404, b"<Error><Code>NoSuchKey</Code></Error>")
        return self._reply(200, objects[key])

    do_GET = do_PUT = do_HEAD = _handle


@pytest.fixture
def bucket():
    server = _Bucket()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def store(bucket, monkeypatch):
    monkeypatch.setattr("tensor_factory.store.time.sleep", lambda _s: None)
    endpoint = f"http://127.0.0.1:{bucket.server_address[1]}"
    return S3Store("s3://test-bucket/pre/fix", credentials=DOC_CREDS, endpoint=endpoint)


def _blob_key(sha: str) -> str:
    return f"test-bucket/pre/fix/blobs/sha256/{sha}"


@pytest.mark.unit
def test_model_round_trip_by_sha_and_by_name(store, bucket, tmp_path):
    src = tmp_path / "m.onnx"
    src.write_bytes(os.urandom(4096))
    sha = store.push_model(src, "cam-v9")
    assert sha == sha256_file(src)
    assert bucket.objects[_blob_key(sha)] == src.read_bytes()

    by_sha, by_name = tmp_path / "a" / "by-sha.onnx", tmp_path / "by-name.onnx"
    assert store.pull_model(sha, by_sha) == sha
    assert store.pull_model("cam-v9", by_name) == sha
    assert by_sha.read_bytes() == by_name.read_bytes() == src.read_bytes()


@pytest.mark.unit
def test_push_is_idempotent_and_does_not_reupload(store, bucket, tmp_path):
    src = tmp_path / "m.onnx"
    src.write_bytes(b"model-bytes")
    store.push_file(src)
    store.push_file(src)
    puts = [r for r in bucket.requests if r[0] == "PUT"]
    assert len(puts) == 1


@pytest.mark.unit
def test_corrupted_blob_is_refused_and_the_old_file_survives(store, bucket, tmp_path):
    src = tmp_path / "m.onnx"
    src.write_bytes(b"the real model")
    sha = store.push_model(src)
    bucket.objects[_blob_key(sha)] = b"swapped in the bucket"

    serving = tmp_path / "serving.onnx"
    serving.write_bytes(b"model currently serving")
    with pytest.raises(StoreError, match="sha256 mismatch"):
        store.pull_model(sha, serving)
    assert serving.read_bytes() == b"model currently serving"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".part")] == []


@pytest.mark.unit
def test_pull_skips_download_when_the_file_already_matches(store, bucket, tmp_path):
    src = tmp_path / "m.onnx"
    src.write_bytes(b"abc")
    sha = store.push_model(src)
    bucket.requests.clear()
    store.pull_blob(sha, src)
    assert bucket.requests == []


@pytest.mark.unit
@pytest.mark.parametrize("ref", ["nope", "0" * 64])
def test_missing_model_is_a_clear_error(store, tmp_path, ref):
    with pytest.raises(StoreError, match="not in the store"):
        store.pull_model(ref, tmp_path / "x.onnx")
    assert not (tmp_path / "x.onnx").exists()


@pytest.mark.unit
@pytest.mark.parametrize("name", ["../up", "a/b", ".hidden", "x" * 200, "a b", "é"])
def test_hostile_names_never_become_keys(store, bucket, tmp_path, name):
    src = tmp_path / "m.onnx"
    src.write_bytes(b"x")
    with pytest.raises(StoreError, match="invalid"):
        store.push_model(src, name)
    with pytest.raises(StoreError, match="invalid"):
        store.pull_dataset(name, tmp_path / "out")
    assert not any("models/" in k for k in bucket.objects)


@pytest.mark.unit
def test_dataset_round_trip_keeps_layout_and_dedupes(store, bucket, tmp_path):
    src = tmp_path / "ds"
    (src / "images").mkdir(parents=True)
    (src / "images" / "a.png").write_bytes(b"same")
    (src / "images" / "b.png").write_bytes(b"same")
    (src / "labels.json").write_text('{"k": 1}')
    (tmp_path / "outside.txt").write_text("secret")
    (src / "link").symlink_to(tmp_path / "outside.txt")

    manifest = store.push_dataset(src)
    assert manifest["name"] == "ds"
    assert [f["path"] for f in manifest["files"]] == ["images/a.png", "images/b.png", "labels.json"]
    assert sum("blobs/sha256/" in k for k in bucket.objects) == 2  # a.png == b.png

    out = tmp_path / "restored"
    store.pull_dataset("ds", out)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "images/a.png",
        "images/b.png",
        "labels.json",
    ]
    assert (out / "labels.json").read_text() == '{"k": 1}'


@pytest.mark.unit
@pytest.mark.parametrize(
    "path", ["../escape.txt", "/etc/passwd", "a/../../escape.txt", "a\\..\\b", "", "./x", "a//b"]
)
def test_manifest_path_traversal_writes_nothing(store, bucket, tmp_path, path):
    payload = b"owned"
    sha = hashlib.sha256(payload).hexdigest()
    bucket.objects[_blob_key(sha)] = payload
    good = {"path": "ok.txt", "sha256": sha, "size": len(payload)}
    evil = {"path": path, "sha256": sha, "size": len(payload)}
    bucket.objects["test-bucket/pre/fix/datasets/evil/manifest.json"] = json.dumps(
        {"name": "evil", "files": [good, evil]}
    ).encode()

    out = tmp_path / "sandbox" / "out"
    with pytest.raises(StoreError, match="unsafe path"):
        store.pull_dataset("evil", out)
    assert not out.exists()  # validated before the first write, so not even ok.txt
    assert list(tmp_path.rglob("escape.txt")) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "doc", [b"not json", b"[]", b'{"files": "x"}', b'{"files": [{"path": "a", "sha256": "zz"}]}']
)
def test_malformed_manifest_is_a_store_error(store, bucket, tmp_path, doc):
    bucket.objects["test-bucket/pre/fix/datasets/bad/manifest.json"] = doc
    with pytest.raises(StoreError):
        store.pull_dataset("bad", tmp_path / "out")


@pytest.mark.unit
def test_empty_dataset_directory_is_refused(store, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(StoreError, match="no files"):
        store.push_dataset(tmp_path / "empty")


@pytest.mark.unit
def test_list_follows_pagination(store, bucket, tmp_path):
    for i in range(5):
        store.put_bytes(f"models/m{i}.json", b"{}")
    assert [k for k, _ in store.list("models/")] == [f"models/m{i}.json" for i in range(5)]


@pytest.mark.unit
def test_server_errors_are_retried_then_reported_without_secrets(store, bucket):
    bucket.fail_next = 2
    store.put_bytes("models/x.json", b"{}")  # third attempt lands

    bucket.fail_next = 99
    with pytest.raises(StoreError) as err:
        store.get_bytes("models/x.json")
    assert "503" in str(err.value) and "SlowDown" in str(err.value)
    assert DOC_CREDS.secret_key not in str(err.value)


@pytest.mark.unit
def test_unreachable_endpoint_is_a_store_error(monkeypatch):
    monkeypatch.setattr("tensor_factory.store.time.sleep", lambda _s: None)
    dead = S3Store("s3://test-bucket", credentials=DOC_CREDS, endpoint="http://127.0.0.1:9")
    with pytest.raises(StoreError, match="network error"):
        dead.exists("k")


@pytest.mark.unit
def test_cli_push_then_pull(bucket, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", DOC_CREDS.access_key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", DOC_CREDS.secret_key)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    endpoint = f"http://127.0.0.1:{bucket.server_address[1]}"
    monkeypatch.setenv("TENSOR_FACTORY_STORE_ENDPOINT", endpoint)
    monkeypatch.setenv("TENSOR_FACTORY_STORE", "s3://test-bucket")
    src = tmp_path / "m.onnx"
    src.write_bytes(b"cli model")

    assert cli.main(["store", "push-model", str(src)]) == 0
    sha = capsys.readouterr().out.split()[0]
    assert sha == sha256_file(src)
    assert cli.main(["store", "pull-model", sha, "--out", str(tmp_path / "o.onnx")]) == 0
    assert (tmp_path / "o.onnx").read_bytes() == b"cli model"
    assert cli.main(["store", "pull-model", "missing", "--out", str(tmp_path / "n.onnx")]) == 1
    assert "not in the store" in capsys.readouterr().err


@pytest.mark.unit
def test_cli_without_a_store_configured_fails_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("TENSOR_FACTORY_STORE", raising=False)
    assert cli.main(["store", "ls"]) == 1
    assert "TENSOR_FACTORY_STORE" in capsys.readouterr().err


# -- the real bucket ----------------------------------------------------------------------

_REAL = os.environ.get("TENSOR_FACTORY_STORE")
needs_bucket = pytest.mark.skipif(
    not (_REAL and os.environ.get("AWS_ACCESS_KEY_ID")),
    reason="set TENSOR_FACTORY_STORE and AWS credentials to run against the real bucket",
)


@pytest.fixture
def real_store():
    # A unique prefix per run; versioning plus the bucket's lifecycle rule expire what is
    # left behind (this client deliberately has no delete).
    return S3Store.from_env(f"{(_REAL or '').rstrip('/')}/ci-tests/{uuid.uuid4().hex}")


@needs_bucket
@pytest.mark.integration
def test_real_bucket_model_and_dataset_round_trip(real_store, tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(os.urandom(100_000))
    sha = real_store.push_model(model, "it-model")
    assert real_store.pull_model("it-model", tmp_path / "back.onnx") == sha
    assert (tmp_path / "back.onnx").read_bytes() == model.read_bytes()

    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    for i in range(3):
        (ds / "images" / f"frame {i} (é).png").write_bytes(os.urandom(2048))
    real_store.push_dataset(ds, "it-ds")
    real_store.pull_dataset("it-ds", tmp_path / "restored")
    for src in (ds / "images").iterdir():
        assert (tmp_path / "restored" / "images" / src.name).read_bytes() == src.read_bytes()

    keys = dict(real_store.list())
    assert keys[f"blobs/sha256/{sha}"] == 100_000
    assert "models/it-model.json" in keys and "datasets/it-ds/manifest.json" in keys


@needs_bucket
@pytest.mark.integration
def test_real_bucket_missing_key_and_bad_credentials(real_store, tmp_path):
    with pytest.raises(StoreError, match="not in the store"):
        real_store.pull_model("f" * 64, tmp_path / "x.onnx")

    wrong = S3Store(
        f"s3://{real_store.bucket}/{real_store.prefix}",
        credentials=Credentials("AKIAIOSFODNN7EXAMPLE", "not-the-secret"),
        region=real_store.region,
    )
    with pytest.raises(StoreError, match="HTTP 403") as err:
        wrong.put_bytes("models/x.json", b"{}")
    assert "not-the-secret" not in str(err.value)
