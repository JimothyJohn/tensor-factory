"""Model hot reload: replace the file, the next call runs the new model, no restart.

And the failure half of that contract: a replacement that does not load must never take
down a server that was working.
"""

import hashlib
import os
import shutil
import threading

import pytest
from PIL import Image

from tensor_factory_mcp import core, http_server

MODELS = core.default_model_path().parent
PRESENCE = MODELS / "helicoil-presence-cam-v1.onnx"  # has a presence head
BOX_ONLY = MODELS / "helicoil-mock-v1.onnx"  # does not


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace(src, dst) -> None:
    """Swap a model in the way a deploy does: write beside it, then rename over it."""
    tmp = dst.with_suffix(".incoming")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    # Same-size swaps within one mtime tick would be invisible; real deploys are seconds
    # apart, the test makes that explicit instead of sleeping.
    st = dst.stat()
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.onnx"
    shutil.copyfile(BOX_ONLY, path)
    return path


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(p)
    return str(p)


@pytest.mark.unit
def test_model_info_identifies_the_loaded_model(model):
    info = core.model_info(str(model))
    assert info["sha256"] == _sha(BOX_ONLY)
    assert info["bytes"] == BOX_ONLY.stat().st_size
    assert "reload_error" not in info


@pytest.mark.unit
def test_replacing_the_file_swaps_the_model_without_a_restart(model, image):
    before = core.detect(image, str(model))
    assert "present" not in before  # box-only model

    _replace(PRESENCE, model)

    after = core.detect(image, str(model))
    assert "present" in after  # the presence-head model is now the one answering
    assert core.model_info(str(model))["sha256"] == _sha(PRESENCE)


@pytest.mark.unit
def test_unchanged_file_reuses_the_loaded_detector(model):
    first = core._detector(str(model), 480)
    assert core._detector(str(model), 480) is first


@pytest.mark.unit
def test_a_corrupt_replacement_keeps_the_working_model_serving(model, image):
    good = core.detect(image, str(model))

    model.write_bytes(b"this is not a model" * 50)

    still = core.detect(image, str(model))  # must not raise
    assert still["box_norm"] == good["box_norm"]
    info = core.model_info(str(model))
    assert info["sha256"] == _sha(BOX_ONLY)  # reports what is actually running
    assert info["reload_error"]


@pytest.mark.unit
def test_a_truncated_copy_in_progress_does_not_break_serving(model, image):
    # scp writes in place: mid-transfer the file is a valid prefix of a model.
    core.detect(image, str(model))
    model.write_bytes(PRESENCE.read_bytes()[:1000])
    assert "present" not in core.detect(image, str(model))  # old model still answering
    assert core.model_info(str(model))["reload_error"]


@pytest.mark.unit
def test_serving_recovers_once_a_good_file_lands(model, image):
    core.detect(image, str(model))
    model.write_bytes(b"garbage")
    assert core.model_info(str(model))["reload_error"]

    _replace(PRESENCE, model)

    info = core.model_info(str(model))
    assert info["sha256"] == _sha(PRESENCE)
    assert "reload_error" not in info
    assert "present" in core.detect(image, str(model))


@pytest.mark.unit
def test_a_deleted_file_keeps_the_loaded_model_serving(model, image):
    good = core.detect(image, str(model))
    key = (str(model), 480, "auto")
    model.unlink()
    # resolve_model refuses a missing path up front, so go through the cache directly:
    # the rm-then-cp window of a careless deploy.
    loaded = core._loaded(*key)
    assert loaded.reload_error
    assert loaded.sha256 == _sha(BOX_ONLY)
    assert good["box_norm"] is not None


@pytest.mark.unit
def test_a_bad_model_with_nothing_loaded_still_fails_loudly(tmp_path, image):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"nope")
    with pytest.raises(Exception):  # noqa: B017, PT011 -- engine-specific error type
        core.detect(image, str(bad))


@pytest.mark.unit
def test_detector_cache_stays_bounded(tmp_path):
    for i in range(7):
        p = tmp_path / f"m{i}.onnx"
        shutil.copyfile(BOX_ONLY, p)
        core._detector(str(p), 480)
    assert len(core._detectors) <= 4


@pytest.mark.unit
def test_reload_under_concurrent_requests_never_errors(model, image):
    # Requests racing a swap must each get an answer from one model or the other.
    errors: list[BaseException] = []
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            try:
                out = core.detect(image, str(model))
                assert out["image_size"] == {"width": 64, "height": 64}
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for src in (PRESENCE, BOX_ONLY, PRESENCE):
            _replace(src, model)
            core.model_info(str(model))
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)
    assert not errors, errors[:1]
    assert core.model_info(str(model))["sha256"] == _sha(PRESENCE)


@pytest.mark.unit
def test_http_model_info_reports_sha_and_follows_a_swap(model):
    import json
    import urllib.request

    httpd = http_server.make_server("127.0.0.1", 0, model_path=str(model))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:

        def info() -> dict:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/model_info", timeout=10) as r:
                return json.loads(r.read())

        assert info()["sha256"] == _sha(BOX_ONLY)
        _replace(PRESENCE, model)
        assert info()["sha256"] == _sha(PRESENCE)
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
