import pytest

from tensor_factory_synth.device import enable_mps_fallback, resolve_device


@pytest.mark.unit
def test_resolve_device_returns_valid_string():
    # torch is absent in the default env, so this must degrade to cpu without raising.
    assert resolve_device() in ("cuda", "mps", "cpu")


@pytest.mark.unit
def test_resolve_device_cpu_preference_without_torch():
    assert resolve_device("cpu") == "cpu"


@pytest.mark.unit
def test_enable_mps_fallback_sets_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    enable_mps_fallback()
    import os

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
