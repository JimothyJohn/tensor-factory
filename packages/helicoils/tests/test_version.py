import pytest

import helicoils


@pytest.mark.unit
def test_version_is_exposed():
    assert helicoils.__version__ == "0.1.0"
