import pytest

import tensor_factory


@pytest.mark.unit
def test_version_is_exposed():
    assert tensor_factory.__version__ == "0.1.0"
