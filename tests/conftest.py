"""pytest configuration for the R-vs-PyDLNM differential suite (see tests/README.md)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import rhelpers  # noqa: E402,F401  (initialises R before any PyDLNM module is imported)
import pytest    # noqa: E402


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: takes more than ~30 s')


@pytest.fixture(scope='session')
def chicago_data():
    return rhelpers.chicago()
