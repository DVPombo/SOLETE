# -*- coding: utf-8 -*-
"""pytest configuration.

A test that needs a SOLETE data file which is not in the data folder is SKIPPED
(with the download instructions) instead of failing, so a fresh clone can run
the whole suite. Real failures are never converted: only solete.paths'
DataFileNotFoundError is.
"""
import pytest

from solete.paths import DataFileNotFoundError


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(DataFileNotFoundError) and rep.failed:
        rep.outcome = "skipped"
        reason = "missing data file -- " + str(call.excinfo.value).splitlines()[0] + " (see data/README.md)"
        rep.longrepr = (str(item.path), item.location[1] or 0, "Skipped: " + reason)
