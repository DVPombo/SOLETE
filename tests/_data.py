# -*- coding: utf-8 -*-
"""Where the tests find their data (all through solete.paths).

* examples/SOLETE_short.h5            tracked in git (9 KB) -- always available
* data/hdf5/SOLETE_Pombo_60min.h5     the v3 hourly file from figshare; NOT in git.
  Tests that need it SKIP (with the download instructions) when it is absent,
  so a fresh clone can still run the rest of the suite.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from solete.paths import EXAMPLES_DIR, find_data_file


def short_sample_path():
    return EXAMPLES_DIR / "SOLETE_short.h5"


def v3_60min_path():
    try:
        return find_data_file("60min", version="v3")
    except FileNotFoundError as e:
        pytest.skip("needs the v3 hourly file in data/hdf5/ -- " + str(e).splitlines()[0])
