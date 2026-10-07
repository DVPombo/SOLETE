import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "dataset" / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from build_release import EXPECTED_REAL_COUNTS, MEASURED_COLUMNS, build_clean, build_original
from clean_solete_1sec import clean_dataframe


def _raw_frame(index):
    frame = pd.DataFrame(
        {
            "TEMPERATURE[degC]": 10.0,
            "HUMIDITY[%]": 0.5,
            "WIND_SPEED[m1s]": 2.0,
            "WIND_DIR[deg]": 10.0,
            "GHI[kW1m2]": 0.1,
            "POA Irr[kW1m2]": 0.1,
            "P_Gaia[kW]": 0.0,
            "P_Solar[kW]": 0.0,
            "Pressure[mbar]": np.linspace(1005.0, 1015.0, len(index)),
            "Azimuth[deg]": 0.0,
            "Elevation[deg]": 0.0,
        },
        index=index,
    )
    return frame


def test_inclusive_real_span_counts_left_labelled_buckets():
    start = pd.Timestamp("2018-06-01 00:00:00")
    end = pd.Timestamp("2019-09-01 00:00:00")

    assert int((end - start) / pd.Timedelta(seconds=1)) + 1 == EXPECTED_REAL_COUNTS["1sec"]
    assert int((end - start) / pd.Timedelta(minutes=1)) + 1 == EXPECTED_REAL_COUNTS["1min"]
    assert int((end - start) / pd.Timedelta(minutes=5)) + 1 == EXPECTED_REAL_COUNTS["5min"]
    assert int((end - start) / pd.Timedelta(minutes=60)) + 1 == EXPECTED_REAL_COUNTS["60min"]


def test_original_is_sorted_and_contains_only_measured_columns(tmp_path):
    first = _raw_frame(pd.date_range("2019-03-30", periods=20, freq="1s"))
    second = _raw_frame(pd.date_range("2019-03-31", periods=20, freq="1s"))
    raw = tmp_path / "raw.h5"
    output = tmp_path / "original.h5"
    pd.concat([second, first]).to_hdf(raw, key="DATA")

    build_original(raw, output, chunk_rows=7, overwrite=False)
    actual = pd.read_hdf(output, key="DATA")
    expected = pd.concat([first, second])[MEASURED_COLUMNS]
    expected.index = expected.index.as_unit("ns")

    assert actual.index.is_monotonic_increasing
    assert actual.columns.tolist() == MEASURED_COLUMNS
    pd.testing.assert_frame_equal(actual, expected)


def test_overlapped_slices_equal_whole_cleaning_across_boundaries(tmp_path):
    index = pd.date_range("2019-03-31", periods=1_200, freq="1s")
    raw = _raw_frame(index)[MEASURED_COLUMNS]
    raw.loc[index[498:503], ["HUMIDITY[%]", "WIND_SPEED[m1s]"]] = 0.0
    raw.loc[index[499:502], "TEMPERATURE[degC]"] = -40.0
    raw.loc[index[450:850], "Pressure[mbar]"] = 997.5
    original = tmp_path / "original.h5"
    sliced_path = tmp_path / "sliced.h5"
    raw.to_hdf(original, key="DATA", format="table")

    whole, _ = clean_dataframe(raw, solar_chunk_rows=500)
    build_clean(
        original,
        sliced_path,
        slice_rows=500,
        overlap=300,
        overwrite=False,
    )
    sliced = pd.read_hdf(sliced_path, key="DATA")

    pd.testing.assert_frame_equal(sliced, whole)