# -*- coding: utf-8 -*-
"""
Tests for the Quality Control flag layer (Functions.apply_qc_flags and friends).

Every case is built from a real row pulled out of SOLETE_Pombo_60min.h5 by the 
exact criteria in QC_SCHEMA.md, with its value(s) copied inline -- not a fabricated row. 
The two exceptions are explicitly marked SYNTHETIC below: SOLETE_Pombo_60min.h5 
contains no row with WIND_DIR[deg] exactly 360.0, and none with a negative Pressure[mbar]/
HUMIDITY[%]/WIND_DIR[deg] value, so those specific boundaries can't be
sourced from the real file and are hand-built instead.

Run with: pytest tests/test_qc_flags.py -v  (from the repo root)
"""

import sys
import types
import pathlib

# Functions.py imports keras/tensorflow at module level for the ML forecasting
# code, which these tests never touch. Stub them out so this test file has no
# dependency on those (heavy, unrelated) packages being installed.
for _modname in ["keras", "keras.models", "keras.layers"]:
    sys.modules.setdefault(_modname, types.ModuleType(_modname))
sys.modules["keras.models"].Sequential = object
sys.modules["keras.models"].load_model = lambda *a, **k: None
for _n in ["LSTM", "Dense", "Masking", "Flatten", "Conv1D", "MaxPooling1D"]:
    setattr(sys.modules["keras.layers"], _n, object)

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from Functions import (
    apply_qc_flags,
    build_raw_value_qc_rules,
    build_substitution_qc_rule,
    QC_VALID,
    QC_MISSING,
    QC_PHYSICALLY_IMPLAUSIBLE,
    QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_60min():
    return pd.read_hdf(REPO_ROOT / "SOLETE_Pombo_60min.h5")


@pytest.fixture(scope="module")
def real_short():
    return pd.read_hdf(REPO_ROOT / "SOLETE_short.h5")


# ---------------------------------------------------------------------------
# Pressure[mbar] -- finding #1
# ---------------------------------------------------------------------------

def test_pressure_sentinel_1000_real_row(real_60min):
    # Real row: 2019-01-01 01:00:00, Pressure[mbar] == 1000.0 (the placeholder
    # covering 95.5% of the file).
    row = real_60min.loc[["2019-01-01 01:00:00"]].copy()
    assert row["Pressure[mbar]"].iloc[0] == 1000.0
    _, counts = apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["Pressure[mbar]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE
    assert counts["Pressure[mbar]_qc"] == 1


def test_pressure_sentinel_2000_real_row(real_60min):
    # Real row: 2018-11-17 01:00:00, Pressure[mbar] == 2000.0.
    row = real_60min.loc[["2018-11-17 01:00:00"]].copy()
    assert row["Pressure[mbar]"].iloc[0] == 2000.0
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["Pressure[mbar]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE


def test_pressure_sentinel_3000_real_row(real_60min):
    # Real row: 2018-11-17 00:00:00, one of only two rows at 3000.0.
    row = real_60min.loc[["2018-11-17 00:00:00"]].copy()
    assert row["Pressure[mbar]"].iloc[0] == 3000.0
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["Pressure[mbar]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE


def test_pressure_plausible_real_row_stays_valid(real_60min):
    # Real row: 2019-01-16 06:00:00, Pressure[mbar] ~= 998.34 -- one of the 13
    # non-sentinel rows in the file, well inside the plausible range.
    row = real_60min.loc[["2019-01-16 06:00:00"]].copy()
    assert row["Pressure[mbar]"].iloc[0] == pytest.approx(998.3448114183213)
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["Pressure[mbar]_qc"].iloc[0] == QC_VALID


def test_pressure_negative_synthetic():
    # SYNTHETIC: no row in either real file has a negative Pressure[mbar].
    # Built to isolate the general-range side of the detector
    # (PRESSURE_PLAUSIBLE_RANGE in Functions.py), which the three known
    # sentinels alone don't exercise.
    row = pd.DataFrame({"Pressure[mbar]": [-5.0]})
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["Pressure[mbar]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE


# ---------------------------------------------------------------------------
# HUMIDITY[%] -- finding #2
# ---------------------------------------------------------------------------

def test_humidity_over_one_real_row(real_60min):
    # Real row: 2018-11-18 00:00:00, HUMIDITY[%] ~= 2.70 -- the worst offender
    # in the file (188 rows exceed 1.0 in total).
    row = real_60min.loc[["2018-11-18 00:00:00"]].copy()
    assert row["HUMIDITY[%]"].iloc[0] == pytest.approx(2.700000000000057)
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["HUMIDITY[%]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE


def test_humidity_plausible_real_row_stays_valid(real_60min):
    # Real row: 2019-01-01 00:00:00, HUMIDITY[%] ~= 0.897, a normal fraction.
    row = real_60min.loc[["2019-01-01 00:00:00"]].copy()
    assert row["HUMIDITY[%]"].iloc[0] == pytest.approx(0.8969444444445012)
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["HUMIDITY[%]_qc"].iloc[0] == QC_VALID


def test_humidity_boundary_values_real_data():
    # Real data happens to contain both boundary values exactly (0.0 and
    # 1.0), so no synthetic case is needed here -- both must stay VALID
    # (the rule is strictly > 1.0 or < 0.0).
    row = pd.DataFrame({"HUMIDITY[%]": [0.0, 1.0]})
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert (row["HUMIDITY[%]_qc"] == QC_VALID).all()


# ---------------------------------------------------------------------------
# WIND_DIR[deg] -- finding #3
# ---------------------------------------------------------------------------

def test_winddir_over_360_real_row(real_60min):
    # Real row: 2018-09-01 00:00:00, WIND_DIR[deg] ~= 639.34 -- the worst
    # offender in the file (103 rows total exceed 360deg).
    row = real_60min.loc[["2018-09-01 00:00:00"]].copy()
    assert row["WIND_DIR[deg]"].iloc[0] == pytest.approx(639.3448180050013)
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["WIND_DIR[deg]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE


def test_winddir_plausible_real_row_stays_valid(real_60min):
    # Real row: 2018-11-17 00:00:00, WIND_DIR[deg] ~= 315.2, a normal bearing.
    row = real_60min.loc[["2018-11-17 00:00:00"]].copy()
    assert row["WIND_DIR[deg]"].iloc[0] == pytest.approx(315.2088888888889)
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["WIND_DIR[deg]_qc"].iloc[0] == QC_VALID


def test_winddir_exact_360_synthetic():
    # SYNTHETIC: no row in either real file lands on exactly 360.0deg.
    # Built to pin down the upper boundary -- 360.0 is invalid (a compass
    # bearing's valid range is [0, 360)), 0.0 is valid and is already
    # covered by a real row above.
    row = pd.DataFrame({"WIND_DIR[deg]": [360.0]})
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["WIND_DIR[deg]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE


def test_winddir_negative_synthetic():
    # SYNTHETIC: no row in either real file has a negative WIND_DIR[deg].
    row = pd.DataFrame({"WIND_DIR[deg]": [-1.0]})
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["WIND_DIR[deg]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE


# ---------------------------------------------------------------------------
# Azimuth[deg] / Elevation[deg] -- finding #4
# ---------------------------------------------------------------------------

def test_azimuth_elevation_zero_real_row_flagged_missing(real_60min):
    # Real row: 2018-11-17 00:00:00 -- part of the 99.9% of rows where these
    # columns sit at exactly 0.0 (i.e. "not really computed for this row").
    row = real_60min.loc[["2018-11-17 00:00:00"]].copy()
    assert row["Azimuth[deg]"].iloc[0] == 0.0
    assert row["Elevation[deg]"].iloc[0] == 0.0
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["Azimuth[deg]_qc"].iloc[0] == QC_MISSING
    assert row["Elevation[deg]_qc"].iloc[0] == QC_MISSING


def test_azimuth_elevation_populated_real_row_stays_valid(real_60min):
    # Real row: 2019-01-16 07:00:00 -- the one calendar day where these
    # columns actually carry computed solar-position values.
    row = real_60min.loc[["2019-01-16 07:00:00"]].copy()
    assert row["Azimuth[deg]"].iloc[0] != 0.0
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["Azimuth[deg]_qc"].iloc[0] == QC_VALID
    # Elevation happens to be 0.0 here too (sun right at the horizon at
    # 07:00 on 2019-01-16 in winter) -- this is the one real row where the
    # "== 0.0 means missing" proxy is a genuine false positive, called out
    # explicitly in QC_SCHEMA.md section 6 rather than papered over.
    assert row["Elevation[deg]"].iloc[0] == 0.0
    assert row["Elevation[deg]_qc"].iloc[0] == QC_MISSING


def test_azimuth_elevation_absent_columns_skipped_cleanly(real_short):
    # SOLETE_short.h5 has no Azimuth[deg]/Elevation[deg] columns at all.
    # apply_qc_flags must skip those rules without raising.
    df = real_short.copy()
    rules = build_raw_value_qc_rules(df)
    rule_columns = {r["column"] for r in rules}
    assert "Azimuth[deg]" not in rule_columns
    assert "Elevation[deg]" not in rule_columns
    apply_qc_flags(df, rules)  # should not raise
    assert "Azimuth[deg]_qc" not in df.columns


# ---------------------------------------------------------------------------
# P_Solar_model_substituted -> P_Solar[kW]_qc -- finding #6
# ---------------------------------------------------------------------------

def test_substitution_flag_true_maps_to_qc_6():
    # Real substitution outcome for 2018-11-17 00:00:00 on the 60min file:
    # Pac == P_Solar[kW] == 0.0, so Pac >= 1.5*P_Solar[kW] evaluates True
    # (0.0 >= 0.0) -- this is one of the 4204 rows PV_Performance_Model
    # marks as substituted (38.33% of the file, matching KNOWN_ISSUES.md).
    row = pd.DataFrame({
        "P_Solar_model_substituted": [True],
        "P_Solar[kW]": [0.0],
    })
    apply_qc_flags(row, [build_substitution_qc_rule()])
    assert row["P_Solar[kW]_qc"].iloc[0] == QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED


def test_substitution_flag_false_maps_to_valid():
    # Real non-substituted outcome for 2018-11-17 06:00:00: Pac and
    # P_Solar[kW] both ~= 0.009004742176371318, so the boolean is False.
    row = pd.DataFrame({
        "P_Solar_model_substituted": [False],
        "P_Solar[kW]": [0.009004742176371318],
    })
    apply_qc_flags(row, [build_substitution_qc_rule()])
    assert row["P_Solar[kW]_qc"].iloc[0] == QC_VALID


# ---------------------------------------------------------------------------
# Whole-file sanity checks -- reproduce Phase 1's exact counts
# ---------------------------------------------------------------------------

def test_whole_file_counts_match_phase1_60min(real_60min):
    df = real_60min.copy()
    n = len(df)
    _, counts = apply_qc_flags(df, build_raw_value_qc_rules(df))
    assert counts["Pressure[mbar]_qc"] == 10477 + 477 + 2
    assert counts["HUMIDITY[%]_qc"] == 188
    assert counts["WIND_DIR[deg]_qc"] == 103
    assert counts["Azimuth[deg]_qc"] == pytest.approx(0.999 * n, abs=2)
    assert counts["Elevation[deg]_qc"] == pytest.approx(0.999 * n, abs=2)


def test_whole_file_counts_near_zero_short(real_short):
    df = real_short.copy()
    _, counts = apply_qc_flags(df, build_raw_value_qc_rules(df))
    # KNOWN_ISSUES.md reports near-zero rates on the short file for all
    # three physically-implausible checks.
    assert counts["Pressure[mbar]_qc"] == 0
    assert counts["HUMIDITY[%]_qc"] == 0
    assert counts["WIND_DIR[deg]_qc"] == 0


# ---------------------------------------------------------------------------
# apply_qc_flags mechanics
# ---------------------------------------------------------------------------

def test_unflagged_cells_default_to_valid():
    row = pd.DataFrame({"HUMIDITY[%]": [0.5]})
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["HUMIDITY[%]_qc"].iloc[0] == QC_VALID


def test_missing_source_column_is_skipped_not_raised():
    df = pd.DataFrame({"SomeOtherColumn": [1, 2, 3]})
    # None of the raw-value rules' source columns exist here.
    result_df, counts = apply_qc_flags(df, build_raw_value_qc_rules(df))
    assert counts == {}
    assert result_df is df
