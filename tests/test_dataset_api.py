# -*- coding: utf-8 -*-
"""
tests/test_dataset_api.py -- SOLETE Phase 5 continuation, Task 5.8

Covers:
    1. Correct shapes from SOLETE(...).forecasting()
    2. Correct QC-flag exclusion, matching metrics.qc_mask()'s documented policy
    3. Exact-reproduction round trip: scoring a persistence baseline through
       this API reproduces Task 5.3's already-computed results/persistence.json
       numbers EXACTLY (not approximately) -- the regression guard that the
       API wrapper didn't subtly change the split or metric behaviour it
       wraps.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics as M
from solete.dataset import SOLETE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_shapes():
    ds = SOLETE(resolution="60min", target="pv_power", split="v1")
    X_train, y_train, X_test, y_test = ds.forecasting(horizon="1h")

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) > 1000  # sanity: most of train survives lag/NaN drop
    assert len(X_test) > 1000
    expected_cols = {f"lag_{l}" for l in (1, 2, 3, 24)} | {
        "GHI[kW1m2]", "POA Irr[kW1m2]", "WIND_SPEED[m1s]", "TEMPERATURE[degC]",
        "hour", "month", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    }
    assert set(X_train.columns) == expected_cols
    assert set(X_test.columns) == expected_cols
    # X and y for the same block share the same index (same target timestamps)
    assert list(X_train.index) == list(y_train.index)
    assert list(X_test.index) == list(y_test.index)


def test_qc_exclusion_matches_qc_mask():
    ds_incl = SOLETE(resolution="60min", target="pv_power", split="v1", qc_exclude=False)
    ds_excl = SOLETE(resolution="60min", target="pv_power", split="v1", qc_exclude=True)

    _, _, X_test_incl, y_test_incl = ds_incl.forecasting(horizon="1h")
    _, _, X_test_excl, y_test_excl = ds_excl.forecasting(horizon="1h")

    # qc_exclude=True must be a strict subset of qc_exclude=False's rows
    assert set(X_test_excl.index).issubset(set(X_test_incl.index))
    assert len(X_test_excl) < len(X_test_incl)

    # Directly check against metrics.qc_mask() on the same rows
    qc_col = ds_incl._df["P_Solar[kW]_qc"].loc(axis=0)[X_test_incl.index]
    keep = M.qc_mask(qc_col)
    expected_kept_index = set(X_test_incl.index[keep.to_numpy()])
    assert set(X_test_excl.index) == expected_kept_index


def test_wind_has_no_qc_column_to_exclude():
    # P_Gaia[kW] has no QC column in this dataset (see bench_common.QC_COLUMNS) --
    # qc_exclude should therefore be a no-op for wind, not an error.
    ds_incl = SOLETE(resolution="60min", target="wind_power", split="v1", qc_exclude=False)
    ds_excl = SOLETE(resolution="60min", target="wind_power", split="v1", qc_exclude=True)
    _, _, X_test_incl, _ = ds_incl.forecasting(horizon="1h")
    _, _, X_test_excl, _ = ds_excl.forecasting(horizon="1h")
    assert list(X_test_incl.index) == list(X_test_excl.index)


def test_persistence_round_trip_matches_task_5_3_exactly():
    """
    Task 5.8.3's regression guard: build the plain-persistence prediction
    (y_hat(t) = y(t-1), i.e. this API's `lag_1` column) from this wrapper's
    test block and confirm the resulting MAE/RMSE/nRMSE match
    results/persistence.json's pv_power numbers EXACTLY.
    """
    results_path = os.path.join(REPO_ROOT, "results", "persistence.json")
    with open(results_path) as f:
        expected = json.load(f)["targets"]["pv_power"]

    capacity = M.installed_capacity_kw("pv_power")

    for qc_exclude, key in [(False, "qc_included"), (True, "qc_excluded")]:
        ds = SOLETE(resolution="60min", target="pv_power", split="v1", qc_exclude=qc_exclude)
        _, _, X_test, y_test = ds.forecasting(horizon="1h")

        y_pred = X_test["lag_1"]  # y(t-1) == plain persistence prediction for y(t)

        exp = expected[key]
        assert len(y_test) == exp["n"], f"{key}: n mismatch {len(y_test)} vs {exp['n']}"

        mae_ = M.mae(y_test, y_pred)
        rmse_ = M.rmse(y_test, y_pred)
        nrmse_cap = M.nrmse(y_test, y_pred, capacity=capacity, method="capacity")
        nrmse_mean = M.nrmse(y_test, y_pred, method="mean")

        assert mae_ == exp["mae"], f"{key}: MAE mismatch {mae_} vs {exp['mae']}"
        assert rmse_ == exp["rmse"], f"{key}: RMSE mismatch {rmse_} vs {exp['rmse']}"
        assert nrmse_cap == exp["nrmse_capacity"], f"{key}: nRMSE(cap) mismatch"
        assert nrmse_mean == exp["nrmse_mean"], f"{key}: nRMSE(mean) mismatch"


if __name__ == "__main__":
    test_shapes()
    print("test_shapes: OK")
    test_qc_exclusion_matches_qc_mask()
    print("test_qc_exclusion_matches_qc_mask: OK")
    test_wind_has_no_qc_column_to_exclude()
    print("test_wind_has_no_qc_column_to_exclude: OK")
    test_persistence_round_trip_matches_task_5_3_exactly()
    print("test_persistence_round_trip_matches_task_5_3_exactly: OK")
