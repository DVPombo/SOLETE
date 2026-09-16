# -*- coding: utf-8 -*-
"""
Tests for metrics.py (Phase 5, Task 5.2).

Real-data cases pull an actual observed slice out of SOLETE_Pombo_60min.h5 and
build a forecast-like array from it (a plain persistence shift, which is
exactly what Task 5.3's baseline does) rather than fabricating both arrays
from scratch. Expected values are computed independently in this file with
plain numpy, so a bug shared between metrics.py and this test wouldn't be
caught by having the two use the same code path.

A couple of hand-computed synthetic edge cases (perfect forecast -> 0 error;
all-zero window, matching the real wind-power situation documented in
splits/README.md) are included too, since real data can't cleanly demonstrate
a division-by-zero / degenerate-target case on demand.

Run with: pytest tests/test_metrics.py -v  (from the repo root)
"""

import sys
import types
import pathlib

# Functions.py imports keras/tensorflow at module level (for the ML
# forecasting code) purely so metrics.py's installed_capacity_kw() can call
# Functions.import_PV_WT_data(). These tests never touch the ML code, so stub
# keras/tensorflow out -- same convention as tests/test_qc_flags.py -- to keep
# this test file's dependency footprint light.
for _modname in ["keras", "keras.models", "keras.layers"]:
    sys.modules.setdefault(_modname, types.ModuleType(_modname))
sys.modules["keras.models"].Sequential = object
sys.modules["keras.models"].load_model = lambda *a, **k: None
for _n in ["LSTM", "Dense", "Masking", "Flatten", "Conv1D", "MaxPooling1D"]:
    setattr(sys.modules["keras.layers"], _n, object)

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from metrics import mae, rmse, nrmse, skill_score, qc_mask, installed_capacity_kw

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_FILE = REPO_ROOT / "SOLETE_Pombo_60min.h5"


# ---------------------------------------------------------------------------
# Real-data-derived cases
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_pv_window():
    """
    72 consecutive real hourly P_Solar[kW] rows from the test split
    (2019-06-01 -> 2019-06-03, well inside splits/v1.json's test block,
    2019-05-01 to 2019-09-01). Not synthetic -- pulled directly from
    SOLETE_Pombo_60min.h5.
    """
    df = pd.read_hdf(REAL_FILE).sort_index()  # KNOWN_ISSUES.md #8: must sort
    window = df.loc["2019-06-01":"2019-06-03", "P_Solar[kW]"]
    assert len(window) == 72
    return window


def test_mae_matches_independent_calc_on_real_persistence_forecast(real_pv_window):
    y_true = real_pv_window.to_numpy()[1:]        # skip first row (no t-1)
    y_pred = real_pv_window.to_numpy()[:-1]        # plain persistence: yhat(t) = y(t-1)

    expected_mae = float(np.mean(np.abs(y_true - y_pred)))
    assert expected_mae > 0  # sanity: persistence is not perfect on real PV data
    assert mae(y_true, y_pred) == pytest.approx(expected_mae)


def test_rmse_matches_independent_calc_on_real_persistence_forecast(real_pv_window):
    y_true = real_pv_window.to_numpy()[1:]
    y_pred = real_pv_window.to_numpy()[:-1]

    expected_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse)


def test_rmse_ge_mae_on_real_data(real_pv_window):
    # Mathematical property (Cauchy-Schwarz / power-mean inequality):
    # RMSE >= MAE always. Cheap real-data sanity check that both are
    # computing genuinely different things, not the same formula twice.
    y_true = real_pv_window.to_numpy()[1:]
    y_pred = real_pv_window.to_numpy()[:-1]
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


def test_nrmse_capacity_method_matches_manual_calc(real_pv_window):
    y_true = real_pv_window.to_numpy()[1:]
    y_pred = real_pv_window.to_numpy()[:-1]

    capacity = installed_capacity_kw("pv_power")
    assert capacity == pytest.approx(7.44, abs=0.01)  # 18*2*165W + 6*2*125W, see metrics.py

    expected_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    expected_nrmse = expected_rmse / capacity

    assert nrmse(y_true, y_pred, capacity=capacity, method="capacity") == pytest.approx(expected_nrmse)


def test_nrmse_mean_vs_capacity_methods_disagree_on_real_data(real_pv_window):
    # The two normalizations are NOT interchangeable -- this is exactly why
    # Task 5.2 required a maintainer decision on which one is the default
    # (see metrics.py's nrmse() docstring). Confirms on real data that they
    # actually produce different numbers, not just different code paths for
    # the same result.
    y_true = real_pv_window.to_numpy()[1:]
    y_pred = real_pv_window.to_numpy()[:-1]

    capacity = installed_capacity_kw("pv_power")
    by_capacity = nrmse(y_true, y_pred, capacity=capacity, method="capacity")
    by_mean = nrmse(y_true, y_pred, method="mean")

    assert by_capacity != pytest.approx(by_mean)


def test_skill_score_real_persistence_vs_itself_is_zero(real_pv_window):
    # A model that IS the reference baseline should score exactly 0 skill
    # (neither better nor worse than itself) -- checked on the real window,
    # not a synthetic one.
    y_true = real_pv_window.to_numpy()[1:]
    y_persistence = real_pv_window.to_numpy()[:-1]

    assert skill_score(y_true, y_persistence, y_persistence) == pytest.approx(0.0, abs=1e-10)


def test_qc_mask_excludes_flagged_rows_on_real_substituted_column():
    # Pull real P_Solar[kW]_qc values via the real Build pipeline (small
    # slice only, to keep this test fast) and confirm qc_mask's default
    # (exclude flag 6, model-substituted) actually drops those rows.
    sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
    from Functions import import_SOLETE_data, import_PV_WT_data

    Control_Var = {
        "resolution": "60min",
        "SOLETE_builvsimport": "Build",
        "SOLETE_save": False,
        "OriginalFeatures": [],
        "PossibleFeatures": [],
    }
    PVinfo, WTinfo = import_PV_WT_data()
    df = import_SOLETE_data(Control_Var, PVinfo, WTinfo).sort_index()
    window = df.loc["2019-06-01":"2019-06-03", "P_Solar[kW]_qc"]

    assert (window == 6).any(), "expected at least one model-substituted row in this window"

    mask = qc_mask(window)
    assert mask.sum() == (window != 6).sum()
    assert not mask[window == 6].any()


# ---------------------------------------------------------------------------
# Synthetic edge cases (real data can't demonstrate these cleanly)
# ---------------------------------------------------------------------------

def test_perfect_forecast_gives_zero_mae_and_rmse():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    assert mae(y_true, y_pred) == 0.0
    assert rmse(y_true, y_pred) == 0.0


def test_hand_computed_mae_rmse():
    # y_true - y_pred = [1, -2, 3] -> |errors| = [1, 2, 3], errors^2 = [1, 4, 9]
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 4.0, 0.0])
    assert mae(y_true, y_pred) == pytest.approx((1 + 2 + 3) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((1 + 4 + 9) / 3))


def test_mask_argument_restricts_rows():
    y_true = np.array([1.0, 100.0, 2.0, 100.0])
    y_pred = np.array([1.0, 0.0, 2.0, 0.0])  # perfect on rows 0,2; way off on rows 1,3
    mask = np.array([True, False, True, False])
    assert mae(y_true, y_pred, mask=mask) == 0.0
    assert rmse(y_true, y_pred, mask=mask) == 0.0


def test_nrmse_requires_capacity_for_capacity_method():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 2.5])
    with pytest.raises(ValueError):
        nrmse(y_true, y_pred, method="capacity")  # no capacity given


def test_nrmse_zero_denominator_raises_on_degenerate_window():
    # Mirrors the real all-zero-wind-power situation documented in
    # splits/README.md: outside the two active days, P_Gaia[kW] is exactly
    # zero, so method="mean"/"range" would divide by zero. Confirms this
    # fails loudly instead of silently returning inf/nan.
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.1])
    with pytest.raises(ValueError):
        nrmse(y_true, y_pred, method="mean")
    with pytest.raises(ValueError):
        nrmse(y_true, y_pred, method="range")


def test_skill_score_hand_computed():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([10.0, 10.0])          # perfect model, RMSE = 0
    y_reference = np.array([8.0, 8.0])        # reference off by 2 every time, RMSE = 2
    assert skill_score(y_true, y_pred, y_reference) == pytest.approx(1.0)


def test_skill_score_negative_when_model_worse_than_reference():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([4.0, 4.0])             # model off by 6, RMSE = 6
    y_reference = np.array([8.0, 8.0])         # reference off by 2, RMSE = 2
    assert skill_score(y_true, y_pred, y_reference) == pytest.approx(1.0 - 6.0 / 2.0)  # -2.0


def test_installed_capacity_kw_known_values():
    assert installed_capacity_kw("pv_power") == pytest.approx(7.44, abs=0.01)
    assert installed_capacity_kw("wind_power") == pytest.approx(11.0)


def test_installed_capacity_kw_rejects_unknown_target():
    with pytest.raises(ValueError):
        installed_capacity_kw("not_a_real_target")
