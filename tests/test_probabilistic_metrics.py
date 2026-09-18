# -*- coding: utf-8 -*-
"""
Tests for the probabilistic-forecast metrics added to metrics.py in Phase 7
Session 6 (pinball_loss, crps_from_quantiles, interval_coverage, sharpness).

Same rigor/convention as tests/test_metrics.py: a real-data-derived case
first (an actual observed PV window from SOLETE_Pombo_60min.h5, with a
quantile forecast built from that same window's persistence-residual
quantiles -- not a model, but not fabricated numbers either), then a couple
of hand-built synthetic edge cases (perfect forecast, invalid inputs) that
real data can't demonstrate on demand. Expected values for the real-data
case are computed independently in this file with plain numpy, so a shared
bug between metrics.py and this file wouldn't be caught by reusing the same
code path.

Run with: pytest tests/test_probabilistic_metrics.py -v  (from the repo root)
"""

import sys
import types
import pathlib

# Same convention as tests/test_metrics.py / tests/test_qc_flags.py: stub out
# keras/tensorflow so this file has no dependency on those heavy, unrelated
# packages (Functions.py imports them at module level; metrics.py only needs
# Functions.import_PV_WT_data(), which never touches keras).
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

from metrics import mae, pinball_loss, crps_from_quantiles, interval_coverage, sharpness

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_FILE = REPO_ROOT / "SOLETE_Pombo_60min.h5"


# ---------------------------------------------------------------------------
# Real-data-derived case
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_quantile_forecast():
    """
    Same 72-row real PV window used in test_metrics.py
    (2019-06-01 -> 2019-06-03), turned into a quantile forecast without a
    trained model: point forecast = plain persistence (yhat(t) = y(t-1)),
    quantile spread = that same window's own persistence-residual quantiles
    added on top. This is a real, self-consistent quantile forecast built
    from real data -- not the actual Session 6 LightGBM model (that's
    task7_6_probabilistic_forecast.py's job) -- good enough to exercise the
    metrics against real numbers rather than invented ones.
    """
    df = pd.read_hdf(REAL_FILE).sort_index()  # KNOWN_ISSUES.md #8: must sort
    window = df.loc["2019-06-01":"2019-06-03", "P_Solar[kW]"]
    assert len(window) == 72

    y_true = window.to_numpy()[1:]
    point_pred = window.to_numpy()[:-1]  # plain persistence
    residuals = y_true - point_pred

    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantile_preds = {
        q: point_pred + np.quantile(residuals, q) for q in levels
    }
    return y_true, quantile_preds, levels


def test_pinball_loss_at_median_equals_half_mae(real_quantile_forecast):
    # Mathematical property: pinball loss at quantile=0.5 is exactly half of
    # MAE, regardless of the data -- a useful sanity check independent of
    # metrics.py's own mae() implementation.
    y_true, quantile_preds, _ = real_quantile_forecast
    q50_pred = quantile_preds[0.5]
    expected = 0.5 * float(np.mean(np.abs(y_true - q50_pred)))
    assert pinball_loss(y_true, q50_pred, 0.5) == pytest.approx(expected)
    assert pinball_loss(y_true, q50_pred, 0.5) == pytest.approx(0.5 * mae(y_true, q50_pred))


def test_pinball_loss_matches_independent_calc_on_real_data(real_quantile_forecast):
    y_true, quantile_preds, _ = real_quantile_forecast
    q = 0.25
    y_pred = quantile_preds[q]
    diff = y_true - y_pred
    expected = float(np.mean(np.where(diff >= 0, q * diff, (q - 1.0) * diff)))
    assert pinball_loss(y_true, y_pred, q) == pytest.approx(expected)


def test_crps_from_quantiles_positive_and_below_mae_scale(real_quantile_forecast):
    # No real data can pin down an exact "correct" CRPS value without
    # reimplementing the same trapezoidal approximation (that would just be
    # testing the test), so this checks the properties CRPS must have
    # rather than a specific number: strictly positive on real, imperfect
    # data, and roughly the same order of magnitude as MAE (CRPS and MAE are
    # both in the target's units and CRPS collapses to MAE for a
    # deterministic/degenerate forecast).
    y_true, quantile_preds, _ = real_quantile_forecast
    crps = crps_from_quantiles(y_true, quantile_preds)
    assert crps > 0
    assert crps < 5 * mae(y_true, quantile_preds[0.5])


def test_crps_needs_at_least_two_levels():
    with pytest.raises(ValueError):
        crps_from_quantiles(np.array([1.0, 2.0]), {0.5: np.array([1.0, 2.0])})


def test_pinball_loss_rejects_quantile_outside_open_unit_interval():
    with pytest.raises(ValueError):
        pinball_loss(np.array([1.0]), np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        pinball_loss(np.array([1.0]), np.array([1.0]), 1.0)


def test_interval_coverage_and_sharpness_widen_together_on_real_data(real_quantile_forecast):
    # Nested intervals built from the same real residual quantiles: the 90%
    # interval (q05/q95) must be at least as wide (sharpness) and at least
    # as covering (interval_coverage) as the 50% interval (q25/q75) nested
    # inside it -- both are monotonic in interval width by construction.
    y_true, quantile_preds, _ = real_quantile_forecast
    lower_90, upper_90 = quantile_preds[0.05], quantile_preds[0.95]
    lower_50, upper_50 = quantile_preds[0.25], quantile_preds[0.75]

    sharp_90 = sharpness(lower_90, upper_90)
    sharp_50 = sharpness(lower_50, upper_50)
    assert sharp_90 > sharp_50 > 0

    cov_90 = interval_coverage(y_true, lower_90, upper_90)
    cov_50 = interval_coverage(y_true, lower_50, upper_50)
    assert 0.0 <= cov_50 <= cov_90 <= 1.0


def test_interval_coverage_matches_independent_calc_on_real_data(real_quantile_forecast):
    y_true, quantile_preds, _ = real_quantile_forecast
    lower, upper = quantile_preds[0.05], quantile_preds[0.95]
    expected = float(np.mean((y_true >= lower) & (y_true <= upper)))
    assert interval_coverage(y_true, lower, upper) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Synthetic edge cases -- real data can't demonstrate a perfect/degenerate
# forecast on demand, same rationale as test_metrics.py's all-zero-window case
# ---------------------------------------------------------------------------

def test_perfect_quantile_forecast_synthetic():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Every quantile level predicts the true value exactly -- a degenerate
    # but valid "perfectly confident and always right" forecast.
    quantile_preds = {q: y_true.copy() for q in (0.1, 0.5, 0.9)}

    for q, pred in quantile_preds.items():
        assert pinball_loss(y_true, pred, q) == pytest.approx(0.0)
    assert crps_from_quantiles(y_true, quantile_preds) == pytest.approx(0.0)
    assert interval_coverage(y_true, quantile_preds[0.1], quantile_preds[0.9]) == pytest.approx(1.0)
    assert sharpness(quantile_preds[0.1], quantile_preds[0.9]) == pytest.approx(0.0)


def test_interval_coverage_zero_when_interval_never_contains_truth_synthetic():
    y_true = np.array([10.0, 10.0, 10.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])  # true value is always outside [0, 1]
    assert interval_coverage(y_true, lower, upper) == pytest.approx(0.0)


def test_sharpness_matches_independent_calc_synthetic():
    lower = np.array([1.0, 2.0, 3.0])
    upper = np.array([4.0, 6.0, 3.5])
    expected = float(np.mean(upper - lower))
    assert sharpness(lower, upper) == pytest.approx(expected)


def test_pinball_loss_mask_matches_manual_filter_on_real_data(real_quantile_forecast):
    y_true, quantile_preds, _ = real_quantile_forecast
    q = 0.75
    y_pred = quantile_preds[q]
    mask = np.arange(len(y_true)) % 2 == 0  # keep every other row

    expected = pinball_loss(y_true[mask], y_pred[mask], q)
    assert pinball_loss(y_true, y_pred, q, mask=mask) == pytest.approx(expected)
