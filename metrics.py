# -*- coding: utf-8 -*-
"""
metrics.py -- SOLETE Phase 5, Task 5.2 (+ Phase 7 Session 6 probabilistic
metrics)

Standard evaluation metrics for the canonical SOLETE forecasting benchmark
(splits/v1.json). Every function accepts an optional QC-flag mask/exclusion
argument so the exclusion logic lives in exactly one place, consistent with
Task 5.1.3's decision: `P_Solar[kW]_qc == 6` (model-substituted) rows are
kept in every split, and it is up to whoever calls these functions to decide
whether to exclude them from a given evaluation.

Point-forecast metrics:
    - mae(y_true, y_pred, mask=None)
    - rmse(y_true, y_pred, mask=None)
    - nrmse(y_true, y_pred, capacity=None, method="capacity", mask=None)
    - skill_score(y_true, y_pred, y_reference, mask=None)

Probabilistic-forecast metrics (Phase 7 Session 6):
    - pinball_loss(y_true, y_pred_quantile, quantile, mask=None)
    - crps_from_quantiles(y_true, quantile_preds, mask=None)
    - interval_coverage(y_true, lower, upper, mask=None)
    - sharpness(lower, upper, mask=None)

QC-mask helpers:
    - qc_mask(qc_column, exclude_flags=(6,))
    - installed_capacity_kw(target)

See BENCHMARKS.md / splits/README.md for how these are used together.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# QC exclusion helpers
# ---------------------------------------------------------------------------

def qc_mask(qc_column, exclude_flags=(6,)):
    """
    Build a boolean "keep this row" mask from a `<column>_qc` Series
    (see QC_SCHEMA.md for the flag value set: 0=valid, 1=missing,
    2=sensor_error, 3=physically_implausible, 4=interpolated,
    5=aggregation_affected_by_gaps, 6=suspected_curtailment_or_model_substituted).

    Parameters
    ----------
    qc_column : pandas.Series
        A `<column>_qc` column, e.g. df['P_Solar[kW]_qc'].
    exclude_flags : iterable of int
        Flag values to exclude. Defaults to (6,) -- the model-substitution
        flag -- matching the exclusion principle already established in
        examples/03_pv_forecasting.ipynb (do not evaluate against rows where
        the "ground truth" is itself a physics-model estimate). Task 5.1's
        canonical split does NOT apply this automatically (it keeps qc==6
        rows in every block); this helper is how a caller opts into excluding
        them for a given evaluation, without needing a second split version.

    Returns
    -------
    pandas.Series of bool, aligned to qc_column's index. True = keep.
    """
    exclude_flags = set(exclude_flags)
    return ~qc_column.isin(exclude_flags)


def installed_capacity_kw(target):
    """
    Installed/rated capacity in kW for a benchmark target, read directly from
    this repo's own PVinfo/WTinfo dicts (Functions.import_PV_WT_data()) rather
    than a hardcoded literal, so it can't silently drift from the physical
    system description used everywhere else in this codebase.

    Parameters
    ----------
    target : str
        "pv_power" / "P_Solar[kW]" -> PV nameplate DC capacity, computed as
            sum over each PV string of Ns * Np * Pmp_stc (module count x
            rated power per module), matching PVinfo's own string layout.
            Currently: string A (Ns=18, Np=2, Pmp_stc=165W) + string B
            (Ns=6, Np=2, Pmp_stc=125W) = 7.44 kW.
        "wind_power" / "P_Gaia[kW]" -> WTinfo['Pn'], the Gaia turbine's
            rated power (11 kW).

    Returns
    -------
    float, capacity in kW.
    """
    from Functions import import_PV_WT_data

    PVinfo, WTinfo = import_PV_WT_data()

    if target in ("pv_power", "P_Solar[kW]", "pv", "solar"):
        pmp = PVinfo["Pmp_stc"]
        ns = PVinfo["Ns"]
        np_ = PVinfo["Np"]
        watts = sum(p * n * m for p, n, m in zip(pmp, ns, np_))
        return watts / 1000.0
    elif target in ("wind_power", "P_Gaia[kW]", "wind", "wt"):
        return float(WTinfo["Pn"])
    else:
        raise ValueError(
            "Unknown target %r -- expected one of "
            "'pv_power'/'P_Solar[kW]' or 'wind_power'/'P_Gaia[kW]'." % (target,)
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _apply_mask(y_true, y_pred, mask):
    """Align y_true/y_pred to numpy arrays, dropping rows where mask is False."""
    y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
    y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred

    if mask is not None:
        mask = pd.Series(mask) if not isinstance(mask, pd.Series) else mask
        # Align on position, not label, if indices don't match (e.g. plain arrays).
        if len(mask) != len(y_true):
            raise ValueError(
                "mask length (%d) does not match y_true length (%d)"
                % (len(mask), len(y_true))
            )
        mask_arr = mask.to_numpy(dtype=bool)
        y_true = y_true.to_numpy()[mask_arr]
        y_pred = y_pred.to_numpy()[mask_arr]
    else:
        y_true = y_true.to_numpy()
        y_pred = y_pred.to_numpy()

    if len(y_true) == 0:
        raise ValueError("No rows left to score after applying mask.")

    return y_true, y_pred


def _apply_mask_pair(a, b, mask):
    """Same alignment/filtering as _apply_mask, for two arrays that aren't
    necessarily (y_true, y_pred) -- e.g. (lower_bound, upper_bound) for the
    interval metrics below. Reuses _apply_mask's logic without implying
    anything about which argument is "truth"."""
    return _apply_mask(a, b, mask)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(y_true, y_pred, mask=None):
    """Mean Absolute Error. Optionally restricted to rows where mask is True."""
    y_true, y_pred = _apply_mask(y_true, y_pred, mask)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred, mask=None):
    """Root Mean Squared Error. Optionally restricted to rows where mask is True."""
    y_true, y_pred = _apply_mask(y_true, y_pred, mask)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true, y_pred, capacity=None, method="capacity", mask=None):
    """
    Normalized RMSE.

    Parameters
    ----------
    capacity : float, optional
        Required when method="capacity". The fixed denominator (e.g. from
        installed_capacity_kw()). Ignored for method in {"mean", "range"}.
    method : {"capacity", "mean", "range"}
        "capacity" (DEFAULT, per maintainer decision): normalize by the
            target's installed/rated capacity (a fixed denominator that does
            not depend on the evaluation window). This is the more common
            convention in solar/wind forecasting papers, and lets nRMSE be
            compared across different time windows or splits, since the
            denominator never changes. Downside: on a window with low
            average output (e.g. winter for PV), nRMSE-by-capacity looks
            worse than nRMSE-by-mean would, even for an equally "good"
            forecast in absolute terms.
        "mean": normalize by the observed y_true mean over the scored rows
            (relative-error style). More forgiving/comparable across sites
            with different capacities, but the denominator changes with the
            evaluation window (e.g. day vs. night, season), which can make
            two nRMSE numbers computed on different slices misleading to
            compare directly.
        "range": normalize by (max(y_true) - min(y_true)) over the scored
            rows. Also window-dependent, and can be dominated by a small
            number of extreme rows.
    mask : array-like of bool, optional
        Same as in rmse()/mae() -- restrict to rows where True.

    Returns
    -------
    float, dimensionless (RMSE / denominator). Multiply by 100 for a percentage.

    Note
    ----
    Per maintainer decision (2026-09-16): BOTH normalizations are supported.
    "capacity" is the default (matches the leaderboard in BENCHMARKS.md
    unless stated otherwise); "mean"/"range" are available for callers who
    want a relative-error reading instead. Always state which one a reported
    number used -- the two are not comparable to each other.
    """
    y_true_arr, y_pred_arr = _apply_mask(y_true, y_pred, mask)
    error_rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

    if method == "capacity":
        if capacity is None or capacity <= 0:
            raise ValueError(
                "nrmse(method='capacity') requires a positive `capacity` "
                "argument, e.g. capacity=installed_capacity_kw('pv_power')."
            )
        denom = capacity
    elif method == "mean":
        denom = float(np.mean(y_true_arr))
    elif method == "range":
        denom = float(np.max(y_true_arr) - np.min(y_true_arr))
    else:
        raise ValueError("method must be one of 'capacity', 'mean', 'range', got %r" % (method,))

    if denom == 0:
        raise ValueError(
            "nrmse denominator is zero (method=%r) -- cannot normalize. "
            "This can happen with method='mean'/'range' on an all-zero "
            "window (e.g. wind power outside the two active days, see "
            "splits/README.md)." % (method,)
        )

    return error_rmse / denom


def skill_score(y_true, y_pred, y_reference, mask=None):
    """
    Skill score of a forecast vs. a reference (naive persistence) baseline,
    following the standard forecast-skill convention:

        skill = 1 - RMSE(y_pred) / RMSE(y_reference)

    skill > 0 means the model beats the reference baseline (1.0 = perfect,
    0.0 = exactly as good as the reference, negative = worse than the
    reference). y_reference should be the reference/baseline model's
    predictions over the SAME rows as y_pred (e.g. Task 5.3's plain
    persistence forecast), not the raw observed series.

    Parameters
    ----------
    y_true : array-like
        Observed ground truth.
    y_pred : array-like
        The model's forecast being evaluated.
    y_reference : array-like
        The reference/baseline forecast (same length/alignment as y_pred).
    mask : array-like of bool, optional
        Restrict to rows where True, applied identically to y_pred and
        y_reference against the same y_true.
    """
    rmse_model = rmse(y_true, y_pred, mask=mask)
    rmse_reference = rmse(y_true, y_reference, mask=mask)
    if rmse_reference == 0:
        raise ValueError(
            "Reference RMSE is zero -- skill score is undefined (division by "
            "zero). This can happen on a degenerate all-zero window."
        )
    return 1.0 - (rmse_model / rmse_reference)


# ---------------------------------------------------------------------------
# Probabilistic-forecast metrics (Phase 7 Session 6)
# ---------------------------------------------------------------------------

def pinball_loss(y_true, y_pred, quantile, mask=None):
    """
    Pinball (quantile) loss at a single quantile level.

        L_q(y, q_hat) = q * (y - q_hat)      if y >= q_hat
                      = (q - 1) * (y - q_hat) if y <  q_hat

    Lower is better; 0 is a perfect quantile forecast. At quantile=0.5 this
    is exactly half of mae() (the median-optimal loss) -- a useful sanity
    check when validating a new quantile model.

    Parameters
    ----------
    y_true : array-like
        Observed ground truth.
    y_pred : array-like
        Forecast for THIS quantile level only (not a matrix of quantiles --
        call once per level, same convention as crps_from_quantiles below).
    quantile : float
        Quantile level in (0, 1), e.g. 0.05 for the 5th percentile.
    mask : array-like of bool, optional
        Same convention as mae()/rmse() -- restrict to rows where True.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1), got %r" % (quantile,))
    y_true_arr, y_pred_arr = _apply_mask(y_true, y_pred, mask)
    diff = y_true_arr - y_pred_arr
    loss = np.where(diff >= 0, quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


def crps_from_quantiles(y_true, quantile_preds, mask=None):
    """
    Approximate the Continuous Ranked Probability Score (CRPS) from a
    discrete set of quantile forecasts, using the identity

        CRPS(F, y) = 2 * integral_0^1  QS_tau(y, F^-1(tau))  d(tau)

    (the quantile-score / pinball-loss decomposition of CRPS -- see
    Gneiting & Raftery 2007, "Strictly Proper Scoring Rules...", eq. 21;
    also Laio & Tamea 2007). The integral is approximated here via the
    trapezoidal rule over whatever quantile levels are supplied, so it
    handles an uneven grid (e.g. denser near the tails) correctly, not just
    an evenly-spaced one.

    Caveat: this is an APPROXIMATION, not the exact CRPS, for two reasons:
    (1) the trapezoidal rule is only exact for a piecewise-linear pinball-
    loss-vs-tau curve, and (2) the integral outside [min(levels), max(levels)]
    is not covered at all -- e.g. with levels 0.05..0.95 the tails beyond the
    5th/95th percentile contribute nothing here, which slightly UNDERSTATES
    the true CRPS. The narrower the outermost levels are to 0/1 and the
    denser the grid, the closer this gets to the exact value.

    Parameters
    ----------
    y_true : array-like
        Observed ground truth.
    quantile_preds : dict[float, array-like]
        Mapping quantile level -> predicted value at that level, each
        array aligned the same way as y_true. Needs at least 2 levels.
    mask : array-like of bool, optional
        Applied identically to y_true and every quantile's predictions.
    """
    levels = sorted(quantile_preds.keys())
    if len(levels) < 2:
        raise ValueError(
            "crps_from_quantiles needs at least 2 quantile levels to "
            "integrate over, got %d." % (len(levels),)
        )
    pinballs = np.array(
        [pinball_loss(y_true, quantile_preds[q], q, mask=mask) for q in levels]
    )
    integral = np.trapezoid(pinballs, np.array(levels))
    return float(2.0 * integral)


def interval_coverage(y_true, lower, upper, mask=None):
    """
    Empirical coverage of a prediction interval: the fraction of observed
    values that fall within [lower, upper] (inclusive of both bounds).

    Compare against the interval's NOMINAL coverage (e.g. an interval built
    from the 5th/95th percentile forecasts has nominal coverage 0.90) to
    check calibration -- this function only computes the empirical number;
    it doesn't know what the nominal level was supposed to be.

    Parameters
    ----------
    y_true, lower, upper : array-like
        lower/upper are this row's predicted interval bounds; lower should
        be <= upper elementwise (not checked here -- a quantile-crossing
        lower > upper would just report an empirical coverage of 0 for that
        row, which is arguably the honest answer for a broken interval).
    mask : array-like of bool, optional
        Applied identically to y_true, lower, and upper.
    """
    y_true_arr, lower_arr = _apply_mask(y_true, lower, mask)
    _, upper_arr = _apply_mask(y_true, upper, mask)
    within = (y_true_arr >= lower_arr) & (y_true_arr <= upper_arr)
    return float(np.mean(within))


def sharpness(lower, upper, mask=None):
    """
    Mean prediction-interval width (upper - lower). Narrower is sharper
    (more informative) -- but sharpness on its own is meaningless without
    interval_coverage() alongside it: an interval that's narrow because it's
    also wrong (poor coverage) is not a good interval, just a confident one.

    Parameters
    ----------
    lower, upper : array-like
        Predicted interval bounds, same alignment convention as elsewhere.
    mask : array-like of bool, optional
    """
    lower_arr, upper_arr = _apply_mask_pair(lower, upper, mask)
    return float(np.mean(upper_arr - lower_arr))
