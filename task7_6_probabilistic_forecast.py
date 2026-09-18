# -*- coding: utf-8 -*-
"""
task7_6_probabilistic_forecast.py -- SOLETE Phase 7, Session 6

Probabilistic (quantile/interval) forecasting benchmark, extending the
Phase 5/6 point-forecast infrastructure (bench_common.py, metrics.py,
splits/v1.json) rather than duplicating it.

ASK FIRST outcome (maintainer, 2026-09-18): quantile regression, via
LightGBM's native `objective="quantile"` -- the same library
baseline_gbm.py (Task 5.5) already chose, not a second gradient-boosting
library, per the session prompt's instruction. Reuses baseline_gbm.py's own
feature set (build_features, MET_COLS, LAGS) and hyperparameter grid
unchanged, for direct parity with the point-forecast GBM row in
BENCHMARKS.md -- the only thing that changes per quantile level is the
model's objective/alpha and which metric (pinball loss instead of RMSE)
selects the best hyperparameters on val.

ASK FIRST outcome (maintainer, 2026-09-18) -- wind: attempt it anyway,
despite the same P_Gaia[kW] near-total zero-degeneracy caveat that already
applies to every wind row in BENCHMARKS.md (KNOWN_ISSUES.md #10) --
maintainer's reasoning: better wind data may arrive later (see
KNOWN_ISSUES.md #10's note about pursuing raw/finer-resolution data), and
this session's job is to have a WORKING probabilistic method ready to
re-point at that data when it lands, not to produce a meaningful wind
result today. Every wind number in results/probabilistic_wind.json and
BENCHMARKS.md's wind section carries this caveat explicitly -- read the
numbers as "the pipeline runs end-to-end," not "wind is forecastable here."

Quantile levels: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95] -- gives three
nested nominal intervals (90%, 80%, 50%) plus the median, a standard grid in
probabilistic-forecasting benchmarks (e.g. GEFCom-style). CRPS is
approximated from this same 7-point grid via metrics.crps_from_quantiles's
trapezoidal integration (see that function's docstring for the
under-the-tails caveat this implies).

Fit on train, hyperparameters selected per-quantile on val (by pinball loss
at that quantile), scored on test via bench_common.py's split boundaries and
QC-exclusion convention (both qc_included/qc_excluded reported for PV; wind
has no QC column, same as every other wind row in this benchmark).
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import bench_common as bc
import metrics as M
import baseline_gbm as gbm  # reuse Task 5.5's feature set, not a second one

QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
INTERVALS = {
    "interval_90": (0.05, 0.95),
    "interval_80": (0.10, 0.90),
    "interval_50": (0.25, 0.75),
}


def fit_and_predict_quantile(df, train, val, col, quantile):
    """Same structure as baseline_gbm.fit_and_predict, generalized to a
    single quantile level: LightGBM's quantile objective at alpha=quantile,
    hyperparameters selected on val by pinball loss at that same quantile
    (not RMSE -- RMSE is the wrong criterion for picking a quantile model)."""
    X_full = gbm.build_features(df, col)
    y_full = df[col]

    X_train = X_full.loc[train.index].dropna()
    y_train = y_full.loc[X_train.index]
    X_val = X_full.loc[val.index].dropna()
    y_val = y_full.loc[X_val.index]

    best_pinball, best_model, best_params = np.inf, None, None
    for params in gbm.HYPERPARAM_GRID:
        model = LGBMRegressor(
            objective="quantile", alpha=quantile, random_state=0, verbosity=-1, **params
        )
        model.fit(X_train.values, y_train.values)
        pred_val = model.predict(X_val.values)
        val_pinball = M.pinball_loss(y_val.values, pred_val, quantile)
        if val_pinball < best_pinball:
            best_pinball, best_model, best_params = val_pinball, model, params

    valid_idx = X_full.dropna().index
    pred_full = pd.Series(np.nan, index=df.index)
    pred_full.loc[valid_idx] = best_model.predict(X_full.loc[valid_idx].values)

    return pred_full, best_params, best_pinball


def _score_probabilistic(y_true_block, quantile_preds_block, bool_mask=None):
    """Score one (optionally QC-masked) slice: pinball loss per quantile
    level, CRPS from the full quantile set, and coverage/sharpness for each
    nested interval in INTERVALS. Mirrors bench_common.score_block's
    qc_included/qc_excluded shape but for the probabilistic metrics."""
    if bool_mask is not None:
        y = y_true_block[bool_mask]
        preds = {q: p[bool_mask] for q, p in quantile_preds_block.items()}
        n = int(bool_mask.sum())
    else:
        y = y_true_block
        preds = quantile_preds_block
        n = int(len(y_true_block))

    out = {
        "n": n,
        "pinball_loss": {str(q): M.pinball_loss(y, preds[q], q) for q in QUANTILE_LEVELS},
        "crps": M.crps_from_quantiles(y, preds),
        "intervals": {},
    }
    for name, (lo_q, hi_q) in INTERVALS.items():
        lower, upper = preds[lo_q], preds[hi_q]
        out["intervals"][name] = {
            "nominal_coverage": round(hi_q - lo_q, 2),
            "empirical_coverage": M.interval_coverage(y, lower, upper),
            "sharpness": M.sharpness(lower, upper),
        }
    return out


def score_target_on_test(df_full, target_key, quantile_preds_full, test_index):
    y_true_full = df_full[bc.TARGETS[target_key]]
    y_true_block = y_true_full.loc[test_index].to_numpy()
    preds_block = {q: quantile_preds_full[q].loc[test_index].to_numpy() for q in QUANTILE_LEVELS}

    included = _score_probabilistic(y_true_block, preds_block)

    excluded = None
    if target_key in bc.QC_COLUMNS:
        qc_col_full = df_full[bc.QC_COLUMNS[target_key]]
        qc_block = qc_col_full.loc[test_index]
        keep_mask = M.qc_mask(qc_block, exclude_flags=(6,)).to_numpy()
        if keep_mask.sum() > 0:
            excluded = _score_probabilistic(y_true_block, preds_block, bool_mask=keep_mask)

    return {"qc_included": included, "qc_excluded": excluded}


def run_target(df, train, val, test, target_key):
    col = bc.TARGETS[target_key]
    quantile_preds = {}
    chosen_params = {}
    val_pinball_at_chosen = {}

    for q in QUANTILE_LEVELS:
        pred, params, val_pinball = fit_and_predict_quantile(df, train, val, col, q)
        quantile_preds[q] = pred
        chosen_params[str(q)] = params
        val_pinball_at_chosen[str(q)] = val_pinball

    scored = score_target_on_test(df, target_key, quantile_preds, test.index)
    scored["chosen_hyperparameters_per_quantile"] = chosen_params
    scored["val_pinball_loss_at_chosen_params"] = val_pinball_at_chosen
    return scored


def run():
    df = bc.load_full_df()
    train, val, test = bc.split_df(df)

    shared_meta = {
        "model": "probabilistic_gbm",
        "library": "lightgbm",
        "description": (
            "LightGBM quantile regression (objective='quantile'), one model per "
            "quantile level, on the same lagged-target + same-timestamp-weather + "
            "calendar feature set as baseline_gbm.py (Task 5.5) -- see that file's "
            "own same-timestamp-weather caveat, which applies identically here."
        ),
        "split_version": "v1",
        "horizon": "1 step (1h)",
        "fitting": "fit on train; hyperparameters selected per-quantile on val by pinball loss",
        "quantile_levels": QUANTILE_LEVELS,
        "crps_note": (
            "Approximated via metrics.crps_from_quantiles's trapezoidal integration "
            "over the quantile_levels grid above -- see that function's docstring "
            "for the under-the-tails caveat this implies."
        ),
        "feature_set": {
            "lags": gbm.LAGS,
            "meteorological_same_timestamp": gbm.MET_COLS,
            "calendar": ["hour", "month", "hour_sin", "hour_cos", "doy_sin", "doy_cos"],
        },
    }

    pv_payload = dict(shared_meta)
    pv_payload["target"] = "pv_power"
    pv_payload["result"] = run_target(df, train, val, test, "pv_power")
    pv_path = bc.save_result("probabilistic_pv", pv_payload)
    print("Saved:", pv_path)

    wind_payload = dict(shared_meta)
    wind_payload["target"] = "wind_power"
    wind_payload["caveat"] = bc.WIND_CAVEAT + (
        " ASK FIRST outcome (maintainer, 2026-09-18): this wind result was produced "
        "anyway, specifically so a working probabilistic pipeline exists to re-point "
        "at better wind data later (KNOWN_ISSUES.md #10) -- it is not offered as a "
        "meaningful forecast-skill result today."
    )
    wind_payload["result"] = run_target(df, train, val, test, "wind_power")
    wind_path = bc.save_result("probabilistic_wind", wind_payload)
    print("Saved:", wind_path)

    return pv_payload, wind_payload


if __name__ == "__main__":
    import json

    pv_payload, wind_payload = run()
    print(json.dumps({"pv_power": pv_payload["result"], "wind_power": wind_payload["result"]}, indent=2))
