# -*- coding: utf-8 -*-
"""
baseline_climatology_ar.py -- SOLETE Phase 5 continuation, Task 5.4

Climatology
-----------
Hour-of-day average computed from `train` only, applied to `test`.

Design note (train does NOT span a full annual cycle): `train` runs
2018-06-01 -> 2019-03-31, so May/July/August/September (all of which are in
`test`) have ZERO days represented in `train`. A month-of-year (or
day-of-year) climatology therefore cannot be estimated for those months from
train data alone -- there is nothing to average. Hour-of-day-only
climatology (averaging every occurrence of a given hour across all of
train, regardless of month) is used instead: it is estimable for every hour
with train data covering it, and does not silently extrapolate a seasonal
pattern the training window never observed. This is a real limitation of
this baseline on this split, documented here rather than hidden -- a
richer climatology (e.g. hour-of-day x day-of-year with circular smoothing)
would still be interpolating across a seasonal gap the training data does
not cover, not genuinely observing it.

AR (autoregressive, lagged target only)
----------------------------------------
Ordinary least squares on y(t) ~ y(t-1), y(t-2), ..., y(t-p), no exogenous
features. Lag order p is tuned on `val` (grid search over a small range),
fit on `train`, scored on `test`. No intercept-free trick -- plain
LinearRegression with intercept.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import bench_common as bc

LAG_GRID = [1, 2, 3, 6, 12, 24, 48]


def climatology_predict(df_full, train, col):
    """Hour-of-day average from `train`, broadcast back onto df_full's index."""
    hourly_mean = train[col].groupby(train.index.hour).mean()
    pred = df_full.index.hour.map(hourly_mean)
    return pd.Series(pred, index=df_full.index, dtype=float)


def _build_lag_matrix(series, lags):
    X = pd.concat({f"lag_{l}": series.shift(l) for l in lags}, axis=1)
    return X


def ar_fit_and_predict(df_full, train, val, test, col):
    best_p, best_val_rmse, best_model = None, np.inf, None

    for p in LAG_GRID:
        lags = list(range(1, p + 1))
        X_full = _build_lag_matrix(df_full[col], lags)
        y_full = df_full[col]

        X_train = X_full.loc[train.index].dropna()
        y_train = y_full.loc[X_train.index]
        if len(X_train) < 50:
            continue

        model = LinearRegression()
        model.fit(X_train.values, y_train.values)

        X_val = X_full.loc[val.index].dropna()
        y_val = y_full.loc[X_val.index]
        if len(X_val) == 0:
            continue
        val_pred = model.predict(X_val.values)
        val_rmse = float(np.sqrt(np.mean((y_val.values - val_pred) ** 2)))

        if val_rmse < best_val_rmse:
            best_val_rmse, best_p, best_model = val_rmse, p, model

    lags = list(range(1, best_p + 1))
    X_full = _build_lag_matrix(df_full[col], lags)
    pred_full = pd.Series(np.nan, index=df_full.index)
    valid_idx = X_full.dropna().index
    pred_full.loc[valid_idx] = best_model.predict(X_full.loc[valid_idx].values)

    return pred_full, best_p, best_val_rmse


def run():
    df = bc.load_full_df()
    train, val, test = bc.split_df(df)

    results = {}

    # --- Climatology ---
    payload = {
        "model": "climatology",
        "description": "Hour-of-day average from train (see module docstring for why month-of-year is not used)",
        "split_version": "v1",
        "horizon": "1 step (1h)",
        "fitting": "train only (hour-of-day means)",
        "targets": {},
    }
    for target_key, col in bc.TARGETS.items():
        pred = climatology_predict(df, train, col)
        scored = bc.score_target_on_test(df, target_key, pred, test.index)
        payload["targets"][target_key] = scored
        if target_key == "wind_power":
            payload["targets"][target_key]["caveat"] = bc.WIND_CAVEAT
    path = bc.save_result("climatology", payload)
    results["climatology"] = payload
    print("Saved:", path)

    # --- AR ---
    payload = {
        "model": "ar",
        "description": "OLS autoregression on lagged target only (no exogenous features); lag order p tuned on val",
        "split_version": "v1",
        "horizon": "1 step (1h)",
        "fitting": "OLS fit on train, lag order p selected on val by RMSE",
        "targets": {},
    }
    for target_key, col in bc.TARGETS.items():
        pred, best_p, best_val_rmse = ar_fit_and_predict(df, train, val, test, col)
        scored = bc.score_target_on_test(df, target_key, pred, test.index)
        scored["chosen_lag_order_p"] = best_p
        scored["val_rmse_at_chosen_p"] = best_val_rmse
        payload["targets"][target_key] = scored
        if target_key == "wind_power":
            payload["targets"][target_key]["caveat"] = bc.WIND_CAVEAT
    path = bc.save_result("ar", payload)
    results["ar"] = payload
    print("Saved:", path)

    return results


if __name__ == "__main__":
    import json
    results = run()
    print(json.dumps(results, indent=2))
