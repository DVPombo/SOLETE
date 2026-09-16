# -*- coding: utf-8 -*-
"""
baseline_gbm.py -- SOLETE Phase 5 continuation, Task 5.5

ASK FIRST (library choice): LightGBM was picked over XGBoost. Reasoning,
for the maintainer to override if preferred: LightGBM's default
histogram-based split-finding trains noticeably faster on this feature
count/row count with no accuracy trade-off worth caring about at this
dataset size (~10^4 rows), pip-installs a smaller wheel, and its sklearn-
style API (LGBMRegressor) is a drop-in match for the rest of this codebase's
sklearn conventions (see baseline_climatology_ar.py's LinearRegression
usage). If the maintainer prefers XGBoost for ecosystem-consistency reasons,
swapping is a small, isolated change confined to this file (fit_one() below)
-- nothing else in Tasks 5.6-5.8 depends on which library was used here.
`requirements.txt` gets a new `lightgbm` line for this.

Feature set (documented per the task -- kept simple/legible, this is a
benchmark baseline, not a competition entry):
    - Lagged target: y(t-1), y(t-2), y(t-3), y(t-24)
    - "Same-timestamp" meteorological readings at t: GHI[kW1m2],
      POA Irr[kW1m2], WIND_SPEED[m1s], TEMPERATURE[degC]
    - Calendar features: hour-of-day, month, hour sin/cos, day-of-year sin/cos

IMPORTANT ASSUMPTION carried over from the task's own feature-set spec:
"same-timestamp" meteorological readings means the model is given the
*measured* weather at the target timestamp t, not a t-1 lookback of it. In a
real deployed forecast this would have to come from an actual NWP forecast,
not a measurement -- using the measurement is a common simplifying
convention in PV/wind forecasting benchmarks (treating perfect/measured
weather as a stand-in for an accurate short-horizon weather forecast), and
matches this task's own instruction verbatim. It is *not* target leakage
(weather is not the target), but it does mean this baseline's skill is not
directly comparable to a baseline that only sees weather up to t-1 -- flagged
here and again in BENCHMARKS.md.

Fit on train, lag/hyperparameters tuned on val, scored on test via
bench_common.py / metrics.py.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import bench_common as bc

MET_COLS = ["GHI[kW1m2]", "POA Irr[kW1m2]", "WIND_SPEED[m1s]", "TEMPERATURE[degC]"]
LAGS = [1, 2, 3, 24]

HYPERPARAM_GRID = [
    {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.1},
    {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05},
    {"n_estimators": 500, "num_leaves": 63, "learning_rate": 0.03},
]


def build_features(df, col):
    feats = {}
    for l in LAGS:
        feats[f"lag_{l}"] = df[col].shift(l)
    for m in MET_COLS:
        feats[m] = df[m]
    idx = df.index
    hour = idx.hour.values
    doy = idx.dayofyear.values
    feats["hour"] = hour
    feats["month"] = idx.month.values
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X = pd.DataFrame(feats, index=idx)
    return X


def fit_and_predict(df, train, val, col):
    X_full = build_features(df, col)
    y_full = df[col]

    X_train = X_full.loc[train.index].dropna()
    y_train = y_full.loc[X_train.index]
    X_val = X_full.loc[val.index].dropna()
    y_val = y_full.loc[X_val.index]

    best_rmse, best_model, best_params = np.inf, None, None
    for params in HYPERPARAM_GRID:
        model = LGBMRegressor(random_state=0, verbosity=-1, **params)
        model.fit(X_train.values, y_train.values)
        pred_val = model.predict(X_val.values)
        val_rmse = float(np.sqrt(np.mean((y_val.values - pred_val) ** 2)))
        if val_rmse < best_rmse:
            best_rmse, best_model, best_params = val_rmse, model, params

    valid_idx = X_full.dropna().index
    pred_full = pd.Series(np.nan, index=df.index)
    pred_full.loc[valid_idx] = best_model.predict(X_full.loc[valid_idx].values)

    return pred_full, best_params, best_rmse


def run():
    df = bc.load_full_df()
    train, val, test = bc.split_df(df)

    payload = {
        "model": "gradient_boosting",
        "library": "lightgbm",
        "description": "LightGBM regressor on lagged target + same-timestamp weather + calendar features (see module docstring)",
        "split_version": "v1",
        "horizon": "1 step (1h)",
        "fitting": "fit on train, hyperparameters selected on val by RMSE",
        "feature_set": {
            "lags": LAGS,
            "meteorological_same_timestamp": MET_COLS,
            "calendar": ["hour", "month", "hour_sin", "hour_cos", "doy_sin", "doy_cos"],
        },
        "targets": {},
    }
    for target_key, col in bc.TARGETS.items():
        pred, best_params, best_val_rmse = fit_and_predict(df, train, val, col)
        scored = bc.score_target_on_test(df, target_key, pred, test.index)
        scored["chosen_hyperparameters"] = best_params
        scored["val_rmse_at_chosen_params"] = best_val_rmse
        payload["targets"][target_key] = scored
        if target_key == "wind_power":
            payload["targets"][target_key]["caveat"] = bc.WIND_CAVEAT

    path = bc.save_result("gradient_boosting", payload)
    print("Saved:", path)
    return payload


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
