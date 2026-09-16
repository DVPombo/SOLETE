# -*- coding: utf-8 -*-
"""
baseline_persistence.py -- SOLETE Phase 5 continuation, Task 5.3

Plain persistence:        y_hat(t) = y(t - 1h)      (1-step-ahead, 60min res.)
Smart/seasonal persistence: y_hat(t) = y(t - 24h)    (same hour, previous day)

Seasonal-persistence choice: "same hour yesterday" (24-step lag at 60min
resolution) is the standard seasonal-naive baseline for hourly solar/wind
forecasting benchmarks (e.g. it is the de facto "smart persistence"
definition used across the solar forecasting literature -- it captures the
strong diurnal cycle that plain persistence misses, without needing any
fitted parameters). No fitting is performed for either baseline; both need
only historical lookback, which is allowed to reach across split boundaries
per the continuation prompt (only which rows are *scored* respects the
split).

Evaluated at horizon = 1 (single-step, 1 hour ahead at this dataset's 60min
resolution) against splits/v1.json's test block, via bench_common.py /
metrics.py exclusively (no parallel scoring path).
"""

import bench_common as bc

WIND_TARGET = "wind_power"
PV_TARGET = "pv_power"


def make_predictions(df_full):
    preds = {}
    for target_key, col in bc.TARGETS.items():
        series = df_full[col]
        preds[target_key] = {
            "persistence": series.shift(1),
            "smart_persistence": series.shift(24),
        }
    return preds


def run():
    df = bc.load_full_df()
    _, _, test = bc.split_df(df)
    preds = make_predictions(df)

    results = {}
    for model_name in ["persistence", "smart_persistence"]:
        payload = {
            "model": model_name,
            "description": (
                "y_hat(t) = y(t-1h)" if model_name == "persistence"
                else "y_hat(t) = y(t-24h) (same hour, previous day)"
            ),
            "split_version": "v1",
            "horizon": "1 step (1h)",
            "fitting": "none (no train-set fitting; history lookback only)",
            "targets": {},
        }
        for target_key in bc.TARGETS:
            scored = bc.score_target_on_test(df, target_key, preds[target_key][model_name], test.index)
            payload["targets"][target_key] = scored
            if target_key == WIND_TARGET:
                payload["targets"][target_key]["caveat"] = bc.WIND_CAVEAT

        path = bc.save_result(model_name, payload)
        results[model_name] = payload
        print("Saved:", path)

    return results


if __name__ == "__main__":
    import json
    results = run()
    print(json.dumps(results, indent=2))
