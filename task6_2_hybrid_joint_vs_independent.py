# -*- coding: utf-8 -*-
"""
task6_2_hybrid_joint_vs_independent.py -- SOLETE Phase 6, Task 6.2

Joint-vs-independent forecasting comparison for P_hybrid[kW] = P_Solar[kW] +
P_Gaia[kW], scoped by Task 6.0's decision (path (b), see KNOWN_ISSUES.md #10
and examples/05_hybrid_forecasting.ipynb): this script exists to build and
document the *methodology*, honestly, on data too wind-degenerate to support
a general complementarity claim -- not to manufacture one.

Two comparisons are reported, not one, because the single number the Task
6.2 spec asks for (sum of independent RMSEs vs. RMSE of a jointly-trained
model) is easy to build but easy to misread in isolation:

  (A) "Same feature set" comparison (primary, most defensible):
      independent = AR(p) fit separately on P_Solar[kW] and on P_Gaia[kW],
                    RMSE(solar) + RMSE(wind), reported on the SAME lag-only
                    feature protocol as the joint model below.
      joint       = AR(p) fit directly on P_hybrid[kW] (its own tuned lag
                    order), scored against true P_hybrid[kW].
      This is the literal Task 6.2 ask, with "whichever baseline scored best
      per-target" interpreted as "the best baseline usable under an
      identical, fair feature set for both the independent and joint arms" --
      not literally the single lowest RMSE per target in BENCHMARKS.md, which
      would silently mix the same-timestamp-weather GBM row (PV) with a
      lag-only row (wind) and make "joint, same feature set" impossible to
      define. See module docstring below the RESULTS constant for why.

  (B) BENCHMARKS.md-literal reference number (secondary, NOT feature-set-
      matched, reported for transparency only): RMSE(best PV row) +
      RMSE(best wind row) taken directly from BENCHMARKS.md, regardless of
      what feature set each used. Explicitly NOT compared against a joint
      model here, because no joint model trained on GBM's same-timestamp-
      weather features was built for this task -- listed only so a reader
      who wants the "literal best available number per target" isn't left
      wondering why it isn't the headline.

A third quantity is also reported: RMSE of the SUMMED independent
predictions (pred_solar + pred_wind) against the true P_hybrid[kW] series.
This is a stricter, more standard way to ask "does forecasting the parts
separately and adding them beat forecasting the sum directly?" than adding
two RMSEs computed on different target series (which don't share units of
comparison the way two RMSEs on the SAME target do) -- included because the
Task 6.2 spec's literal sum-of-RMSEs number is a common but slightly loose
proxy, and the more rigorous version is one extra computation away.
"""

import json

import numpy as np
import pandas as pd

import bench_common as bc
import metrics as M
from baseline_climatology_ar import ar_fit_and_predict

HYBRID_COL = "P_hybrid[kW]"
SOLAR_COL = "P_Solar[kW]"
WIND_COL = "P_Gaia[kW]"


def run():
    df = bc.load_full_df()
    train, val, test = bc.split_df(df)
    test_idx = test.index

    # --- Same-feature-set AR(p) models, one per target, each independently
    # lag-tuned on val (matches baseline_climatology_ar.py's own protocol
    # exactly, reused unchanged rather than reimplemented) ---
    pred_solar, p_solar, val_rmse_solar = ar_fit_and_predict(df, train, val, test, SOLAR_COL)
    pred_wind, p_wind, val_rmse_wind = ar_fit_and_predict(df, train, val, test, WIND_COL)
    pred_hybrid_joint, p_hybrid, val_rmse_hybrid = ar_fit_and_predict(df, train, val, test, HYBRID_COL)

    y_solar = df[SOLAR_COL].loc[test_idx]
    y_wind = df[WIND_COL].loc[test_idx]
    y_hybrid = df[HYBRID_COL].loc[test_idx]

    rmse_solar = M.rmse(y_solar, pred_solar.loc[test_idx])
    rmse_wind = M.rmse(y_wind, pred_wind.loc[test_idx])
    rmse_hybrid_joint = M.rmse(y_hybrid, pred_hybrid_joint.loc[test_idx])

    # (A) literal Task 6.2 ask: sum of independent RMSEs vs. joint RMSE
    sum_independent_rmse = rmse_solar + rmse_wind

    # Stricter version: sum the independent PREDICTIONS, then score once
    # against the true hybrid series -- apples-to-apples with rmse_hybrid_joint.
    pred_independent_summed = (pred_solar + pred_wind).loc[test_idx]
    rmse_independent_summed = M.rmse(y_hybrid, pred_independent_summed)

    # (B) BENCHMARKS.md-literal per-target best (NOT feature-set-matched;
    # reference only, copied by hand from BENCHMARKS.md's existing tables,
    # qc_included rows): GBM PV (0.0254) uses same-timestamp weather features
    # the AR/joint models here do not use; persistence wind (0.224) is
    # feature-set-trivial (y(t)=y(t-1)). Kept separate from (A) on purpose.
    benchmarks_literal_best = {
        "pv_best_row": {"model": "gradient_boosting (same-timestamp weather, caveated)", "rmse": 0.0254},
        "wind_best_row": {"model": "persistence", "rmse": 0.224},
        "sum_rmse": 0.0254 + 0.224,
        "note": "NOT feature-set-matched with any joint model in this script; "
                "reference number only, see module docstring.",
    }

    result = {
        "task": "6.2 -- joint vs independent hybrid forecasting comparison",
        "decision_gate": "Task 6.0 chose path (b): infrastructure + honest limitation. "
                          "See KNOWN_ISSUES.md #10 and examples/05_hybrid_forecasting.ipynb.",
        "split_version": "v1",
        "horizon": "1 step (1h)",
        "test_n": int(len(test_idx)),
        "wind_active_rows_in_test": int((y_wind > 0).sum()),
        "method": "AR(p), lagged-target-only OLS, lag order tuned per-target on val "
                  "(identical protocol/features across all three targets below -- "
                  "this is what makes the comparison 'same feature set').",
        "primary_comparison_A_same_feature_set": {
            "independent": {
                "solar_ar": {"chosen_lag_p": p_solar, "rmse_test": rmse_solar},
                "wind_ar": {"chosen_lag_p": p_wind, "rmse_test": rmse_wind},
                "sum_of_rmses_literal_task_spec": sum_independent_rmse,
                "rmse_of_summed_predictions_vs_true_hybrid_stricter_check": rmse_independent_summed,
            },
            "joint": {
                "hybrid_ar": {"chosen_lag_p": p_hybrid, "rmse_test": rmse_hybrid_joint},
            },
            "joint_minus_independent_stricter_check": rmse_hybrid_joint - rmse_independent_summed,
            "interpretation": (
                "Joint RMSE ({:.4f}) vs. independent-summed-predictions RMSE ({:.4f}) against "
                "the SAME true P_hybrid[kW] series: difference = {:+.4f} kW. Given test-split "
                "wind is active on 24 of {} rows (0.81%) with ~zero solar-wind covariance "
                "(0.0068, see KNOWN_ISSUES.md #10), ANY difference this small, in either "
                "direction, is not distinguishable from noise/near-zero-target artifacts on a "
                "target that is 96%+ solar by energy. This is NOT a demonstrated "
                "'joint beats independent' or 'independent beats joint' finding -- it is what "
                "near-identical performance looks like when the summed variable has almost no "
                "wind content for a joint model to exploit."
            ).format(rmse_hybrid_joint, rmse_independent_summed,
                      rmse_hybrid_joint - rmse_independent_summed, len(test_idx)),
        },
        "secondary_reference_B_benchmarks_literal_not_feature_matched": benchmarks_literal_best,
    }

    with open("results/hybrid_joint_vs_independent.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
