# -*- coding: utf-8 -*-
"""
task6_3_ramp_rate_analysis.py -- SOLETE Phase 6, Task 6.3

Ramp-rate (period-to-period change) characterization of P_hybrid[kW] vs.
P_Solar[kW] alone. Scoped per Task 6.0's decision (path (b)): the
test-split-wide distribution comparison is a legitimate, general
"how much, and how rarely" characterization regardless of the wind-
complementarity question (KNOWN_ISSUES.md #10) -- it describes the data as
it is. The narrower "did the two known wind-active days show smoothing"
question is kept explicitly separate and scoped to n=2 calendar days: an
observation, not a general claim about wind-solar ramp smoothing.

Outputs:
    results/hybrid_ramp_rate_summary.json
    results/hybrid_ramp_rate_histogram.png
    results/hybrid_ramp_rate_active_days.png
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bench_common as bc

HYBRID_COL = "P_hybrid[kW]"
SOLAR_COL = "P_Solar[kW]"
WIND_COL = "P_Gaia[kW]"
ACTIVE_DAYS = ["2018-08-31", "2019-05-25"]  # splits/README.md


def summary_stats(series):
    s = series.dropna()
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "max": float(s.max()),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
    }


def run():
    df = bc.load_full_df()
    train, val, test = bc.split_df(df)

    # --- Part 1: test-split-wide ramp distribution (the general, non-caveat-
    # gated characterization Task 6.3.1 asks for) ---
    d_hybrid = test[HYBRID_COL].diff().abs()
    d_solar = test[SOLAR_COL].diff().abs()

    stats_hybrid = summary_stats(d_hybrid)
    stats_solar = summary_stats(d_solar)

    # rows where the two diverge at all (i.e. where a wind delta contributed)
    diff_of_diffs = (d_hybrid - d_solar).dropna()
    rows_diverging = int((diff_of_diffs.abs() > 1e-9).sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(d_hybrid.max(), d_solar.max()), 60)
    ax.hist(d_solar.dropna(), bins=bins, alpha=0.6, label="|ΔP_Solar[kW]|", color="#d9822b")
    ax.hist(d_hybrid.dropna(), bins=bins, alpha=0.6, label="|ΔP_hybrid[kW]|", color="#2b6cb0")
    ax.set_xlabel("Absolute period-to-period change (kW)")
    ax.set_ylabel("Count (test split, n={})".format(len(test)))
    ax.set_title("Ramp-rate distribution, test split: P_hybrid vs P_Solar alone\n"
                  "(near-identical by construction -- wind active on only 24/2953 rows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/hybrid_ramp_rate_histogram.png", dpi=130)
    plt.close(fig)

    # --- Part 2: narrow, two-day case study (NOT the test-split-wide claim) ---
    active_day_findings = {}
    for day in ACTIVE_DAYS:
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(hours=23)
        # pad by 1 hour on each side so the diff() at the day's first/last
        # row (relative to the neighbouring day) is captured too
        window = df.loc[day_start - pd.Timedelta(hours=1): day_end + pd.Timedelta(hours=1)]

        w_hybrid = window[HYBRID_COL].diff().abs()
        w_solar = window[SOLAR_COL].diff().abs()
        w_wind = window[WIND_COL]

        active_day_findings[day] = {
            "in_split": ("train" if day_start <= train.index.max() else
                         "val" if day_start <= val.index.max() else "test"),
            "wind_active_hours": int((w_wind.loc[day_start:day_end] > 0).sum()),
            "max_abs_delta_hybrid_kw": float(w_hybrid.max()),
            "max_abs_delta_solar_kw": float(w_solar.max()),
            "hours_where_hybrid_and_solar_ramps_differ": int(
                (w_hybrid - w_solar).abs().gt(1e-9).sum()
            ),
            "largest_single_divergence_kw": float((w_hybrid - w_solar).abs().max()),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, day in zip(axes, ACTIVE_DAYS):
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(hours=23)
        window = df.loc[day_start - pd.Timedelta(hours=1): day_end + pd.Timedelta(hours=1)]
        ax.plot(window.index, window[SOLAR_COL], label="P_Solar[kW]", color="#d9822b")
        ax.plot(window.index, window[HYBRID_COL], label="P_hybrid[kW]", color="#2b6cb0", linestyle="--")
        ax.set_title(f"{day} ({active_day_findings[day]['in_split']} split)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.suptitle("The two known wind-active days: P_hybrid vs P_Solar alone")
    fig.tight_layout()
    fig.savefig("results/hybrid_ramp_rate_active_days.png", dpi=130)
    plt.close(fig)

    result = {
        "task": "6.3 -- ramp-rate / volatility analysis",
        "decision_gate": "Task 6.0 chose path (b). See KNOWN_ISSUES.md #10.",
        "part_1_test_split_wide": {
            "scope": "All 2,953 test-split rows, general characterization, "
                     "not gated on wind activity.",
            "abs_delta_P_solar_kw": stats_solar,
            "abs_delta_P_hybrid_kw": stats_hybrid,
            "rows_where_hybrid_and_solar_ramp_differ_at_all": rows_diverging,
            "rows_where_hybrid_and_solar_ramp_differ_pct": round(
                100.0 * rows_diverging / len(d_hybrid.dropna()), 3),
            "interpretation": (
                "Across the full test split, P_hybrid's and P_Solar's ramp-rate "
                "distributions are effectively identical (matching means/medians/"
                "percentiles above to 2-3 significant figures) -- the two series "
                "differ at all in only {} of {} scored rows ({:.3f}%), which is "
                "consistent with wind being active on 0.81% of test rows and "
                "contributing a small, spiky term when it is. This is NOT evidence "
                "of a general ramp-smoothing (or ramp-worsening) effect from wind on "
                "this dataset -- it is evidence there is almost no wind signal "
                "present in the test split to smooth or worsen anything with."
            ).format(rows_diverging, len(d_hybrid.dropna()),
                      100.0 * rows_diverging / len(d_hybrid.dropna())),
        },
        "part_2_two_known_active_days_case_study": {
            "scope": "n=2 calendar days ONLY -- not a general claim, see module docstring.",
            "days": active_day_findings,
        },
        "figures": [
            "results/hybrid_ramp_rate_histogram.png",
            "results/hybrid_ramp_rate_active_days.png",
        ],
    }

    with open("results/hybrid_ramp_rate_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
