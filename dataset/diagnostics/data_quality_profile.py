"""
Generic, per-column data-quality profiler -- designed to run against the
1-second-resolution file and surface anything "fishy" in ANY column, not
just wind. Extends the pressure-sentinel-style detection from earlier
phases (finding exact-repeated round numbers) to every numeric column
automatically, plus gap/duplicate/stuck-value detection and a wind-speed-
vs-wind-power physical-consistency check.

Usage (small/medium files -- tries a direct full load first):
    python data_quality_profile.py path/to/SOLETE_1sec.h5 --key DATA

For files too large to load directly, see `profile_all_columns_chunked`
below and adjust the __main__ block to call it instead.
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipeline"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import resolve_input
from solete_report import print_report


def profile_column(series: pd.Series, name: str, valid_range=None) -> dict:
    """
    One column's summary: basic stats, missingness, zero-fraction, the
    single longest run of an identical consecutive value ("stuck sensor"
    check), and the top-5 most frequent exact values with their share of
    the column (generalizes the pressure-sentinel-value finding from
    Phase 1 to any column, automatically).
    """
    n = len(series)
    non_null = series.dropna()
    n_missing = n - len(non_null)

    # Stuck-value run detection: longest run of consecutive identical values.
    if len(non_null) > 0:
        vals = non_null.to_numpy()
        change_points = np.flatnonzero(np.diff(vals) != 0)
        run_lengths = np.diff(np.concatenate(([0], change_points + 1, [len(vals)])))
        longest_run = int(run_lengths.max()) if len(run_lengths) else len(vals)
    else:
        longest_run = 0

    # Top repeated exact values -- surfaces sentinel/placeholder values
    # (like the 1000.0/2000.0/3000.0 pressure case) without hardcoding
    # what to look for.
    top_values = non_null.value_counts().head(5)

    result = {
        "column": name,
        "n": n,
        "n_missing": n_missing,
        "pct_missing": 100 * n_missing / n if n else float("nan"),
        "n_zero": int((non_null == 0).sum()),
        "pct_zero": 100 * (non_null == 0).sum() / n if n else float("nan"),
        "min": float(non_null.min()) if len(non_null) else float("nan"),
        "max": float(non_null.max()) if len(non_null) else float("nan"),
        "mean": float(non_null.mean()) if len(non_null) else float("nan"),
        "std": float(non_null.std()) if len(non_null) else float("nan"),
        "longest_constant_run": longest_run,
        "top_repeated_values": list(top_values.items()),
    }

    if valid_range is not None:
        lo, hi = valid_range
        n_out_of_range = int(((non_null < lo) | (non_null > hi)).sum())
        result["n_out_of_range"] = n_out_of_range
        result["pct_out_of_range"] = 100 * n_out_of_range / n if n else float("nan")

    # Distinct calendar dates where this column has a real (non-zero,
    # non-null) reading -- this is the single most useful number for a
    # sparse column like Azimuth[deg]/Elevation[deg]/P_Gaia[kW]: "real
    # values on N of M days" is a much clearer signal than a bare
    # zero-percentage, and is exactly the number that mattered when the
    # hourly file's azimuth/elevation turned out to be real on only one
    # calendar day.
    if isinstance(series.index, pd.DatetimeIndex) and len(non_null) > 0:
        nonzero = non_null[non_null != 0]
        n_total_days = series.index.normalize().nunique()
        n_nonzero_days = nonzero.index.normalize().nunique() if len(nonzero) else 0
        result["n_total_calendar_days"] = int(n_total_days)
        result["n_calendar_days_with_nonzero_value"] = int(n_nonzero_days)
        if n_nonzero_days > 0:
            result["first_nonzero_timestamp"] = str(nonzero.index.min())
            result["last_nonzero_timestamp"] = str(nonzero.index.max())
            # Up to 10 example nonzero calendar dates -- enough to spot a
            # pattern (all one contiguous block? scattered singletons?)
            # without dumping every date if there are many.
            example_dates = sorted(nonzero.index.normalize().unique())[:10]
            result["example_nonzero_dates"] = [str(d.date()) for d in example_dates]

    return result


def check_timestamp_integrity(index: pd.DatetimeIndex, expected_freq="1s") -> dict:
    """
    Duplicate timestamps, non-monotonic ordering, and gaps versus the
    expected sampling interval -- worth checking before trusting anything
    else, since a broken index can masquerade as a data-quality problem
    in every column at once.
    """
    n = len(index)
    n_duplicates = int(index.duplicated().sum())
    is_monotonic = bool(index.is_monotonic_increasing)
    expected_delta = pd.Timedelta(expected_freq)
    deltas = index.to_series().diff().dropna()
    n_gaps = int((deltas > expected_delta).sum())
    largest_gap = deltas.max() if len(deltas) else pd.Timedelta(0)
    return {
        "n_rows": n,
        "n_duplicate_timestamps": n_duplicates,
        "is_monotonic_increasing": is_monotonic,
        "n_gaps_larger_than_expected": n_gaps,
        "largest_gap": str(largest_gap),
    }


def wind_power_speed_consistency_check(
    df: pd.DataFrame,
    power_col: str = "P_Gaia[kW]",
    speed_col: str = "WIND_SPEED[m1s]",
    cutin_speed: float = 3.5,
    cutout_speed: float = 25.0,
    power_floor: float = 0.01,
) -> dict:
    """
    The actual "is something fishy" check for wind, as opposed to just
    profiling power in isolation. Two physically-grounded checks against
    the real Gaia-Wind 133-11kW datasheet specs (cut-in 3.5 m/s, cut-out
    25.0 m/s -- confirmed from the manufacturer's own published datasheet,
    not guessed):

    1. Zero power at low wind speed (below cut-in) is normal. Zero power
       at meaningfully-above-cut-in wind speed is a red flag -- the
       turbine should be producing something in that regime if genuinely
       operational.
    2. Symmetrically: nonzero power ABOVE cut-out speed is also a red
       flag -- the turbine should have shut down/feathered by then, so a
       real reading there would itself be unusual (though check for
       plausible brief transients around the exact threshold before
       treating a handful of borderline rows as a real problem).
    """
    above_cutin = df[speed_col] > cutin_speed
    above_cutout = df[speed_col] > cutout_speed
    near_zero_power = df[power_col].abs() < power_floor
    nonzero_power = ~near_zero_power

    suspicious_low = above_cutin & ~above_cutout & near_zero_power
    suspicious_high = above_cutout & nonzero_power

    n_above_cutin = int((above_cutin & ~above_cutout).sum())
    n_suspicious_low = int(suspicious_low.sum())
    n_above_cutout = int(above_cutout.sum())
    n_suspicious_high = int(suspicious_high.sum())

    result = {
        "cutin_speed_used": cutin_speed,
        "cutout_speed_used": cutout_speed,
        "n_rows_between_cutin_and_cutout": n_above_cutin,
        "n_rows_in_that_range_with_near_zero_power": n_suspicious_low,
        "pct_of_that_range_suspicious": (
            100 * n_suspicious_low / n_above_cutin if n_above_cutin else float("nan")
        ),
        "n_rows_above_cutout": n_above_cutout,
        "n_rows_above_cutout_with_nonzero_power": n_suspicious_high,
        "pct_of_above_cutout_rows_suspicious": (
            100 * n_suspicious_high / n_above_cutout if n_above_cutout else float("nan")
        ),
    }
    if n_suspicious_low > 0:
        result["example_suspicious_low_timestamps"] = [
            str(t) for t in df.index[suspicious_low][:10].tolist()
        ]
    if n_suspicious_high > 0:
        result["example_suspicious_high_timestamps"] = [
            str(t) for t in df.index[suspicious_high][:10].tolist()
        ]
    return result


def profile_all_columns(df: pd.DataFrame, valid_ranges: dict | None = None) -> pd.DataFrame:
    """
    valid_ranges: optional {column_name: (lo, hi)} -- e.g.
        {"HUMIDITY[%]": (0, 1), "WIND_DIR[deg]": (0, 360)}
    to flag known physically-implausible bounds per QC_SCHEMA.md's existing
    rules, applied here at 1-second resolution to see whether the same
    problems already exist in the raw data or only appear after resampling.
    """
    valid_ranges = valid_ranges or {}
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        rows.append(profile_column(df[col], col, valid_range=valid_ranges.get(col)))
    return pd.DataFrame(rows)


def profile_all_columns_chunked(hdf_path: str, key: str, chunksize: int = 2_000_000):
    """
    Skeleton for files too large to load directly. Runs a two-pass
    incremental profile: pass 1 accumulates count/sum/sumsq/min/max/zero-
    count/value-counts per column across chunks; pass 2 (if you need exact
    longest-constant-run across chunk boundaries) would need to carry the
    trailing value+run-length state from each chunk into the next -- left
    as a documented extension point rather than implemented here, since a
    single chunk's own longest run is usually informative enough for an
    initial pass. Start here if `profile_all_columns` runs out of memory
    on the real file; try the direct approach first, most 3GB HDF5 files
    load fine in an 8GB+ RAM environment.
    """
    raise NotImplementedError(
        "Not implemented -- try profile_all_columns() with a direct "
        "pd.read_hdf() load first. If that genuinely runs out of memory, "
        "this is the place to add chunked accumulation (see docstring)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--key", default="DATA")
    args = parser.parse_args()
    args.path = str(resolve_input(args.path))

    df = pd.read_hdf(args.path, key=args.key)

    print("=== Timestamp integrity ===")
    ts_integrity = check_timestamp_integrity(df.index)
    print(ts_integrity)
    print()

    print("=== Per-column profile ===")
    profile_df = profile_all_columns(
        df,
        valid_ranges={
            "HUMIDITY[%]": (0, 1),
            "WIND_DIR[deg]": (0, 360),
        },
    )
    print(profile_df.to_string())
    print()

    wind_consistency = None
    if "P_Gaia[kW]" in df.columns and "WIND_SPEED[m1s]" in df.columns:
        print("=== Wind speed vs. wind power consistency ===")
        wind_consistency = wind_power_speed_consistency_check(df)
        print(wind_consistency)

    # --- Structured report for copy-paste-back analysis ---
    print_report(
        "1sec_data_quality_profile",
        {
            "source_file": args.path,
            "n_rows": len(df),
            "columns": list(df.columns),
            "timestamp_integrity": ts_integrity,
            "column_profiles": profile_df.to_dict(orient="records"),
            "wind_speed_power_consistency": wind_consistency,
        },
    )
