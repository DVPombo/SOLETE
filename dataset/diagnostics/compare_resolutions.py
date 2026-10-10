"""
Compare a freshly-resampled file (from resample_solete.py) against the
originally-published file at the same resolution, to determine whether
known anomalies (wind-direction >360 deg, wind-power zero-degeneracy)
already exist in a correctly-resampled version from raw, or only appear
in the originally-published file -- i.e. whether they're a genuine raw-
data characteristic or an artifact of the original resampling pipeline.

Checks for a labeling-convention mismatch FIRST, before concluding
anything is a real discrepancy -- see the shift-search below.

Usage:
    python compare_resolutions.py \
        --fresh SOLETE_Pombo_60min_v4.h5 --fresh-key DATA \
        --original SOLETE_Pombo_60min.h5 --original-key DATA
"""
import argparse

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipeline"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import resolve_input
from solete_report import print_report


def detect_label_shift(fresh: pd.DataFrame, original: pd.DataFrame, column: str, max_shift=3) -> int | None:
    """
    Tries shifting `fresh` by -max_shift..+max_shift periods and returns
    the shift that minimizes total absolute difference against `original`
    on `column`, IF that shifted alignment is meaningfully better than no
    shift at all. A non-zero best shift is a strong hint the two files use
    different interval-labeling conventions rather than genuinely
    disagreeing on values.
    """
    best_shift, best_diff = 0, None
    common_cols_diff_at_0 = None
    for shift in range(-max_shift, max_shift + 1):
        shifted = fresh[column].shift(shift)
        aligned = pd.concat([shifted, original[column]], axis=1, join="inner").dropna()
        if aligned.empty:
            continue
        diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().mean()
        if shift == 0:
            common_cols_diff_at_0 = diff
        if best_diff is None or diff < best_diff:
            best_diff, best_shift = diff, shift

    if best_shift != 0 and common_cols_diff_at_0 is not None and best_diff < common_cols_diff_at_0 * 0.1:
        print(
            f"  NOTE: shifting by {best_shift} period(s) reduces mean abs diff from "
            f"{common_cols_diff_at_0:.4f} to {best_diff:.4f} -- this looks like a "
            f"labeling-convention mismatch (interval start vs. end), not a real "
            f"data discrepancy. Re-check both files' resample(..., label=...) "
            f"convention before concluding anything is actually wrong."
        )
        return best_shift
    return None


def compare_column(fresh: pd.Series, original: pd.Series, tolerance: float = 1e-6) -> dict:
    aligned = pd.concat([fresh, original], axis=1, join="inner").dropna()
    if aligned.empty:
        return {"n_compared": 0}
    diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs()
    return {
        "n_compared": len(aligned),
        "n_matching_within_tolerance": int((diff <= tolerance).sum()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def compare_files(fresh: pd.DataFrame, original: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    common_cols = sorted(set(fresh.columns) & set(original.columns))
    rows = []
    for col in common_cols:
        shift = detect_label_shift(fresh, original, col)
        result = compare_column(fresh[col], original[col], tolerance=tolerance)
        result["column"] = col
        result["suspected_label_shift"] = shift
        rows.append(result)
    return pd.DataFrame(rows).set_index("column")


def check_known_anomalies(fresh: pd.DataFrame, original: pd.DataFrame) -> dict:
    """
    The actual diagnostic question this whole investigation is for:
    do the two specific known anomalies exist in the properly-resampled
    fresh version, or only in the original? Returns a dict alongside the
    printed narration, so main() can fold it into the structured report.
    """
    result = {}

    if "WIND_DIR[deg]" in fresh.columns and "WIND_DIR[deg]" in original.columns:
        fresh_over = int((fresh["WIND_DIR[deg]"] > 360).sum())
        orig_over = int((original["WIND_DIR[deg]"] > 360).sum())
        print(f"WIND_DIR[deg] > 360: fresh resample = {fresh_over} rows, original = {orig_over} rows")
        conclusion = None
        if fresh_over == 0 and orig_over > 0:
            conclusion = "does_not_reproduce_from_raw -- likely a resampling-pipeline bug"
            print("  -> Anomaly does NOT reproduce in a correct resample from raw. "
                  "Strong evidence this is a bug specific to the original resampling "
                  "pipeline, not a raw-data characteristic.")
        elif fresh_over > 0:
            conclusion = "reproduces_from_raw -- check 1-second file directly for the same values"
            print("  -> Anomaly reproduces even from a correctly-resampled raw source. "
                  "Check the 1-second file itself for the same out-of-range values -- "
                  "if present there too, this traces back to raw sensor/logging, not "
                  "resampling.")
        else:
            conclusion = "no_anomaly_in_either_file"
        result["wind_direction_over_360"] = {
            "fresh_rows_over_360": fresh_over,
            "original_rows_over_360": orig_over,
            "conclusion": conclusion,
        }

    if "P_Gaia[kW]" in fresh.columns and "P_Gaia[kW]" in original.columns:
        fresh_zero_pct = 100 * (fresh["P_Gaia[kW]"].abs() < 1e-6).mean()
        orig_zero_pct = 100 * (original["P_Gaia[kW]"].abs() < 1e-6).mean()
        print(f"P_Gaia[kW] near-zero: fresh resample = {fresh_zero_pct:.2f}%, "
              f"original = {orig_zero_pct:.2f}%")
        diff = abs(fresh_zero_pct - orig_zero_pct)
        if diff < 1.0:
            conclusion = "consistent -- check 1-second raw data directly for genuine downtime vs. artifact"
            print("  -> Consistent between fresh and original -- if the 1-second raw "
                  "data (checked separately via data_quality_profile.py) also shows "
                  "near-total inactivity, this points to genuine turbine downtime, "
                  "not a resampling artifact.")
        else:
            conclusion = "meaningfully_different -- investigate further"
            print("  -> Meaningfully different between fresh and original -- worth "
                  "investigating further before concluding either way.")
        result["wind_power_near_zero"] = {
            "fresh_pct_near_zero": fresh_zero_pct,
            "original_pct_near_zero": orig_zero_pct,
            "pct_point_difference": diff,
            "conclusion": conclusion,
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", required=True)
    parser.add_argument("--fresh-key", default="DATA")
    parser.add_argument("--original", required=True)
    parser.add_argument("--original-key", default="DATA")
    args = parser.parse_args()
    args.fresh = str(resolve_input(args.fresh))
    args.original = str(resolve_input(args.original))

    fresh = pd.read_hdf(args.fresh, key=args.fresh_key)
    original = pd.read_hdf(args.original, key=args.original_key)

    print("=== Per-column comparison (fresh resample vs. originally published) ===")
    comparison_df = compare_files(fresh, original)
    print(comparison_df.to_string())
    print()
    print("=== Known-anomaly check ===")
    anomaly_result = check_known_anomalies(fresh, original)

    print_report(
        "compare_resolutions_summary",
        {
            "fresh_file": args.fresh,
            "original_file": args.original,
            "n_rows_fresh": len(fresh),
            "n_rows_original": len(original),
            "per_column_comparison": comparison_df.reset_index().to_dict(orient="records"),
            "known_anomaly_check": anomaly_result,
        },
    )


if __name__ == "__main__":
    main()
