# -*- coding: utf-8 -*-
"""
availability_report.py

Per-column-per-file completeness and QC-flag report, built on top of
inspect_dataset.py's HDF5-key discovery and the QC flag layer
(Functions.apply_qc_flags / QC_SCHEMA.md).

For every column in every file/key, reports two related-but-distinct things:
  - completeness: expected sample count (inferred from the file's own time
    span and its own row spacing -- not a hardcoded "hourly" assumption, so
    this still works if pointed at a 5min/1min/1sec file) vs. actual
    non-null count.
  - QC-validity: for columns the QC layer covers, what fraction of the
    non-null values are flagged, broken down by flag type. A column can be
    100% complete (no NaNs) and still be substantially QC-flagged -- that's
    exactly the Pressure[mbar] case.

Usage
-----
    python scripts/availability_report.py SOLETE_short.h5 SOLETE_Pombo_60min.h5 \
        --csv availability_report.csv

Note: this imports Functions.py to reuse apply_qc_flags/build_*_qc_rules and
(for the P_Solar[kW]_qc breakdown) PV_Performance_Model -- so it shares
Functions.py's dependency footprint (scikit-learn, keras/TensorFlow,
CoolProp; see requirements.txt), not just h5py/pandas/numpy.
"""
import argparse
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from inspect_dataset import discover_keys  # Phase 1, reused as-is

from Functions import (  # Phase 2
    apply_qc_flags,
    build_raw_value_qc_rules,
    build_substitution_qc_rule,
    import_PV_WT_data,
    PV_Performance_Model,
    QC_VALID,
    QC_MISSING,
    QC_SENSOR_ERROR,
    QC_PHYSICALLY_IMPLAUSIBLE,
    QC_INTERPOLATED,
    QC_AGGREGATION_AFFECTED_BY_GAPS,
    QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED,
)

QC_FLAG_NAMES = {
    QC_VALID: "valid",
    QC_MISSING: "missing",
    QC_SENSOR_ERROR: "sensor_error",
    QC_PHYSICALLY_IMPLAUSIBLE: "physically_implausible",
    QC_INTERPOLATED: "interpolated",
    QC_AGGREGATION_AFFECTED_BY_GAPS: "aggregation_affected_by_gaps",
    QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED: "suspected_curtailment_or_model_substituted",
}


def infer_resolution(index: pd.DatetimeIndex) -> pd.Timedelta:
    """
    Infer the nominal sample spacing from the data itself (the *mode* of
    consecutive gaps after sorting -- not the file's on-disk order, which
    KNOWN_ISSUES.md/finding #5 established isn't always chronological, and
    not a hardcoded '60min' assumption, so this generalizes to whatever
    resolution file it's pointed at).
    """
    sorted_index = index.sort_values()
    diffs = sorted_index[1:] - sorted_index[:-1]
    if len(diffs) == 0:
        return pd.Timedelta(0)
    return diffs.value_counts().idxmax()


def add_qc_columns(df: pd.DataFrame, pv_info: dict) -> pd.DataFrame:
    """Run the full Phase 2 QC layer on a copy of df, mirroring what
    import_SOLETE_data()/ExpandSOLETE() do, so the report reflects the same
    flags a real Build would produce -- without mutating the caller's df."""
    df = df.copy()
    apply_qc_flags(df, build_raw_value_qc_rules(df))

    if "P_Solar[kW]" in df.columns:
        Pac, _, _, _ = PV_Performance_Model(df, pv_info)
        df["Pac"] = Pac
        df["P_Solar_model_substituted"] = df["Pac"] >= 1.5 * df["P_Solar[kW]"]
        apply_qc_flags(df, [build_substitution_qc_rule()])

    return df


def report_for_file(path: str, pv_info: dict) -> pd.DataFrame:
    rows = []
    for key in discover_keys(path):
        df = pd.read_hdf(path, key=key) if key != "/" else pd.read_hdf(path)
        df = add_qc_columns(df, pv_info)

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            resolution = infer_resolution(df.index)
            span = df.index.max() - df.index.min()
            expected_count = int(span / resolution) + 1 if resolution else len(df)
        else:
            resolution = None
            expected_count = len(df)

        source_cols = [c for c in df.columns if not c.endswith("_qc") and c != "Pac"
                       and c != "P_Solar_model_substituted"]

        for col in source_cols:
            s = df[col]
            actual_non_null = int(s.notna().sum())
            row = {
                "file": path,
                "key": key,
                "column": col,
                "inferred_resolution": str(resolution) if resolution is not None else "n/a",
                "expected_count": expected_count,
                "actual_non_null_count": actual_non_null,
                "completeness_pct": round(100 * actual_non_null / expected_count, 3)
                if expected_count else float("nan"),
            }

            qc_col = f"{col}_qc"
            if qc_col in df.columns:
                counts = df.loc[s.notna(), qc_col].value_counts()
                n_nonnull = max(actual_non_null, 1)
                for flag_value, flag_name in QC_FLAG_NAMES.items():
                    row[f"qc_{flag_name}_count"] = int(counts.get(flag_value, 0))
                row["qc_flagged_pct_of_nonnull"] = round(
                    100 * (n_nonnull - counts.get(QC_VALID, 0)) / n_nonnull, 3
                )
            else:
                for flag_name in QC_FLAG_NAMES.values():
                    row[f"qc_{flag_name}_count"] = None
                row["qc_flagged_pct_of_nonnull"] = None

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5_files", nargs="+", help="One or more .h5 files to report on")
    ap.add_argument("--csv", default="availability_report.csv",
                     help="Output CSV path (default: availability_report.csv)")
    args = ap.parse_args()

    PV, _WT = import_PV_WT_data()

    combined = pd.concat(
        [report_for_file(path, PV) for path in args.h5_files],
        ignore_index=True,
    )
    combined.to_csv(args.csv, index=False)
    print(f"Wrote {len(combined)} rows to {args.csv}")

    worst_completeness = combined.nsmallest(5, "completeness_pct")[
        ["file", "column", "completeness_pct"]
    ]
    worst_qc = combined.dropna(subset=["qc_flagged_pct_of_nonnull"]).nlargest(
        5, "qc_flagged_pct_of_nonnull"
    )[["file", "column", "qc_flagged_pct_of_nonnull"]]

    print("\nWorst completeness:\n", worst_completeness.to_string(index=False))
    print("\nWorst QC-flag rate:\n", worst_qc.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
