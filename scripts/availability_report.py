# -*- coding: utf-8 -*-
"""
availability_report.py

Per-column-per-file completeness and QC-flag report, built on top of
inspect_dataset.py's HDF5-key discovery and the QC flag layer
(solete.qc / dataset/docs/QC_SCHEMA.md).

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
    python scripts/availability_report.py examples/SOLETE_short.h5 SOLETE_Pombo_60min.h5 \
        --csv my_report.csv

The report consumes QC columns already present in v4 files and computes only
the platform-owned P_Solar substitution flag when needed.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root: `import solete` works from any cwd / Spyder
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # sibling script inspect_dataset.py

import h5py
import numpy as np
import pandas as pd


from inspect_dataset import discover_keys  # Phase 1, reused as-is

from solete.paths import resolve_input
from solete.io import import_PV_WT_data
from solete.physics import PV_Performance_Model
from solete.qc import add_substitution_flag
from solete.qc_codes import QC_LABELS, QC_OK

QC_FLAG_NAMES = QC_LABELS


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
    """Read release QC columns and add model substitution on a copy."""
    df = df.copy()

    if "P_Solar[kW]" in df.columns:
        Pac, _, _, _ = PV_Performance_Model(df, pv_info)
        df["Pac"] = Pac
        df["P_Solar_model_substituted"] = df["Pac"] >= 1.5 * df["P_Solar[kW]"]
        add_substitution_flag(df, df["P_Solar_model_substituted"])

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
                    100 * (n_nonnull - counts.get(QC_OK, 0)) / n_nonnull, 3
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
    ap.add_argument("--csv", default=None,
                     help="Output CSV path (default: outputs/availability_report.csv)")
    args = ap.parse_args()

    PV, _WT = import_PV_WT_data()

    combined = pd.concat(
        [report_for_file(str(resolve_input(path)), PV) for path in args.h5_files],
        ignore_index=True,
    )
    from solete.paths import output_path
    csv_path = args.csv or str(output_path("availability_report.csv"))
    combined.to_csv(csv_path, index=False)
    print(f"Wrote {len(combined)} rows to {csv_path}")

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
