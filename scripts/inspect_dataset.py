# -*- coding: utf-8 -*-
"""
inspect_dataset.py

Standing inspection tool for SOLETE .h5 files. For every key/table found in
the given HDF5 file, reports per-column: dtype, min/max/mean, null count,
and a few example values. Also reports index characteristics (monotonicity,
duplicates, span) since SOLETE is a time-indexed dataset and that matters
as much as the column stats.

Usage
-----
    python scripts/inspect_dataset.py SOLETE_short.h5
    python scripts/inspect_dataset.py SOLETE_Pombo_60min.h5
    python scripts/inspect_dataset.py SOLETE_Pombo_60min.h5 --csv out.csv

This was written for Phase 1 (data-dictionary) work and is meant to be kept
around as a standing tool -- e.g. to re-run against a newly delivered
resolution file, or after a data refresh, to see what changed.
"""
import argparse
import sys

import h5py
import numpy as np
import pandas as pd


def discover_keys(path):
    """Return the list of top-level pandas HDF5 store keys in the file."""
    keys = []
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            # pandas 'fixed' format stores write a Group per key, containing
            # axis0/axis1/block*_items/block*_values datasets. We only want
            # the group itself (the key), not its internal datasets.
            if isinstance(obj, h5py.Group):
                children = set(obj.keys())
                if children & {"axis0", "axis1"} or any(
                    c.startswith("block") for c in children
                ):
                    keys.append(name)

        h.visititems(visit)
    return keys or ["/"]  # fall back if structure isn't a recognizable pandas store


def describe_column(s: pd.Series) -> dict:
    n_null = int(s.isna().sum())
    n = len(s)
    row = {
        "column": s.name,
        "dtype": str(s.dtype),
        "n_rows": n,
        "n_null": n_null,
        "pct_null": round(100 * n_null / n, 3) if n else float("nan"),
    }
    non_null = s.dropna()
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        row["min"] = non_null.min() if len(non_null) else float("nan")
        row["max"] = non_null.max() if len(non_null) else float("nan")
        row["mean"] = non_null.mean() if len(non_null) else float("nan")
        row["n_negative"] = int((non_null < 0).sum()) if len(non_null) else 0
    elif pd.api.types.is_bool_dtype(s):
        row["min"] = bool(non_null.min()) if len(non_null) else None
        row["max"] = bool(non_null.max()) if len(non_null) else None
        row["mean"] = round(float(non_null.mean()), 4) if len(non_null) else float("nan")
        row["n_negative"] = 0
    else:
        row["min"] = row["max"] = row["mean"] = row["n_negative"] = None

    examples = non_null.iloc[:3].tolist() if len(non_null) else []
    row["examples"] = examples
    return row


def inspect_file(path: str) -> pd.DataFrame:
    print(f"\n{'=' * 70}\n{path}\n{'=' * 70}")
    keys = discover_keys(path)
    all_rows = []

    for key in keys:
        try:
            df = pd.read_hdf(path, key=key) if key != "/" else pd.read_hdf(path)
        except Exception as e:
            print(f"  [skip] could not read key '{key}': {e}")
            continue

        print(f"\n--- key: {key!r} | shape: {df.shape} ---")

        # Index characteristics -- important for a time-series dataset.
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"  index: DatetimeIndex, {df.index.min()} .. {df.index.max()}")
            print(f"  index monotonic increasing: {df.index.is_monotonic_increasing}")
            n_dup = int(df.index.duplicated().sum())
            print(f"  duplicated index timestamps: {n_dup}")
        else:
            print(f"  index: {type(df.index).__name__}, {len(df.index)} entries")

        for col in df.columns:
            row = describe_column(df[col])
            row["key"] = key
            all_rows.append(row)

        summary = pd.DataFrame(
            [r for r in all_rows if r["key"] == key]
        ).drop(columns=["key"])
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(summary.to_string(index=False))

    return pd.DataFrame(all_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5_files", nargs="+", help="One or more .h5 files to inspect")
    ap.add_argument("--csv", help="Optional path to write the combined summary as CSV")
    args = ap.parse_args()

    combined = []
    for path in args.h5_files:
        result = inspect_file(path)
        result.insert(0, "file", path)
        combined.append(result)

    combined_df = pd.concat(combined, ignore_index=True)

    if args.csv:
        combined_df.to_csv(args.csv, index=False)
        print(f"\nWrote combined summary to {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
