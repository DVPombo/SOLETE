"""
Resample the 1-second SOLETE file to 1min/5min/1h, with correct per-column
aggregation (circular mean for angular columns, plain mean for everything
else) and an explicit, documented labeling convention -- so the methodology
is reproducible from this file alone, not something a future reader has to
reverse-engineer.

METHODOLOGY (also written into <out-prefix>_METHODOLOGY.md alongside the
output files):
  - Aggregation window: right-open, left-closed intervals (`closed='left'`),
    labeled by the START of the interval (`label='left'`). E.g. the "1h"
    bucket labeled 2019-01-01 03:00:00 covers [03:00:00, 04:00:00).
  - Aggregation function: arithmetic mean for every column EXCEPT angular
    columns (wind direction, and azimuth if computed at 1-second resolution
    and then resampled rather than recomputed directly at the target
    resolution -- recomputing directly at the target timestamps is
    preferable where possible, see solar_position.py), which use a
    circular mean (vector-average via sin/cos, then atan2 back to degrees,
    wrapped to [0, 360)).
  - This differs from a plain `.resample(...).mean()` specifically for
    angular columns -- averaging 359 deg and 1 deg naively gives 180 deg,
    which is physically wrong; the circular mean correctly gives ~0 deg.
  - `<column>_qc` flag columns (see qc_flags.py) are NOT averaged -- a mean
    of integer codes is meaningless. Each is collapsed to TWO fields per
    bucket instead:
      `<column>_qc_worst`       the single highest-severity code present in
                                 the bucket (QC_SEVERITY_ORDER in
                                 qc_flags.py), so a forecasting user can spot
                                 a bad bucket from one column.
      `<column>_qc_frac_flagged` fraction of the bucket's 1-second samples
                                 that were not QC_OK, so a consumer can pick
                                 their own tolerance threshold instead of
                                 inheriting the resampler's.
    - Model-derived columns and code 6 are excluded. They are recomputed by
        solete.expansion.expand_physical from each target resolution's inputs.

Usage:
    python dataset/pipeline/resample_solete.py SOLETE_clean_1sec.h5 --key DATA --out-prefix SOLETE_clean
    # writes SOLETE_clean_1min.h5, SOLETE_clean_5min.h5, SOLETE_clean_1h.h5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import resolve_input, resolve_output_prefix  # noqa: E402
from solete.qc_codes import (  # noqa: E402
    MODEL_DERIVED_COLUMNS,
    PIPELINE_QC_COLUMNS,
    QC_OK,
    QC_MODEL_SUBSTITUTED,
    QC_SEVERITY_ORDER,
)
try:
    from .solete_report import print_report
except ImportError:
    from solete_report import print_report

# Columns that represent a compass bearing / angle wrapping at 360 degrees.
# Extend this if a column list check reveals others (e.g. a recomputed
# azimuth column, if resampled rather than recomputed directly per bucket).
ANGULAR_COLUMNS = {"WIND_DIR[deg]"}

QC_SUFFIX = "_qc"
_SEVERITY_RANK = {code: i for i, code in enumerate(QC_SEVERITY_ORDER)}  # 0 = worst


def circular_mean_deg(angles_deg: pd.Series) -> pd.Series:
    """Vector-average of angles in degrees, wrapped to [0, 360), resampled
    to `angles_deg`'s own DatetimeIndex-with-freq -- i.e. this expects an
    ALREADY-INDEXED-BY-BUCKET input (sin/cos means), not a raw angle Series.
    Left here only for the module's public name; the actual per-bucket work
    happens in resample_angular_column below via built-in (fast, vectorized)
    mean() on sin/cos, not a per-group Python callback -- see that function's
    docstring for why."""
    raise NotImplementedError("use resample_angular_column")


def resample_angular_column(angles_deg: pd.Series, rule: str) -> pd.Series:
    """Circular mean per bucket, vectorized. Resampling with a custom Python
    function via `.resample(...).agg(some_func)` calls `some_func` once per
    bucket -- hundreds of thousands of times at 1-second source resolution --
    which is orders of magnitude slower than pandas' built-in reductions
    (measured ~50x on a 2M-row synthetic column). So instead: take the
    sin/cos of the angle up front (this is already vectorized, one array op
    over the whole column), resample EACH of those with the built-in `mean`
    (implemented in C, fast), and only then take atan2 of the two resampled
    columns -- one more whole-array op, not a per-bucket one."""
    radians = np.deg2rad(angles_deg.to_numpy(dtype=np.float64))
    sincos = pd.DataFrame({"sin": np.sin(radians), "cos": np.cos(radians)}, index=angles_deg.index)
    resampled = sincos.resample(rule, label="left", closed="left").mean()
    mean_angle = np.degrees(np.arctan2(resampled["sin"].to_numpy(), resampled["cos"].to_numpy()))
    # Floating-point note: for angles that average to ~0 deg, mean_angle can
    # come out as a tiny NEGATIVE epsilon (e.g. -1.4e-15) due to floating-
    # point rounding in the sin/cos round-trip. A single `% 360.0` on a
    # value that close to zero rounds UP to exactly 360.0 rather than down
    # to 0.0 -- which would silently produce the exact same "value == 360"
    # boundary artifact this toolkit exists to catch elsewhere. Applying
    # the modulo twice collapses that edge case to a clean 0.0.
    wrapped = np.mod(np.mod(mean_angle, 360.0), 360.0)
    # An empty bucket has NaN sin AND cos means (mean of nothing), so
    # arctan2(nan, nan) is already nan -- no separate empty-bucket branch
    # needed here, unlike the old per-group version.
    return pd.Series(wrapped, index=resampled.index)


def build_agg_dict(columns) -> dict:
    """Aggregation for the plain-mean DATA columns only -- excludes
    `<column>_qc` (see resample_qc_columns) and angular columns (see
    resample_angular_column, computed separately because a circular mean
    can't be expressed as a single column -> single column .agg() function
    without the per-bucket Python-callback slowdown)."""
    return {
        col: "mean"
        for col in columns
        if (not col.endswith(QC_SUFFIX)
            and col not in ANGULAR_COLUMNS
            and col not in MODEL_DERIVED_COLUMNS)
    }


def resample_dataframe(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    plain_cols = [
        column for column in df.columns
        if (not column.endswith(QC_SUFFIX)
            and column not in ANGULAR_COLUMNS
            and column not in MODEL_DERIVED_COLUMNS)
    ]
    angular_cols = [column for column in df.columns if column in ANGULAR_COLUMNS]
    agg = build_agg_dict(df.columns)
    resampled = df[plain_cols].resample(rule, label="left", closed="left").agg(agg)
    for col in angular_cols:
        resampled[col] = resample_angular_column(df[col], rule)
    if angular_cols:
        output_columns = [
            column for column in df.columns
            if not column.endswith(QC_SUFFIX) and column not in MODEL_DERIVED_COLUMNS
        ]
        resampled = resampled[output_columns]
    qc_cols = [column for column in df.columns if column in PIPELINE_QC_COLUMNS]
    if qc_cols:
        qc_resampled = resample_qc_columns(df[qc_cols], rule)
        resampled = resampled.join(qc_resampled, how="left")
    return resampled


def resample_qc_columns(qc_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """For each `<column>_qc` column, produce `<column>_qc_worst` (highest-
    severity code in the bucket) and `<column>_qc_frac_flagged` (fraction of
    seconds in the bucket that were not QC_OK). See module docstring.

    Vectorized the same way as resample_angular_column, and for the same
    reason (a per-bucket Python callback here was the actual cause of a
    ~15-minute hang measured on the real 39M-row file -- if you're wondering why
    this function doesn't just call `.resample(rule).apply(...)`, which is
    the natural-looking way to write it and about 50x slower here):
      - `_frac_flagged`: cast the whole `!= QC_OK` boolean column to float
        ONCE (one vectorized array op), then a plain built-in `mean()`
        resample gives the per-bucket fraction directly -- no custom
        function at all.
      - `_worst`: map codes to a severity RANK once (`Series.map` over the
        whole column, still just one vectorized pass, not per-bucket), take
        the built-in (fast) `min()` per bucket on the numeric rank, then map
        the small number of resulting rank values back to QC codes.
    An empty bucket (zero 1-second samples, a real source gap) gets NaN in
    both fields from the mean()/min() of nothing, automatically -- no
    separate branch needed, unlike the old per-group version.
    """
    qc_df = qc_df.replace(QC_MODEL_SUBSTITUTED, QC_OK)
    is_flagged = (qc_df != QC_OK).astype(np.float64)
    frac = is_flagged.resample(rule, label="left", closed="left").mean()
    frac.columns = [f"{c}_frac_flagged" for c in qc_df.columns]

    rank_to_code = {rank: code for code, rank in _SEVERITY_RANK.items()}
    ranks = qc_df.apply(lambda s: s.map(_SEVERITY_RANK).astype(np.float64))
    worst_rank = ranks.resample(rule, label="left", closed="left").min()
    worst = worst_rank.apply(lambda s: s.map(rank_to_code))
    worst.columns = [f"{c}_worst" for c in qc_df.columns]

    return worst.join(frac)


def write_methodology_doc(path: str, columns_seen) -> None:
    data_cols = [
        column for column in columns_seen
        if not column.endswith(QC_SUFFIX) and column not in MODEL_DERIVED_COLUMNS
    ]
    qc_cols = [column for column in columns_seen if column in PIPELINE_QC_COLUMNS]
    angular_present = sorted(ANGULAR_COLUMNS & set(data_cols))
    other_present = sorted(set(data_cols) - ANGULAR_COLUMNS)
    with open(path, "w") as f:
        f.write("# Resampling methodology (generated by resample_solete.py)\n\n")
        f.write("- Window convention: `closed='left'`, `label='left'` "
                "(a bucket labeled T covers [T, T+period)).\n")
        f.write(f"- Circular-mean aggregation applied to: {angular_present or 'none found'}\n")
        f.write(f"- Plain arithmetic mean applied to: {other_present}\n")
        if qc_cols:
            f.write(
                f"- QC flag columns ({sorted(qc_cols)}) are NOT averaged. Each "
                "`<column>_qc` becomes two fields per bucket: `<column>_qc_worst` "
                "(highest-severity code present, per QC_SEVERITY_ORDER in "
                "solete/qc_codes.py) and `<column>_qc_frac_flagged` (fraction of seconds "
                "in the bucket that were not QC_OK). A bucket with zero source "
                "samples (a real gap) gets NaN in both, distinguishable from a "
                "genuinely all-OK bucket (`_worst` == 0, `_frac_flagged` == 0.0).\n"
            )
        excluded = sorted(set(columns_seen) & MODEL_DERIVED_COLUMNS)
        if excluded:
            f.write(
                f"- Model-derived columns ({excluded}) are excluded from resampling "
                "and must be recomputed at this resolution with expand_physical.\n"
            )
        f.write(
            "- If comparing against a previously-published file and values don't "
            "match, check that file's own labeling convention before concluding "
            "there's a real data discrepancy -- see compare_resolutions.py, which "
            "checks for exactly this before flagging anything as a genuine bug.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--key", default="DATA")
    parser.add_argument("--out-prefix", default="SOLETE_clean")
    parser.add_argument("--rules", nargs="+", default=["1min", "5min", "1h"])
    args = parser.parse_args()
    args.path = str(resolve_input(args.path))                 # bare names are looked up in data/hdf5/
    args.out_prefix = resolve_output_prefix(args.out_prefix)  # bare prefixes are written to data/hdf5/

    print(f"Loading {args.path} ... (this may take a while / significant RAM for a 3GB file)")
    df = pd.read_hdf(args.path, key=args.key)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

    span_start, span_end = df.index.min(), df.index.max()
    data_cols = [c for c in df.columns if not c.endswith(QC_SUFFIX)]
    qc_cols = [c for c in df.columns if c.endswith(QC_SUFFIX)]
    angular_present = sorted(ANGULAR_COLUMNS & set(data_cols))

    per_resolution_report = {}
    for rule in args.rules:
        t0 = time.time()
        print(f"Resampling to {rule} ...")
        out = resample_dataframe(df, rule)
        print(f"  aggregated in {time.time() - t0:.1f}s ({len(out):,} buckets)")
        out_path = f"{args.out_prefix}_{rule}.h5"
        out.to_hdf(out_path, key="DATA", mode="w")
        print(f"  wrote {out_path} ({len(out):,} rows, {time.time() - t0:.1f}s total)")

        # Expected row count from the actual time span, for a quick sanity
        # check against what was actually written -- a mismatch here means
        # either an unexpected gap pattern in the source, or a resample()
        # edge-case at the very first/last bucket worth looking at directly.
        expected_rows = int((span_end - span_start) / pd.Timedelta(rule)) + 1
        # Per-column count of buckets that came out fully empty (NaN) --
        # i.e. a period with zero 1-second samples in it at all. Non-zero
        # here means there's an actual gap in the source data, not just a
        # value that happens to be reported as zero.
        nan_introduced = out.isna().sum().to_dict()

        per_resolution_report[rule] = {
            "output_file": out_path,
            "rows_written": len(out),
            "expected_rows_from_time_span": expected_rows,
            "row_count_matches_expected": len(out) == expected_rows,
            "nan_buckets_per_column": nan_introduced,
        }

    write_methodology_doc(f"{args.out_prefix}_METHODOLOGY.md", df.columns)
    print(f"Wrote {args.out_prefix}_METHODOLOGY.md")

    print_report(
        "resample_solete_summary",
        {
            "source_file": args.path,
            "source_n_rows": len(df),
            "source_span_start": str(span_start),
            "source_span_end": str(span_end),
            "angular_columns_circular_mean": angular_present,
            "plain_mean_columns": sorted(set(data_cols) - set(angular_present)),
            "qc_columns_aggregated_worst_and_frac_flagged": qc_cols,
            "per_resolution": per_resolution_report,
        },
    )


if __name__ == "__main__":
    main()
