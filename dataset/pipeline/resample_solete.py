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
    columns, which use a circular mean (vector-average via sin/cos, then
    atan2 back to degrees): WIND_DIR[deg], wrapped to [0, 360), and
    Azimuth[deg] (south-referenced, as recomputed at 1 s), wrapped to
    [-180, 180). A plain mean of the azimuth is wrong in the bucket that
    contains solar midnight, where the 1 s values jump from +180 to -180.
    Elevation[deg] is not angular in that sense (-90..90) and uses the plain mean.
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

  - Model-derived columns (Pac, P_hybrid[kW], code 6, ...) are skipped, never averaged up: they are
    computed per resolution afterwards (solete/expansion.py). A code that is not pipeline-owned inside a
    pipeline `<column>_qc` column is replaced by QC_OK before aggregation.

HOW IT RUNS: the cleaned 1 s file is read in slices of whole days (--slice-days, default 31; a day boundary
is a bucket boundary at every resolution), each slice is resampled by `resample_dataframe`, and the pieces
are joined; empty buckets inside the grid stay NaN exactly as in a whole-file resample. Only the (small)
resampled frames are ever held whole. tests/test_release_build.py proves sliced == whole-file.

Usage:
    python dataset/pipeline/resample_solete.py SOLETE_Pombo_1sec_cleaned_scratch.h5 --out-prefix SOLETE_resampled_scratch
    # writes <prefix>_1min.h5, <prefix>_5min.h5, <prefix>_60min.h5 (fixed format, measured + pipeline flags only)
    # `1h` is accepted as an alias of `60min` in --rules.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import normalize_resolution, resolve_input, resolve_output_prefix  # noqa: E402
from solete.h5io import h5_info, read_index, read_rows, write_fixed  # noqa: E402
from qc_flags import MODEL_DERIVED_COLUMNS, PIPELINE_OWNED_CODES, QC_OK, QC_SEVERITY_ORDER
from solete_report import print_report

# Columns that represent a compass bearing / angle wrapping at 360 degrees.
# Extend this if a column list check reveals others (e.g. a recomputed
# azimuth column, if resampled rather than recomputed directly per bucket).
# column -> (lowest, upper-exclusive) of the range its circular mean is wrapped into
ANGULAR_COLUMNS = {"WIND_DIR[deg]": (0.0, 360.0), "Azimuth[deg]": (-180.0, 180.0)}

QC_SUFFIX = "_qc"

# RESOLUTION RULE (docs: dataset/docs/METHODOLOGY.md, "Model columns are per resolution").
# Only MEASURED columns and PIPELINE-OWNED flags are resampled from the 1-second data.
# Everything in MODEL_DERIVED_COLUMNS (Pac, Pdc, TempModule, TempCell, P_Solar_clean[kW],
# P_hybrid[kW], their flags, and code 6 = QC_MODEL_SUBSTITUTED) is computed at each resolution
# by solete.expansion.expand_physical from that resolution's own cleaned inputs. It is dropped
# here even if a 1-second input already contains it, so it can never be averaged upward.
SKIP_MODEL_COLUMNS = frozenset(MODEL_DERIVED_COLUMNS)
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


def resample_angular_column(angles_deg: pd.Series, rule: str, wrap=(0.0, 360.0)) -> pd.Series:
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
    lo = wrap[0]
    wrapped = np.mod(np.mod(mean_angle - lo, 360.0), 360.0) + lo
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
        if not col.endswith(QC_SUFFIX) and col not in ANGULAR_COLUMNS
    }


def resampleable_columns(columns) -> list:
    """Columns that may be resampled: everything except the model-derived columns."""
    return [c for c in columns if c not in SKIP_MODEL_COLUMNS]


def neutralise_foreign_codes(qc_df: pd.DataFrame):
    """Pipeline-owned flag columns may only hold PIPELINE_OWNED_CODES. Any other code (a code 6
    from a previously expanded file, say) is replaced by QC_OK before aggregation so it can never
    reach `_qc_worst`. Returns (cleaned frame, {column: n_replaced})."""
    ok = qc_df.isin(PIPELINE_OWNED_CODES)
    replaced = {c: int((~ok[c]).sum()) for c in qc_df.columns if (~ok[c]).any()}
    if replaced:
        print(f"WARNING: foreign (non-pipeline) flag codes replaced by QC_OK before resampling: {replaced}")
        qc_df = qc_df.where(ok, QC_OK)
    return qc_df, replaced


def resample_dataframe(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    skipped = [c for c in df.columns if c in SKIP_MODEL_COLUMNS]
    if skipped:
        df = df[resampleable_columns(df.columns)]
    plain_cols = [c for c in df.columns if not c.endswith(QC_SUFFIX) and c not in ANGULAR_COLUMNS]
    angular_cols = [c for c in df.columns if c in ANGULAR_COLUMNS]
    agg = build_agg_dict(df.columns)
    resampled = df[plain_cols].resample(rule, label="left", closed="left").agg(agg)
    for col in angular_cols:
        resampled[col] = resample_angular_column(df[col], rule, ANGULAR_COLUMNS[col])
    if angular_cols:
        resampled = resampled[[c for c in df.columns if not c.endswith(QC_SUFFIX)]]  # restore original column order
    qc_cols = [c for c in df.columns if c.endswith(QC_SUFFIX)]
    if qc_cols:
        qc_clean, _ = neutralise_foreign_codes(df[qc_cols])
        qc_resampled = resample_qc_columns(qc_clean, rule)
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
    data_cols = [c for c in columns_seen if not c.endswith(QC_SUFFIX)]
    qc_cols = [c for c in columns_seen if c.endswith(QC_SUFFIX)]
    angular_present = sorted(set(ANGULAR_COLUMNS) & set(data_cols))
    other_present = sorted(set(data_cols) - set(ANGULAR_COLUMNS))
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
                "qc_flags.py) and `<column>_qc_frac_flagged` (fraction of seconds "
                "in the bucket that were not QC_OK). A bucket with zero source "
                "samples (a real gap) gets NaN in both, distinguishable from a "
                "genuinely all-OK bucket (`_worst` == 0, `_frac_flagged` == 0.0).\n"
            )
        f.write(
            "- Model-derived columns (solete.qc_codes.MODEL_DERIVED_COLUMNS: Pac, Pdc, TempModule, "
            "TempCell, P_Solar_clean[kW], P_hybrid[kW] and their flags, incl. code 6) are NOT resampled: "
            "they are computed at each resolution from that resolution's own cleaned inputs by "
            "solete.expansion.expand_physical.\n"
        )
        f.write(
            "- If comparing against a previously-published file and values don't "
            "match, check that file's own labeling convention before concluding "
            "there's a real data discrepancy -- see compare_resolutions.py, which "
            "checks for exactly this before flagging anything as a genuine bug.\n"
        )


def resolution_grid(first, last, rule):
    """Every bucket label a whole-file resample of data spanning [first, last] produces."""
    return pd.date_range(first.floor(rule), last.floor(rule), freq=rule)


def day_slices(index, slice_days):
    """[(start, stop), ...] row positions covering `index` (sorted), each cut on a midnight."""
    first = index[0].normalize()
    last = index[-1].normalize()
    edges = pd.date_range(first, last + pd.Timedelta(days=1), freq=f"{int(slice_days)}D")
    if edges[-1] <= last:
        edges = edges.append(pd.DatetimeIndex([last + pd.Timedelta(days=1)]))
    pos = index.searchsorted(edges, side="left")
    return [(int(a), int(b)) for a, b in zip(pos[:-1], pos[1:]) if b > a], edges


def resample_file(path, key, rules, slice_days=31, verbose=True):
    """Resample a sorted cleaned 1 s file (table or fixed format) to each of `rules`, slice by slice on
    day boundaries. Returns ({rule: DataFrame}, {rule: sorted unique foreign-code warnings})."""
    rules = [normalize_resolution(r) for r in rules]
    info = h5_info(path, key)
    idx = read_index(path, key)
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{path} is not in chronological order")
    slices, _ = day_slices(idx, slice_days)
    grids = {r: resolution_grid(idx[0], idx[-1], r) for r in rules}
    pieces = {r: [] for r in rules}
    for i, (a, b) in enumerate(slices, 1):
        block = read_rows(path, key, a, b)
        for r in rules:
            pieces[r].append(resample_dataframe(block, r))
        if verbose:
            print(f"  slice {i}/{len(slices)}: rows {a:,}..{b:,} resampled", flush=True)
        del block
    # buckets that no slice produced (a whole day missing in the source) exist in a whole-file resample as
    # NaN rows; the global grid puts them back, and nothing outside the data's own span
    return {r: pd.concat(pieces[r]).reindex(grids[r]) for r in rules}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--key", default="DATA")
    parser.add_argument("--out-prefix", default="SOLETE_resampled")
    parser.add_argument("--rules", nargs="+", default=["1min", "5min", "60min"])
    parser.add_argument("--slice-days", type=int, default=31)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.path = str(resolve_input(args.path))                 # bare names are looked up in data/hdf5/
    args.out_prefix = resolve_output_prefix(args.out_prefix)  # bare prefixes are written to data/hdf5/
    rules = [normalize_resolution(r) for r in args.rules]

    t0 = time.time()
    info = h5_info(args.path, args.key)
    idx = read_index(args.path, args.key)
    span_start, span_end = idx[0], idx[-1]
    n_source = len(idx)
    del idx
    columns = info["columns"]
    print(f"Resampling {args.path}: {n_source:,} rows, {len(columns)} columns, slices of {args.slice_days} days")
    skipped_model_cols = [c for c in columns if c in SKIP_MODEL_COLUMNS]
    if skipped_model_cols:
        print(f"Skipping model-derived columns (computed per resolution, never resampled): {skipped_model_cols}")
    kept_cols = resampleable_columns(columns)
    data_cols = [c for c in kept_cols if not c.endswith(QC_SUFFIX)]
    qc_cols = [c for c in kept_cols if c.endswith(QC_SUFFIX)]
    angular_present = sorted(set(ANGULAR_COLUMNS) & set(data_cols))

    results = resample_file(args.path, args.key, rules, args.slice_days)
    per_resolution_report = {}
    for rule in rules:
        out = results[rule]
        out_path = f"{args.out_prefix}_{rule}.h5"
        write_fixed(out, out_path, args.key, overwrite=args.overwrite)
        print(f"  wrote {out_path} ({len(out):,} rows)")
        expected_rows = int((span_end.floor(rule) - span_start.floor(rule)) / pd.Timedelta(rule)) + 1
        per_resolution_report[rule] = {
            "output_file": out_path,
            "rows_written": len(out),
            "expected_rows_from_time_span": expected_rows,
            "row_count_matches_expected": len(out) == expected_rows,
            "nan_buckets_per_column": out.isna().sum().to_dict(),
        }

    write_methodology_doc(f"{args.out_prefix}_METHODOLOGY.md", kept_cols)
    print(f"Wrote {args.out_prefix}_METHODOLOGY.md")

    print_report(
        "resample_solete_summary",
        {
            "source_file": args.path,
            "source_n_rows": n_source,
            "source_span_start": str(span_start),
            "source_span_end": str(span_end),
            "elapsed_seconds": round(time.time() - t0, 1),
            "angular_columns_circular_mean": angular_present,
            "plain_mean_columns": sorted(set(data_cols) - set(angular_present)),
            "qc_columns_aggregated_worst_and_frac_flagged": qc_cols,
            "model_columns_skipped_not_resampled": skipped_model_cols,
            "per_resolution": per_resolution_report,
        },
    )


if __name__ == "__main__":
    main()
