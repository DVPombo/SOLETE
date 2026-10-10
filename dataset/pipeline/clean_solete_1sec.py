"""
clean_solete_1sec.py -- build the cleaned, chronologically-sorted 1-second
file with per-column QC flags merged in. See docs/CLEANING_DECISIONS.md for
the reasoning behind every choice below -- several are provisional and marked
as such.

WHAT THIS DOES, IN ORDER:
  1. Rows must be chronological. The v4 `_original` file is (build_release.py stage
     `original` makes it from the raw v3 file, whose 457 daily blocks are shuffled;
     nothing is added or removed -- see followup_A). An unsorted input is refused
     unless --in-memory is given, which sorts the whole file first (about 7 GB RAM).
  2. WIND_DIR[deg] := value mod 360 (>360 readings are extra sensor
     revolutions, not different bearings -- confirmed by the maintainer).
  3. Pressure[mbar]: replaces placeholder/sentinel values with NaN. A sample
     is a sentinel if it's an exact multiple of 1000 (any count -- catches
     1000.0/2000.0 blocks AND isolated 3000.0 glitches), OR if it sits in a
     run of >= --pressure-flatline-min-run (default 300) consecutive
     bit-for-bit identical samples (catches non-round stuck-sensor plateaus,
     e.g. a value like 997.5 held constant for a full hour -- real pressure
     never does that). The first real run against the actual file caught
     1000.0/2000.0 via the multiples test but missed 997.5-type plateaus
     under an earlier round-numbers-only test; this version catches both.
     Prints what it found in both categories -- check before trusting it.
  4. WIND_SPEED[m1s] & HUMIDITY[%]: seconds where both are exactly 0 at once
     (near-certain sensor dropout, not real calm-and-bone-dry air) are
     linearly interpolated when the run is <= --dropout-max-run seconds
     (default 5, per the maintainer). Longer runs are left as-is and
     flagged QC_DROPOUT_LONG_UNTREATED for manual review.
  5. Generic "logger glitch" pass: any single-column value outside the
     physical bound in BOUNDS below is treated the same way -- runs
     <= --glitch-max-run seconds (default 3) get linearly interpolated
     between the valid values just outside the run; longer runs are set to
     NaN and flagged QC_GLITCH_LONG_UNTREATED_NAN. This is what happens to
     the ~1-day humidity anomaly on 2018-11-17 -- too long to interpolate
     without manufacturing data, and physically impossible to keep.
  6. P_Gaia[kW]: VALUES ARE NOT CHANGED (per the maintainer: "that's what we
     honestly have"). Every row is flagged instead: QC_ACTIVE_DAY on
     2018-08-31 and 2019-05-25 (confirmed real telemetry),
     QC_UNVERIFIED_PROVENANCE everywhere else -- recorded as 0, but whether
     that means the turbine was offline or the channel wasn't being logged
     is not known from this file.
  7. Azimuth[deg] / Elevation[deg]: any published values are DROPPED and
     replaced with a fresh pvlib (NREL SPA) computation for every row, in
     UTC (see solar_position.py's timezone finding) with the south-
     referenced azimuth convention (0=south, east negative) to match the
     original column's semantics. Elevation is NOT clipped at night -- it
     goes negative below the horizon (physically correct), unlike the
     published file which showed 0 then. The input does not need these two
     columns (the v4 `_original` file does not have them). No flag column is
     written for them: the flag would be a constant 8 on every row (decision
     D1, see docs/QC_SCHEMA.md 3b); code 8 stays reserved in solete/qc_codes.py.

Note on step 5 vs step 4: the generic glitch pass NaNs out long-untreated
runs (QC_GLITCH_LONG_UNTREATED_NAN) because those values are physically
impossible by construction -- leaving e.g. a -40.1degC reading in the data
would be worse than admitting it's missing. The WS/HUM dropout pass does
NOT NaN its long-untreated runs (QC_DROPOUT_LONG_UNTREATED) -- 0 m/s wind
and 0% humidity are each individually plausible values, it's only the
*combination* that's suspicious, so the recorded value is kept and flagged
for manual review rather than destroyed. See qc_flags.fix_short_runs.

HOW IT RUNS (slices): every rule lives in `clean_block`, which cleans one chronologically sorted
block of rows. The file is cut into slices of about --slice-days days, each slice is cleaned and
appended to the output table, so peak memory is a few hundred MB per slice instead of ~7 GB.
A cut is only made BETWEEN two rows that no run-based rule touches (`find_safe_cut`, built from the
same masks the rules use): no pressure flatline, WS/HUM dropout run or out-of-bound glitch run
can cross it, and the rows next to it are never a run's interpolation anchor. Under that condition
the sliced result equals the whole-file result exactly (tests/test_release_build.py proves it,
with runs placed across the natural cut points and tiny slices).

OUTPUT:
  <out-prefix>.h5   cleaned data, HDF5 table format: the input's measured columns, then
                     Azimuth[deg], Elevation[deg] (recomputed), then one int8 `<column>_qc`
                     column per treated measured column (WIND_DIR, Pressure, WIND_SPEED, HUMIDITY,
                     TEMPERATURE, GHI, POA Irr, P_Gaia: 8 flag columns; see qc_flags.py), merged
                     into the same DataFrame -- not a separate companion file.

USAGE (Spyder or a terminal):
    python dataset/pipeline/clean_solete_1sec.py SOLETE_Pombo_1sec_original_v4.h5 \\
        --out-prefix SOLETE_Pombo_1sec_cleaned_scratch

A JSON summary is printed at the end of the run (samples touched per rule,
flag value counts, verification checks). Compare it with
docs/CLEANING_DECISIONS.md before trusting a new output file.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import resolve_input, resolve_output_prefix  # noqa: E402
from solete.h5io import TableWriter, h5_info, read_index, read_rows  # noqa: E402
from qc_flags import (  # noqa: E402
    QC_ACTIVE_DAY,
    QC_DROPOUT_LONG_UNTREATED,
    QC_DROPOUT_SHORT_FIXED,
    QC_GLITCH_LONG_UNTREATED_NAN,
    QC_GLITCH_SHORT_FIXED,
    QC_OK,
    QC_PLACEHOLDER,
    QC_UNVERIFIED_PROVENANCE,
    QC_WRAPPED,
    assert_pipeline_codes,
    detect_flatline_mask,
    fix_short_runs,
)
from solar_position import compute_solar_position_bulk  # noqa: E402
from solete_report import print_report  # noqa: E402

WD, WS, HUM, PRES, TEMP, GHI, POA = (
    "WIND_DIR[deg]", "WIND_SPEED[m1s]", "HUMIDITY[%]", "Pressure[mbar]",
    "TEMPERATURE[degC]", "GHI[kW1m2]", "POA Irr[kW1m2]",
)
AZ, EL, PG = "Azimuth[deg]", "Elevation[deg]", "P_Gaia[kW]"

# The columns that get a `<column>_qc` flag, in output order. Azimuth/Elevation have none (decision D1:
# their flag would be a constant 8 on every row).
QC_COLS = [WD, PRES, WS, HUM, TEMP, GHI, POA, PG]

# Provisional physical bounds for the generic glitch pass. Values OUTSIDE
# [low, high] are treated as logger glitches (step 5). None on either side
# means that side isn't checked here. These are NOT validated against site
# records -- sanity-check them against the printed per-column counts.
BOUNDS = {
    TEMP: (-25.0, 40.0),      # only known violation so far: the single -40.1 second
    HUM: (None, 1.0),         # stored as a 0-1 fraction; lower bound handled by the WS/HUM dropout step instead
    PRES: (900.0, 1100.0),    # checked AFTER placeholder sentinels are removed
    GHI: (0.0, 1.5),
    POA: (0.0, 1.6),
}
P_GAIA_ACTIVE_DAYS = ["2018-08-31", "2019-05-25"]  # confirmed real telemetry (followup_C)
HUM_DROPOUT_TOL = 0.05  # HUMIDITY below this, together with WIND_SPEED == 0, counts as joint dropout
DEFAULT_SLICE_DAYS = 31


def reorder_chronologically(df):
    """Sort `df` by its index, one column at a time, to avoid the memory
    spike of sorting the whole block manager at once (only used by --in-memory)."""
    order = np.argsort(df.index.values, kind="stable")
    new_index = pd.DatetimeIndex(df.index.values[order])
    for c in df.columns:
        df[c] = df[c].to_numpy()[order]
    df.index = new_index
    return df


# ---------------------------------------------------------------------------------------------
# masks shared by the rules and by the cut finder (one definition each)
# ---------------------------------------------------------------------------------------------
def pressure_multiples_of_1000(v):
    """True where a pressure sample is an exact multiple of 1000 (a sentinel, whatever the run length)."""
    return np.isclose(np.mod(v, 1000.0), 0.0, atol=1e-6)


def dropout_bad_mask(ws, hum):
    """Joint WIND_SPEED/HUMIDITY dropout: WIND_SPEED exactly 0 together with HUMIDITY below HUM_DROPOUT_TOL."""
    return (np.asarray(ws) == 0) & (np.asarray(hum) < HUM_DROPOUT_TOL)


def glitch_bad_mask(v, low, high):
    """Finite values outside [low, high] (None = side not checked)."""
    v = np.asarray(v, dtype=np.float64)
    bad = np.zeros(len(v), dtype=bool)
    finite = np.isfinite(v)
    if low is not None:
        bad |= finite & (v < low)
    if high is not None:
        bad |= finite & (v > high)
    return bad


def apply_pressure_sentinels(pres, report, flatline_min_run=300):
    """
    A pressure sample is treated as a placeholder/sentinel, not a real
    reading, if EITHER:
      (a) it is an exact multiple of 1000 (1000.0, 2000.0, 3000.0, ...),
          regardless of how many times it occurs -- real atmospheric
          pressure is never a round thousand, so even a single occurrence
          is trusted as a sentinel; or
      (b) it sits inside a run of >= `flatline_min_run` consecutive
          bit-for-bit identical samples -- real pressure drifts second to
          second, so a long flatline (e.g. a plateau at 997.5 mbar for a
          full hour) is itself evidence of a stuck sensor, independent of
          whether the value looks round.
    (a) alone would miss non-round plateaus; (b) alone would miss isolated
    round-thousand glitches too short to register as a "run". Together they
    caught, on the real file, both the 1000.0/2000.0 blocks AND several
    non-round hour-plus plateaus that a round-number-only test had missed.
    """
    v = pres.to_numpy(dtype=np.float64)
    multiples_of_1000 = pressure_multiples_of_1000(v)
    flatline = detect_flatline_mask(v, min_run=flatline_min_run)
    mask = multiples_of_1000 | flatline
    report["pressure_n_flagged_exact_multiples_of_1000"] = int(multiples_of_1000.sum())
    report["pressure_n_flagged_flatline_only"] = int((flatline & ~multiples_of_1000).sum())
    if (flatline & ~multiples_of_1000).any():
        vals, counts = np.unique(v[flatline & ~multiples_of_1000], return_counts=True)
        report["pressure_flatline_only_example_values"] = {
            float(val): int(c) for val, c in sorted(zip(vals, counts), key=lambda t: -t[1])[:10]
        }
    out = v.copy()
    out[mask] = np.nan
    flag = np.where(mask, QC_PLACEHOLDER, QC_OK)
    report["pressure_n_set_to_nan"] = int(mask.sum())
    # kept as a SET of days so slices can be merged exactly (turned into a count in finish_report)
    report["_pressure_days_with_real_value"] = (
        sorted(str(d.date()) for d in pres.index[~mask].normalize().unique()) if (~mask).any() else [])
    return out, flag


def apply_dropout_fix(ws, hum, max_run, report):
    bad = dropout_bad_mask(ws.to_numpy(), hum.to_numpy())
    ws_out, ws_flag = fix_short_runs(ws.to_numpy(), bad, max_run, QC_DROPOUT_SHORT_FIXED, QC_DROPOUT_LONG_UNTREATED)
    hum_out, hum_flag = fix_short_runs(hum.to_numpy(), bad, max_run, QC_DROPOUT_SHORT_FIXED, QC_DROPOUT_LONG_UNTREATED)
    report["dropout_n_seconds_flagged"] = int(bad.sum())
    report["dropout_n_fixed"] = int((ws_flag == QC_DROPOUT_SHORT_FIXED).sum())
    report["dropout_n_untreated"] = int((ws_flag == QC_DROPOUT_LONG_UNTREATED).sum())
    return ws_out, ws_flag, hum_out, hum_flag


def apply_glitch_fix(col, values, low, high, max_run, report):
    v = values.to_numpy(dtype=np.float64) if hasattr(values, "to_numpy") else np.asarray(values, dtype=np.float64)
    bad = glitch_bad_mask(v, low, high)
    # nan_long_runs=True: these are physically-impossible values by
    # construction (outside BOUNDS); a run too long to interpolate gets
    # NaN'd rather than left in the data. See qc_flags.fix_short_runs.
    out, flag = fix_short_runs(
        v, bad, max_run, QC_GLITCH_SHORT_FIXED, QC_GLITCH_LONG_UNTREATED_NAN, nan_long_runs=True
    )
    report[col] = {
        "n_flagged": int(bad.sum()),
        "n_fixed": int((flag == QC_GLITCH_SHORT_FIXED).sum()),
        "n_untreated_set_to_nan": int((flag == QC_GLITCH_LONG_UNTREATED_NAN).sum()),
    }
    return out, flag


def flag_p_gaia(index, active_days, report):
    day = index.normalize()
    is_active = day.isin(pd.DatetimeIndex(active_days))
    flag = np.where(is_active, QC_ACTIVE_DAY, QC_UNVERIFIED_PROVENANCE)
    report["n_rows_active_day"] = int(is_active.sum())
    report["n_rows_unverified_provenance"] = int((~is_active).sum())
    return flag


# ---------------------------------------------------------------------------------------------
# the cleaning rules, applied once, to one chronologically sorted block of rows
# ---------------------------------------------------------------------------------------------
def clean_block(df, dropout_max_run=5, glitch_max_run=3, flatline_min_run=300, chunk_rows=2_000_000, verbose=True):
    """Apply every cleaning rule to `df` (rows already in chronological order) and return
    (cleaned frame, report). `df` is not modified.

    Output columns: the input's measured columns in their input order (Azimuth/Elevation, if the input has
    them, are dropped and recomputed), then Azimuth[deg], Elevation[deg], then the int8 `<column>_qc` columns
    in QC_COLS order. The report holds additive counts (merge with merge_reports)."""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    measured = [c for c in df.columns if c not in (AZ, EL)]
    df = df[measured].copy()
    n = len(df)
    report = {"n_rows": n}

    # QC columns are tracked separately during processing (int64 is easier to do arithmetic/comparisons on)
    # and only cast down to int8 -- the release dtype -- once every code is final.
    qc = pd.DataFrame(QC_OK, index=df.index, columns=QC_COLS, dtype=np.int64)

    log("Wrapping WIND_DIR ...")
    wd = df[WD].to_numpy()
    wrapped = np.mod(wd, 360.0)
    qc[WD] = np.where(wrapped != wd, QC_WRAPPED, QC_OK)
    df[WD] = wrapped
    report["wind_dir_n_wrapped"] = int((wrapped != wd).sum())

    log("Pressure sentinels ...")
    df[PRES], qc[PRES] = apply_pressure_sentinels(df[PRES], report, flatline_min_run)

    log("WIND_SPEED/HUMIDITY joint dropout ...")
    dropout_report = {}
    df[WS], qc[WS], df[HUM], qc[HUM] = apply_dropout_fix(df[WS], df[HUM], dropout_max_run, dropout_report)
    report["dropout"] = dropout_report

    log("Generic glitch pass ...")
    glitch_report = {}
    for col, (low, high) in BOUNDS.items():
        vals = df[col]
        # For humidity/pressure, don't re-flag seconds already handled above.
        already = qc[col].to_numpy() != QC_OK if col in (WS, HUM, PRES) else np.zeros(n, dtype=bool)
        v = vals.to_numpy(dtype=np.float64).copy()
        v[already] = np.nan  # temporarily hide already-treated points from the bound check
        fixed, gflag = apply_glitch_fix(col, pd.Series(v, index=vals.index), low, high, glitch_max_run, glitch_report)
        take = ~already
        df.loc[take, col] = fixed[take]
        qc.loc[take & (gflag != QC_OK), col] = gflag[take & (gflag != QC_OK)]
    report["glitch"] = glitch_report

    log("P_Gaia flags (values unchanged) ...")
    pgaia_report = {}
    qc[PG] = flag_p_gaia(df.index, P_GAIA_ACTIVE_DAYS, pgaia_report)
    report["p_gaia"] = pgaia_report

    log("Recomputing Azimuth/Elevation ...")
    pos = compute_solar_position_bulk(df.index, chunk_rows=chunk_rows)
    df[AZ] = pos["azimuth_south"].to_numpy()
    df[EL] = pos["elevation"].to_numpy()
    report["azimuth_elevation"] = {
        "convention": "UTC timestamps, azimuth south-referenced (0=south, east negative), apparent elevation, not clipped at night",
        "n_rows": n, "elevation_min": float(df[EL].min()) if n else None,
        "elevation_max": float(df[EL].max()) if n else None,
    }

    qc_col_names = []
    for col in QC_COLS:
        qc_name = f"{col}_qc"
        assert_pipeline_codes(qc[col].to_numpy(), qc_name)   # code 6 is platform-owned: never emitted here
        df[qc_name] = qc[col].to_numpy().astype(np.int8)
        qc_col_names.append(qc_name)
    report["qc_columns_merged"] = qc_col_names
    report["qc_flag_value_counts"] = {
        col: {int(k): int(v) for k, v in qc[col].value_counts().sort_index().items()} for col in QC_COLS
    }
    return df, report


# ---------------------------------------------------------------------------------------------
# merging the reports of several slices
# ---------------------------------------------------------------------------------------------
def merge_reports(a, b):
    """Sum two slice reports (counts add; day lists union; min/max combine; settings must agree)."""
    if a is None:
        return b
    out = {}
    for k in a.keys() | b.keys():
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            out[k] = x if y is None else y
        elif k == "_pressure_days_with_real_value":
            out[k] = sorted(set(x) | set(y))
        elif k == "elevation_min":
            out[k] = min(x, y)
        elif k == "elevation_max":
            out[k] = max(x, y)
        elif isinstance(x, dict):
            out[k] = merge_reports(x, y)
        elif isinstance(x, bool) or isinstance(x, str) or isinstance(x, list):
            out[k] = x
        elif isinstance(x, (int, float)):
            out[k] = x + y
        else:
            out[k] = x
    return out


def finish_report(report):
    """Turn the mergeable pieces into their final form (day set -> count; flatline value counts -> top 10)."""
    r = dict(report)
    days = r.pop("_pressure_days_with_real_value", [])
    r["pressure_n_days_with_real_value_remaining"] = len(days)
    r["pressure_days_with_real_value"] = days
    fl = r.get("pressure_flatline_only_example_values")
    if fl:
        r["pressure_flatline_only_example_values"] = dict(sorted(fl.items(), key=lambda t: -t[1])[:10])
    return r


# ---------------------------------------------------------------------------------------------
# slicing
# ---------------------------------------------------------------------------------------------
def unsafe_row_mask(block):
    """Rows that any run-based rule may treat as part of a run or use as a run's anchor:
    joint-dropout rows and out-of-bound values (every column of BOUNDS). Built from the same masks
    the rules use, so it is a superset of every run the rules can form."""
    unsafe = dropout_bad_mask(block[WS].to_numpy(), block[HUM].to_numpy())
    for col, (low, high) in BOUNDS.items():
        unsafe |= glitch_bad_mask(block[col].to_numpy(dtype=np.float64), low, high)
    return unsafe


def safe_boundaries(block):
    """For a block of rows, a boolean array `ok` of length len(block)-1: ok[j] is True when a cut between
    row j and row j+1 cannot split any run-based rule's run (or its anchors):
      * neither neighbour is a dropout row or an out-of-bound value, and
      * the two pressure samples are not the same finite non-multiple-of-1000 value (a flatline
        candidate; identical multiples of 1000 are sentinels by rule (a) whatever the run length)."""
    unsafe = unsafe_row_mask(block)
    p = block[PRES].to_numpy(dtype=np.float64)
    same_p = (p[1:] == p[:-1]) & np.isfinite(p[1:]) & np.isfinite(p[:-1]) & ~pressure_multiples_of_1000(p[1:])
    return ~unsafe[1:] & ~unsafe[:-1] & ~same_p


def find_safe_cut(path, key, target, n_rows, window=200_000, max_slice_rows=None, start=0):
    """First row position c >= target (c < n_rows) such that a cut between rows c-1 and c is safe;
    n_rows if none exists before the end of the file. The cut search reads only the six columns the
    run-based rules look at, window by window."""
    cols = [WS, HUM, TEMP, GHI, POA, PRES]
    c = max(int(target), 1)
    while c < n_rows:
        lo, hi = c - 1, min(c + window, n_rows)
        block = read_rows(path, key, lo, hi, columns=cols)
        ok = safe_boundaries(block)             # ok[j]: cut between lo+j and lo+j+1, i.e. at position lo+j+1
        hits = np.flatnonzero(ok)
        if len(hits):
            return lo + int(hits[0]) + 1
        c = hi
        if max_slice_rows is not None and c - start > max_slice_rows:
            raise RuntimeError(
                f"No safe cut within {max_slice_rows:,} rows after row {start:,}: a single run-based rule "
                f"region is longer than the slice limit. Raise --max-slice-rows (memory!) or inspect the data.")
    return n_rows


def plan_slices(path, key, slice_rows, max_slice_rows=None, window=200_000):
    """[(start, stop), ...] covering the whole file; every internal cut passes find_safe_cut."""
    n = h5_info(path, key)["nrows"]
    cuts, start = [0], 0
    while start < n:
        c = find_safe_cut(path, key, start + slice_rows, n, window=window, max_slice_rows=max_slice_rows, start=start)
        cuts.append(c)
        start = c
    return list(zip(cuts[:-1], cuts[1:]))


def clean_file(in_path, out_path, key="DATA", slice_rows=DEFAULT_SLICE_DAYS * 86400, max_slice_rows=None,
               dropout_max_run=5, glitch_max_run=3, flatline_min_run=300, chunk_rows=2_000_000,
               overwrite=False, verbose=True):
    """Clean a sorted table/fixed-format 1 s file slice by slice. Returns the final report."""
    info = h5_info(in_path, key)
    n = info["nrows"]
    idx = read_index(in_path, key)
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{in_path} is not in chronological order; build the sorted `_original` file first "
                         "(build_release.py --stages original) or use --in-memory.")
    del idx
    if max_slice_rows is None:
        max_slice_rows = max(8 * slice_rows, 1)
    slices = plan_slices(in_path, key, slice_rows, max_slice_rows)
    if verbose:
        print(f"{in_path}: {n:,} rows ({info['format']} format) -> {len(slices)} slices", flush=True)
    report = None
    with TableWriter(out_path, key, overwrite=overwrite) as w:
        for i, (a, b) in enumerate(slices, 1):
            block = read_rows(in_path, key, a, b)
            cleaned, rep = clean_block(block, dropout_max_run, glitch_max_run, flatline_min_run, chunk_rows, verbose=False)
            rep["n_slices"] = 1
            report = merge_reports(report, rep)
            w.append(cleaned)
            if verbose:
                print(f"  slice {i}/{len(slices)}: rows {a:,}..{b:,} ({b - a:,}) done", flush=True)
            del block, cleaned
    report = finish_report(report)
    report["slice_boundaries_rows"] = [a for a, _ in slices[1:]]
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--key", default="DATA")
    ap.add_argument("--out-prefix", default="SOLETE_Pombo_1sec_cleaned")
    ap.add_argument("--dropout-max-run", type=int, default=5)
    ap.add_argument("--glitch-max-run", type=int, default=3)
    ap.add_argument("--pressure-flatline-min-run", type=int, default=300,
                     help="consecutive identical Pressure samples at/above this length are treated as a stuck sensor")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000, help="pvlib chunk")
    ap.add_argument("--slice-days", type=float, default=DEFAULT_SLICE_DAYS, help="approximate slice length in days")
    ap.add_argument("--max-slice-rows", type=int, default=None)
    ap.add_argument("--in-memory", action="store_true",
                    help="old whole-file path: loads and sorts everything first (needs ~7 GB RAM; accepts the shuffled raw file)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.path = str(resolve_input(args.path))                 # bare names are looked up in data/hdf5/
    args.out_prefix = resolve_output_prefix(args.out_prefix)  # bare prefixes are written to data/hdf5/

    out_data = args.out_prefix + ".h5"
    if os.path.abspath(out_data) == os.path.abspath(args.path):
        sys.exit(f"Refusing to write over the input file ({out_data}).")
    if os.path.exists(out_data) and not args.overwrite:
        sys.exit(f"{out_data} exists; pass --overwrite to replace it.")

    common = dict(dropout_max_run=args.dropout_max_run, glitch_max_run=args.glitch_max_run,
                  flatline_min_run=args.pressure_flatline_min_run, chunk_rows=args.chunk_rows)
    if args.in_memory:
        print(f"Loading {args.path} (whole file) ...")
        df = pd.read_hdf(args.path, key=args.key)
        print("Sorting chronologically (one column at a time) ...")
        df = reorder_chronologically(df)
        cleaned, report = clean_block(df, **common)
        del df
        report = finish_report(report)
        with TableWriter(out_data, args.key, overwrite=args.overwrite) as w:
            for i in range(0, len(cleaned), args.chunk_rows):
                w.append(cleaned.iloc[i:i + args.chunk_rows])
        n = len(cleaned)
    else:
        slice_rows = max(int(args.slice_days * 86400), 1)
        report = clean_file(args.path, out_data, args.key, slice_rows=slice_rows,
                            max_slice_rows=args.max_slice_rows, overwrite=args.overwrite, **common)
        n = report["n_rows"]
    report.update(input=args.path, output=out_data, dropout_max_run=args.dropout_max_run,
                  glitch_max_run=args.glitch_max_run, mode="in-memory" if args.in_memory else "sliced")

    print("Verifying ...")
    qc_col_names = report["qc_columns_merged"]
    check_data = read_rows(out_data, args.key, 0, 5)
    idx_full = read_rows(out_data, args.key, columns=[WD])
    report["verification"] = {
        "qc_columns_present": all(c in check_data.columns for c in qc_col_names),
        "qc_columns_are_int8": all(str(check_data[c].dtype) == "int8" for c in qc_col_names),
        "n_rows_written": int(len(idx_full)),
        "n_rows_expected": n,
        "index_monotonic_unique": bool(idx_full.index.is_monotonic_increasing and idx_full.index.is_unique),
        "wind_dir_still_out_of_range": int(((idx_full[WD] < 0) | (idx_full[WD] >= 360)).sum()),
    }
    print_report("clean_solete_1sec_summary", report)


if __name__ == "__main__":
    main()
