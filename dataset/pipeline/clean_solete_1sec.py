"""
clean_solete_1sec.py -- build the cleaned, chronologically-sorted 1-second
file with per-column QC flags merged in. See docs/CLEANING_DECISIONS.md for
the reasoning behind every choice below -- several are provisional and marked
as such.

WHAT THIS DOES, IN ORDER:
  1. Sorts all rows chronologically (the input stores 457 daily blocks in
     shuffled order; nothing is added or removed -- see followup_A).
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
  7. Azimuth[deg] / Elevation[deg]: the published values are DROPPED and
     replaced with a fresh pvlib (NREL SPA) computation for every row, in
     UTC (see solar_position.py's timezone finding) with the south-
     referenced azimuth convention (0=south, east negative) to match the
     original column's semantics. Elevation is NOT clipped at night -- it
     goes negative below the horizon (physically correct), unlike the
     published file which showed 0 then. Every row is flagged QC_RECOMPUTED.

Note on step 5 vs step 4: the generic glitch pass NaNs out long-untreated
runs (QC_GLITCH_LONG_UNTREATED_NAN) because those values are physically
impossible by construction -- leaving e.g. a -40.1degC reading in the data
would be worse than admitting it's missing. The WS/HUM dropout pass does
NOT NaN its long-untreated runs (QC_DROPOUT_LONG_UNTREATED) -- 0 m/s wind
and 0% humidity are each individually plausible values, it's only the
*combination* that's suspicious, so the recorded value is kept and flagged
for manual review rather than destroyed. See qc_flags.fix_short_runs.

OUTPUT:
  <out-prefix>.h5   cleaned data, same columns/dtypes as the input, sorted
                     chronologically, HDF5 table format, PLUS one int8
                     `<column>_qc` column per treated source column (see
                     qc_flags.py for the QC_* codes) merged into the same
                     DataFrame -- not a separate companion file, so the flags
                     survive a plain `pd.read_hdf()` with nothing extra to
                     keep in sync.

MEMORY: the whole 1-second file is loaded once (~3.5 GB). Columns are
reordered into chronological order ONE AT A TIME (not via a single
df.sort_index() on all 11 columns at once, which is what caused the earlier
MemoryError) -- this keeps peak extra memory to roughly one column's size
instead of the whole frame's. The pvlib call is chunked (see
solar_position.compute_solar_position_bulk) and can take several minutes for
39M rows.

USAGE (best from a plain terminal rather than an IDE, to leave the most RAM
free):
    python dataset/pipeline/clean_solete_1sec.py SOLETE_Pombo_1sec.h5 --key DATA \\
        --out-prefix SOLETE_Pombo_1sec_cleaned_v4

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import resolve_input, resolve_output_prefix  # noqa: E402
from qc_flags import (
    QC_ACTIVE_DAY,
    QC_DROPOUT_LONG_UNTREATED,
    QC_DROPOUT_SHORT_FIXED,
    QC_GLITCH_LONG_UNTREATED_NAN,
    QC_GLITCH_SHORT_FIXED,
    QC_OK,
    QC_PLACEHOLDER,
    QC_RECOMPUTED,
    QC_UNVERIFIED_PROVENANCE,
    detect_flatline_mask,
    fix_short_runs,
)
from solar_position import compute_solar_position_bulk
from solete_report import print_report

WD, WS, HUM, PRES, TEMP, GHI, POA = (
    "WIND_DIR[deg]", "WIND_SPEED[m1s]", "HUMIDITY[%]", "Pressure[mbar]",
    "TEMPERATURE[degC]", "GHI[kW1m2]", "POA Irr[kW1m2]",
)
AZ, EL, PG = "Azimuth[deg]", "Elevation[deg]", "P_Gaia[kW]"

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


def reorder_chronologically(df):
    """Sort `df` by its index, one column at a time, to avoid the memory
    spike of sorting the whole block manager at once (see MEMORY note)."""
    order = np.argsort(df.index.values, kind="stable")
    new_index = pd.DatetimeIndex(df.index.values[order])
    for c in df.columns:
        df[c] = df[c].to_numpy()[order]
    df.index = new_index
    return df


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
    multiples_of_1000 = np.isclose(np.mod(v, 1000.0), 0.0, atol=1e-6)
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
    report["pressure_n_days_with_real_value_remaining"] = int(
        pres.index[~mask].normalize().nunique()) if (~mask).any() else 0
    return out, flag


def apply_dropout_fix(ws, hum, max_run, report):
    bad = (ws.to_numpy() == 0) & (hum.to_numpy() < HUM_DROPOUT_TOL)
    ws_out, ws_flag = fix_short_runs(ws.to_numpy(), bad, max_run, QC_DROPOUT_SHORT_FIXED, QC_DROPOUT_LONG_UNTREATED)
    hum_out, hum_flag = fix_short_runs(hum.to_numpy(), bad, max_run, QC_DROPOUT_SHORT_FIXED, QC_DROPOUT_LONG_UNTREATED)
    report["dropout_n_seconds_flagged"] = int(bad.sum())
    report["dropout_n_fixed"] = int((ws_flag == QC_DROPOUT_SHORT_FIXED).sum())
    report["dropout_n_untreated"] = int((ws_flag == QC_DROPOUT_LONG_UNTREATED).sum())
    return ws_out, ws_flag, hum_out, hum_flag


def apply_glitch_fix(col, values, low, high, max_run, report):
    v = values.to_numpy(dtype=np.float64) if hasattr(values, "to_numpy") else np.asarray(values, dtype=np.float64)
    bad = np.zeros(len(v), dtype=bool)
    finite = np.isfinite(v)
    if low is not None:
        bad |= finite & (v < low)
    if high is not None:
        bad |= finite & (v > high)
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


def clean_dataframe(
    df,
    *,
    dropout_max_run=5,
    glitch_max_run=3,
    pressure_flatline_min_run=300,
    solar_chunk_rows=2_000_000,
    solar_verbose=True,
):
    """Apply all cleaning rules to one chronological frame.

    Callers slicing a larger data set must include enough rows on both sides
    to cover the longest stateful rule, then discard that overlap.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("clean_dataframe requires a chronological index")
    df = df.copy()
    report = {}
    n = len(df)
    qc_columns = [WD, PRES, WS, HUM, TEMP, GHI, POA, PG, AZ, EL]
    qc = pd.DataFrame(QC_OK, index=df.index, columns=qc_columns, dtype=np.int64)

    wd = df[WD].to_numpy()
    wrapped = np.mod(wd, 360.0)
    qc[WD] = np.where(wrapped != wd, 1, QC_OK)
    df[WD] = wrapped
    report["wind_dir_n_wrapped"] = int((wrapped != wd).sum())

    df[PRES], qc[PRES] = apply_pressure_sentinels(
        df[PRES], report, pressure_flatline_min_run
    )
    dropout_report = {}
    df[WS], qc[WS], df[HUM], qc[HUM] = apply_dropout_fix(
        df[WS], df[HUM], dropout_max_run, dropout_report
    )
    report["dropout"] = dropout_report

    glitch_report = {}
    for col, (low, high) in BOUNDS.items():
        values = df[col]
        already = (
            qc[col].to_numpy() != QC_OK
            if col in (WS, HUM, PRES)
            else np.zeros(n, dtype=bool)
        )
        hidden = values.to_numpy(dtype=np.float64).copy()
        hidden[already] = np.nan
        fixed, glitch_flag = apply_glitch_fix(
            col,
            pd.Series(hidden, index=values.index),
            low,
            high,
            glitch_max_run,
            glitch_report,
        )
        take = ~already
        df.loc[take, col] = fixed[take]
        flagged = take & (glitch_flag != QC_OK)
        qc.loc[flagged, col] = glitch_flag[flagged]
    report["glitch"] = glitch_report

    pgaia_report = {}
    qc[PG] = flag_p_gaia(df.index, P_GAIA_ACTIVE_DAYS, pgaia_report)
    report["p_gaia"] = pgaia_report

    position = compute_solar_position_bulk(
        df.index, chunk_rows=solar_chunk_rows, verbose=solar_verbose
    )
    df[AZ] = position["azimuth_south"].to_numpy()
    df[EL] = position["elevation"].to_numpy()
    qc[AZ] = QC_RECOMPUTED
    qc[EL] = QC_RECOMPUTED

    for col in qc_columns:
        df[f"{col}_qc"] = qc[col].to_numpy().astype(np.int8)
    report["qc_columns_merged"] = [f"{col}_qc" for col in qc_columns]
    return df, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--key", default="DATA")
    ap.add_argument("--out-prefix", default="SOLETE_Pombo_1sec_cleaned_v4")
    ap.add_argument("--dropout-max-run", type=int, default=5)
    ap.add_argument("--glitch-max-run", type=int, default=3)
    ap.add_argument("--pressure-flatline-min-run", type=int, default=300,
                     help="consecutive identical Pressure samples at/above this length are treated as a stuck sensor")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.path = str(resolve_input(args.path))                 # bare names are looked up in data/hdf5/
    args.out_prefix = resolve_output_prefix(args.out_prefix)  # bare prefixes are written to data/hdf5/

    out_data = args.out_prefix + ".h5"
    if os.path.abspath(out_data) == os.path.abspath(args.path):
        sys.exit(f"Refusing to write over the input file ({out_data}).")
    if os.path.exists(out_data) and not args.overwrite:
        sys.exit(f"{out_data} exists; pass --overwrite to replace it.")

    report = {"input": args.path, "dropout_max_run": args.dropout_max_run, "glitch_max_run": args.glitch_max_run}

    print(f"Loading {args.path} ...")
    df = pd.read_hdf(args.path, key=args.key)
    n = len(df)
    report["n_rows"] = n

    print("Sorting chronologically (one column at a time) ...")
    df = reorder_chronologically(df)
    report["sorted_is_monotonic"] = bool(df.index.is_monotonic_increasing)

    print("Applying cleaning and QC rules ...")
    df, cleaning_report = clean_dataframe(
        df,
        dropout_max_run=args.dropout_max_run,
        glitch_max_run=args.glitch_max_run,
        pressure_flatline_min_run=args.pressure_flatline_min_run,
        solar_chunk_rows=args.chunk_rows,
    )
    report.update(cleaning_report)
    qc_col_names = cleaning_report["qc_columns_merged"]

    if os.path.exists(out_data):
        os.remove(out_data)
    print(f"Writing {out_data} in chunks of {args.chunk_rows:,} rows ...")
    with pd.HDFStore(out_data, mode="w", complevel=1, complib="zlib") as store:
        for i in range(0, n, args.chunk_rows):
            sl = slice(i, i + args.chunk_rows)
            store.append(args.key, df.iloc[sl], format="table", index=False)
            print(f"  {min(i + args.chunk_rows, n):>12,} / {n:,}")

    print("Verifying ...")
    check_data = pd.read_hdf(out_data, key=args.key, stop=5)
    idx_full = pd.read_hdf(out_data, key=args.key, columns=[WD])
    report["verification"] = {
        "columns_match": list(check_data.columns) == list(df.columns),
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
