# -*- coding: utf-8 -*-
"""
make_original.py -- build SOLETE_Pombo_1sec_original_v4.h5 from the raw v3 1-second file.

What the `_original` file is: the raw 1 s data of v3, nothing cleaned, nothing derived, in
CHRONOLOGICAL order, with only the nine measured columns the pipeline and the platform need. Differences
to the v3 file, and only these:
  * rows are sorted by timestamp (the v3 file stores 457 daily blocks in shuffled order; a
    duplicated or missing timestamp is reported, not repaired);
  * the Azimuth[deg] and Elevation[deg] columns are dropped (the build recomputes them with pvlib).
Every measured value is copied bit for bit (verify_original re-reads both files and compares them).

HOW: the raw file (pandas `fixed` or `table` format) is never loaded whole. Only its time index is
(about 316 MB for 39.5 M rows); the stable argsort of that index gives the sorted order, which is cut
into maximal runs of consecutive source rows (for the real file: the 457 daily blocks); the runs are
read with solete.h5io.read_rows and appended to the output table in the sorted order.

USAGE
    python dataset/pipeline/make_original.py SOLETE_Pombo_1sec.h5
        [--out SOLETE_Pombo_1sec_original_v4.h5] [--chunk-rows 2000000] [--overwrite] [--no-verify]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from solete.h5io import TableWriter, h5_info, read_index, read_rows  # noqa: E402
from solete.paths import find_data_file, release_path, resolve_input  # noqa: E402

AZ, EL = "Azimuth[deg]", "Elevation[deg]"
MEASURED = ["TEMPERATURE[degC]", "HUMIDITY[%]", "WIND_SPEED[m1s]", "WIND_DIR[deg]", "GHI[kW1m2]",
            "POA Irr[kW1m2]", "P_Gaia[kW]", "P_Solar[kW]", "Pressure[mbar]"]


def sorted_runs(index: pd.DatetimeIndex):
    """(order, runs): `order` = stable argsort of the index; `runs` = [(source_start, length), ...], the
    maximal stretches of consecutive source rows in sorted order. Concatenating the runs in this order
    gives the chronologically sorted file."""
    order = np.argsort(index.values, kind="stable")
    breaks = np.flatnonzero(np.diff(order) != 1) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(order)]))
    return order, [(int(order[s]), int(e - s)) for s, e in zip(starts, ends)]


def build_original(raw, out, key="DATA", chunk_rows=2_000_000, overwrite=False, verbose=True):
    """Write the sorted nine-column table `out` from the raw file `raw`; return a report dict."""
    info = h5_info(raw, key)
    cols = info["columns"]
    missing = [c for c in MEASURED if c not in cols]
    if missing:
        raise ValueError(f"{raw}: expected measured columns are missing: {missing}")
    keep = [c for c in cols if c not in (AZ, EL)]
    extra = [c for c in keep if c not in MEASURED]
    if extra:
        raise ValueError(f"{raw}: unexpected extra columns {extra}; refusing to guess what to keep.")
    idx = read_index(raw, key)
    n = len(idx)
    order, runs = sorted_runs(idx)
    sorted_idx = idx[order]
    steps = np.diff(sorted_idx.values).astype("timedelta64[s]").astype(np.int64)
    report = {
        "raw": str(raw), "raw_format": info["format"], "n_rows": n,
        "n_source_runs": len(runs),
        "run_lengths": {int(k): int(v) for k, v in zip(*np.unique([r[1] for r in runs], return_counts=True))},
        "was_already_sorted": bool(len(runs) == 1),
        "n_duplicate_timestamps": int((steps == 0).sum()),
        "n_gaps_over_1s": int((steps > 1).sum()),
        "largest_gap_seconds": int(steps.max()) if len(steps) else 0,
        "first_timestamp": str(sorted_idx[0]), "last_timestamp": str(sorted_idx[-1]),
        "dropped_columns": [c for c in cols if c in (AZ, EL)],
        "columns_kept": keep,
    }
    if report["n_duplicate_timestamps"]:
        raise ValueError(f"{raw}: {report['n_duplicate_timestamps']} duplicated timestamps; refusing to build "
                         "the sorted file silently (inspect the raw data first).")
    del idx, order, sorted_idx, steps
    pending, pending_rows, written = [], 0, 0
    with TableWriter(out, key, overwrite=overwrite) as w:
        def flush():
            nonlocal pending, pending_rows, written
            if pending:
                block = pd.concat(pending)
                w.append(block)
                written += len(block)
                if verbose:
                    print(f"  {written:>12,} / {n:,} rows written", flush=True)
                pending, pending_rows = [], 0
        for start, length in runs:
            pos = 0
            while pos < length:                       # a very long run is read in pieces
                take = min(length - pos, chunk_rows)
                pending.append(read_rows(raw, key, start + pos, start + pos + take, columns=keep)[keep])
                pending_rows += take
                pos += take
                if pending_rows >= chunk_rows:
                    flush()
        flush()
    report["n_rows_written"] = written
    return report


def verify_original(raw, out, key="DATA", verbose=True):
    """Re-read both files and check the `_original` file is exactly the raw one, sorted: same row set,
    bit-identical values, strictly increasing index, nine columns, no Azimuth/Elevation. Returns a dict;
    raises AssertionError on any mismatch."""
    info_out = h5_info(out, key)
    idx_raw = read_index(raw, key)
    idx_out = read_index(out, key)
    assert info_out["columns"] and not ({AZ, EL} & set(info_out["columns"])), "Azimuth/Elevation still present"
    assert len(idx_out) == len(idx_raw), f"row counts differ: {len(idx_out)} vs {len(idx_raw)}"
    assert idx_out.is_monotonic_increasing and idx_out.is_unique, "output index is not strictly increasing"
    order, runs = sorted_runs(idx_raw)
    assert idx_out.equals(idx_raw[order]), "output index != sorted raw index"
    keep = info_out["columns"]
    pos, n_compared = 0, 0
    for start, length in runs:
        a = read_rows(raw, key, start, start + length, columns=keep)[keep]
        b = read_rows(out, key, pos, pos + length)
        assert a.index.equals(b.index), f"timestamps differ in the run at source row {start}"
        for c in keep:
            av, bv = a[c].to_numpy(), b[c].to_numpy()
            assert av.dtype == bv.dtype, f"dtype of {c} changed: {av.dtype} -> {bv.dtype}"
            same = (av == bv) | (np.isnan(av) & np.isnan(bv)) if av.dtype.kind == "f" else (av == bv)
            assert same.all(), f"{c}: values differ in the run at source row {start}"
        pos += length
        n_compared += length
    if verbose:
        print(f"verify_original: {n_compared:,} rows compared, identical", flush=True)
    return {"n_rows_compared": n_compared, "identical": True, "columns": keep, "format": info_out["format"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="?", default=None, help="raw v3 1 s file (default: the v3 file in the data folder)")
    ap.add_argument("--key", default="DATA")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    raw = resolve_input(a.raw) if a.raw else find_data_file("1sec", "v3")
    out = Path(a.out) if a.out else release_path("1sec", original=True)
    rep = build_original(raw, out, a.key, a.chunk_rows, a.overwrite)
    if not a.no_verify:
        rep["verification"] = verify_original(raw, out, a.key)
    import json
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
