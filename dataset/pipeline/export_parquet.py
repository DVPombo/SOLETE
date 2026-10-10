"""
export_parquet.py -- convert a SOLETE HDF5 file (the cleaned 1-second file, or
any resampled file produced by resample_solete.py) into a Parquet file that is
ready to use without knowing anything about pandas/PyTables.

The HDF5 files are the *transparent* artefacts of the pipeline; the Parquet
files are the *convenient* ones. Both contain exactly the same numbers.

Differences between the two, by design:
  * The time index becomes a regular column called `timestamp`, FIRST in the file, stored as
    timestamp[ns, tz=UTC]. The unit is pinned in code (TIMESTAMP_UNIT): pandas 3 would otherwise
    write microseconds for some inputs. (The HDF5 files store a tz-naive index whose values are
    UTC -- see docs/DATA_DICTIONARY.md. Parquet makes that explicit.)
    Use `--keep-naive` to write the timestamps exactly as stored (timestamp[ns], no zone) instead.
  * Column names, dtypes, values, NaNs and row order are unchanged.
  * File-level metadata is embedded under the key b"solete" (JSON) so the file documents itself:
    dataset, version v4, resolution, UTC and the timestamp unit/label convention, the site
    latitude/longitude/altitude and the Azimuth/Elevation convention (0 = south, east negative), the
    pvlib version, build date, the QC code table, and the provenance of every column (see
    release_meta.py).
  * Row groups of 1,000,000 rows with column statistics; zstd compression. `--trial` measures zstd
    levels 3, 9, 15 with and without byte_stream_split on the first million rows and prints the table;
    `--auto` applies the smallest result that is not more than 4x slower than zstd-3 and at least 2 % smaller than it (else zstd-3).

Usage:
    python dataset/pipeline/export_parquet.py SOLETE_Pombo_1sec_v4.h5          # -> data/parquet/SOLETE_Pombo_1sec_v4.parquet
    python dataset/pipeline/export_parquet.py SOLETE_Pombo_1sec_v4.h5 out/x.parquet
    python dataset/pipeline/export_parquet.py SOLETE_Pombo_60min_v4.h5 --compression zstd --level 9
    python dataset/pipeline/export_parquet.py SOLETE_Pombo_1sec_v4.h5 --trial   # measure, write nothing

Or from Python:
    from export_parquet import h5_to_parquet
    h5_to_parquet("SOLETE_Pombo_1sec_v4.h5", "SOLETE_Pombo_1sec_v4.parquet")

Reading the result:
    import pandas as pd
    df = pd.read_parquet("SOLETE_Pombo_1sec_v4.parquet").set_index("timestamp")
    # or, for a time slice without loading everything (DuckDB):
    #   duckdb.sql("SELECT * FROM 'SOLETE_Pombo_1sec_v4.parquet' WHERE timestamp >= '2019-01-16' AND timestamp < '2019-01-17'")

Memory: files stored in HDF5 `table` format (the 1-second files) are streamed in chunks of
--chunk-rows rows, so the ~39M-row file never has to fit in RAM. Files stored in `fixed` format
(the resampled files, which are small) are read in the same chunks through solete.h5io.

Requires: pip install pandas numpy tables pyarrow
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import HDF5_DIR, PARQUET_DIR, resolve_input  # noqa: E402
from solete.h5io import h5_info, read_rows  # noqa: E402
from release_meta import encode, file_metadata  # noqa: E402

DATASET_NAME = "SOLETE"
TIMESTAMP_COL = "timestamp"
TIMESTAMP_UNIT = "ns"            # pinned: see the module docstring
ROW_GROUP_SIZE = 1_000_000


def _prepare(df: pd.DataFrame, keep_naive: bool) -> pd.DataFrame:
    """Index -> `timestamp` column (unit pinned to TIMESTAMP_UNIT); localize naive timestamps to UTC."""
    idx = pd.DatetimeIndex(df.index).as_unit(TIMESTAMP_UNIT)
    if not keep_naive:
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = df.reset_index(drop=True)
    out.insert(0, TIMESTAMP_COL, idx)
    return out


def _timestamp_type(keep_naive):
    return pa.timestamp(TIMESTAMP_UNIT) if keep_naive else pa.timestamp(TIMESTAMP_UNIT, tz="UTC")


def _pinned_schema(table, keep_naive, metadata):
    fields = [pa.field(TIMESTAMP_COL, _timestamp_type(keep_naive), nullable=False)] + list(table.schema)[1:]
    return pa.schema(fields).with_metadata(metadata)


def _writer_options(columns_dtypes, compression, compression_level, byte_stream_split):
    """kwargs for ParquetWriter. byte_stream_split applies to the float columns and needs dictionary
    encoding off for them (pyarrow: dictionary wins otherwise); the other columns keep the dictionary."""
    opts = dict(compression=compression, compression_level=compression_level, write_statistics=True)
    if byte_stream_split:
        floats = [c for c, dt in columns_dtypes.items() if dt.kind == "f"]
        opts["use_byte_stream_split"] = floats
        opts["use_dictionary"] = [c for c in columns_dtypes if c not in floats]
    return opts


def compression_trial(h5_path, key="DATA", n_rows=1_000_000, levels=(3, 9, 15), keep_naive=False):
    """Write the first `n_rows` rows with zstd at each level, with and without byte_stream_split, into memory.
    Returns a list of dicts {compression, level, byte_stream_split, size_bytes, write_seconds}."""
    sample = read_rows(h5_path, key, 0, n_rows)
    table = pa.Table.from_pandas(_prepare(sample, keep_naive), preserve_index=False)
    dtypes = {c: sample[c].dtype for c in sample.columns}
    schema = _pinned_schema(table, keep_naive, {})
    rows = []
    for bss in (False, True):
        for level in levels:
            buf = io.BytesIO()
            t0 = time.perf_counter()
            with pq.ParquetWriter(buf, schema, **_writer_options(dtypes, "zstd", level, bss)) as w:
                w.write_table(table.cast(schema), row_group_size=ROW_GROUP_SIZE)
            rows.append({"compression": "zstd", "level": level, "byte_stream_split": bss,
                         "size_bytes": buf.getbuffer().nbytes, "write_seconds": round(time.perf_counter() - t0, 3),
                         "rows": len(sample)})
    return rows


def choose_settings(trial, max_slowdown=4.0, min_gain=0.02):
    """The smallest trial result whose write time is within `max_slowdown` x the zstd-3 / no-BSS time, but only if it
    is at least `min_gain` (2 %) smaller than zstd-3; otherwise zstd-3 (a slower setting must earn its time)."""
    base = next(r for r in trial if r["level"] == 3 and not r["byte_stream_split"])
    ok = [r for r in trial if r["write_seconds"] <= max_slowdown * max(base["write_seconds"], 1e-3)]
    best = min(ok, key=lambda r: (r["size_bytes"], r["level"]))
    if best["size_bytes"] > (1 - min_gain) * base["size_bytes"]:
        best = base
    return {"compression": best["compression"], "compression_level": best["level"],
            "byte_stream_split": best["byte_stream_split"]}


def print_trial(trial):
    base = next(r["size_bytes"] for r in trial if r["level"] == 3 and not r["byte_stream_split"])
    print(f"compression trial on the first {trial[0]['rows']:,} rows")
    print(f"  {'zstd level':>10} {'byte_stream_split':>18} {'size MB':>9} {'vs zstd-3':>10} {'seconds':>8}")
    for r in trial:
        print(f"  {r['level']:>10} {str(r['byte_stream_split']):>18} {r['size_bytes'] / 1e6:>9.2f} "
              f"{r['size_bytes'] / base:>10.3f} {r['write_seconds']:>8.2f}")


def h5_to_parquet(
    h5_path,
    parquet_path=None,
    key="DATA",
    chunk_rows=2_000_000,
    compression="zstd",
    compression_level=3,
    keep_naive=False,
    overwrite=False,
    verbose=True,
    byte_stream_split=False,
    extra_metadata=None,
):
    """Convert `h5_path` to Parquet. Returns the output path and the row count."""
    h5_path = str(h5_path)
    if parquet_path:
        parquet_path = str(parquet_path)
    elif Path(h5_path).resolve().parent == HDF5_DIR:
        # figshare layout: data/hdf5/X.h5 -> data/parquet/X.parquet
        parquet_path = str(PARQUET_DIR / (Path(h5_path).stem + ".parquet"))
    else:
        parquet_path = str(Path(h5_path).with_suffix(".parquet"))
    if os.path.exists(parquet_path) and not overwrite:
        raise FileExistsError(f"{parquet_path} exists; pass overwrite=True / --overwrite to replace it.")
    if os.path.abspath(parquet_path) == os.path.abspath(h5_path):
        raise ValueError("Output path equals input path.")
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)

    info = h5_info(h5_path, key)
    n_rows, fmt = info["nrows"], info["format"]

    def log(msg):
        if verbose:
            print(msg, flush=True)

    log(f"{h5_path}: key={key!r}, format={fmt}, rows={n_rows:,}")
    writer = None
    written = 0
    last_ts = None
    tmp_path = parquet_path + ".tmp"
    try:
        for s in range(0, n_rows, chunk_rows):
            chunk = read_rows(h5_path, key, s, min(s + chunk_rows, n_rows))
            if not chunk.index.is_monotonic_increasing:
                raise ValueError("Index is not sorted chronologically; build the sorted _original file / run clean_solete_1sec.py first.")
            if last_ts is not None and chunk.index[0] <= last_ts:
                raise ValueError("Chunks overlap or are out of order; the file is not chronologically sorted.")
            last_ts = chunk.index[-1]
            table = pa.Table.from_pandas(_prepare(chunk, keep_naive), preserve_index=False)
            if writer is None:
                meta = file_metadata(h5_path, n_rows, chunk.columns, keep_naive,
                                     extra={"compression": f"{compression} level {compression_level}",
                                            "byte_stream_split": bool(byte_stream_split),
                                            "row_group_size": ROW_GROUP_SIZE, **(extra_metadata or {})})
                schema = _pinned_schema(table, keep_naive, encode(meta))
                dtypes = {c: chunk[c].dtype for c in chunk.columns}
                writer = pq.ParquetWriter(tmp_path, schema,
                                          **_writer_options(dtypes, compression, compression_level, byte_stream_split))
            writer.write_table(table.cast(writer.schema), row_group_size=ROW_GROUP_SIZE)
            written += len(chunk)
            log(f"  {written:>12,} / {n_rows:,}")
        if writer is None:
            raise ValueError("The HDF5 key contains no rows.")
        writer.close()
        writer = None
        os.replace(tmp_path, parquet_path)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    pf = pq.ParquetFile(parquet_path)
    if pf.metadata.num_rows != n_rows:
        raise RuntimeError(f"Row count mismatch: parquet {pf.metadata.num_rows} vs h5 {n_rows}")
    size_mb = os.path.getsize(parquet_path) / 1e6
    log(f"Wrote {parquet_path}: {n_rows:,} rows, {len(pf.schema_arrow.names)} columns, {size_mb:,.1f} MB")
    return parquet_path, n_rows


def verify_roundtrip(h5_path, parquet_path, key="DATA", keep_naive=False, chunk_rows=1_000_000):
    """Read the Parquet file back and compare it with the HDF5 file: shape, column names and order, dtypes,
    NaN positions, exact value equality and the timestamps. Raises AssertionError on any difference;
    returns a small dict for the build summary."""
    info = h5_info(h5_path, key)
    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    assert names == [TIMESTAMP_COL] + list(info["columns"]), "column names/order differ"
    assert pf.metadata.num_rows == info["nrows"], "row counts differ"
    ts_type = pf.schema_arrow.field(TIMESTAMP_COL).type
    assert ts_type == _timestamp_type(keep_naive), f"timestamp type is {ts_type}, expected {_timestamp_type(keep_naive)}"
    assert b"solete" in (pf.schema_arrow.metadata or {}), "no solete metadata"
    pos = 0
    for batch in pf.iter_batches(batch_size=chunk_rows):
        pdf = batch.to_pandas()
        h5 = read_rows(h5_path, key, pos, pos + len(pdf))
        ts = pd.DatetimeIndex(pdf[TIMESTAMP_COL])
        ts = ts if keep_naive else ts.tz_convert("UTC").tz_localize(None)
        assert ts.equals(h5.index), f"timestamps differ at row {pos}"
        for c in h5.columns:
            a, b = pdf[c].to_numpy(), h5[c].to_numpy()
            assert a.dtype == b.dtype, f"{c}: dtype {a.dtype} (parquet) vs {b.dtype} (h5)"
            if a.dtype.kind == "f":
                assert (np.isnan(a) == np.isnan(b)).all(), f"{c}: NaN positions differ near row {pos}"
                assert ((a == b) | np.isnan(a)).all(), f"{c}: values differ near row {pos}"
            else:
                assert (a == b).all(), f"{c}: values differ near row {pos}"
        pos += len(pdf)
    assert pos == info["nrows"]
    return {"rows": pos, "columns": len(names), "timestamp_type": str(ts_type), "identical": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("h5_path")
    ap.add_argument("parquet_path", nargs="?", default=None, help="default: same name with .parquet")
    ap.add_argument("--key", default="DATA")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"])
    ap.add_argument("--level", type=int, default=3, help="compression level (zstd/gzip only)")
    ap.add_argument("--byte-stream-split", action="store_true", help="byte_stream_split encoding for the float columns")
    ap.add_argument("--trial", action="store_true", help="measure zstd 3/9/15 +- byte_stream_split on 1 M rows, write nothing")
    ap.add_argument("--auto", action="store_true", help="run the trial and use its best setting")
    ap.add_argument("--keep-naive", action="store_true", help="write timestamps tz-naive, exactly as stored in the h5")
    ap.add_argument("--no-verify", action="store_true", help="skip the round-trip check against the h5")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.h5_path = str(resolve_input(args.h5_path))   # bare names are looked up in data/hdf5/
    if args.trial or args.auto:
        trial = compression_trial(args.h5_path, args.key, keep_naive=args.keep_naive)
        print_trial(trial)
        if args.trial:
            return
        chosen = choose_settings(trial)
        print("using", chosen)
        args.compression, args.level, args.byte_stream_split = chosen["compression"], chosen["compression_level"], chosen["byte_stream_split"]
    path, _ = h5_to_parquet(
        args.h5_path,
        args.parquet_path,
        key=args.key,
        chunk_rows=args.chunk_rows,
        compression=None if args.compression == "none" else args.compression,
        compression_level=args.level if args.compression in ("zstd", "gzip") else None,
        keep_naive=args.keep_naive,
        overwrite=args.overwrite,
        byte_stream_split=args.byte_stream_split,
    )
    if not args.no_verify:
        print("round trip:", json.dumps(verify_roundtrip(args.h5_path, path, args.key, args.keep_naive)))


if __name__ == "__main__":
    main()
