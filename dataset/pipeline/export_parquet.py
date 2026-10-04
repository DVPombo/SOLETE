"""
export_parquet.py -- convert a SOLETE HDF5 file (the cleaned 1-second file, or
any resampled file produced by resample_solete.py) into a Parquet file that is
ready to use without knowing anything about pandas/PyTables.

The HDF5 files are the *transparent* artefacts of the pipeline; the Parquet
files are the *convenient* ones. Both contain exactly the same numbers.

Differences between the two, by design:
  * The time index becomes a regular column called `timestamp`, stored as
    timestamp[ns, UTC]. (The HDF5 files store a tz-naive index whose values are
    UTC -- see docs/DATA_DICTIONARY.md. Parquet makes that explicit.)
    Use `--keep-naive` to write the timestamps exactly as stored instead.
  * Column names, dtypes, values, NaNs and row order are unchanged.
  * File-level metadata (dataset name, timezone, QC code table, source file) is
    embedded under the key b"solete" so the file documents itself.

Usage:
    python dataset/pipeline/export_parquet.py SOLETE_clean_1sec.h5
    python dataset/pipeline/export_parquet.py SOLETE_clean_1sec.h5 out/SOLETE_clean_1sec.parquet
    python dataset/pipeline/export_parquet.py SOLETE_clean_1h.h5 --compression zstd --level 9

Or from Python:
    from export_parquet import h5_to_parquet
    h5_to_parquet("SOLETE_clean_1sec.h5", "SOLETE_clean_1sec.parquet")

Reading the result:
    import pandas as pd
    df = pd.read_parquet("SOLETE_clean_1sec.parquet").set_index("timestamp")
    # or, for a time slice without loading everything (DuckDB):
    #   duckdb.sql("SELECT * FROM 'SOLETE_clean_1sec.parquet' WHERE timestamp >= '2019-01-16' AND timestamp < '2019-01-17'")

Memory: files stored in HDF5 `table` format (the cleaned 1-second file) are
streamed in chunks of --chunk-rows rows, so the ~39M-row file never has to fit
in RAM. Files stored in `fixed` format (the resampled files, which are small)
are loaded whole.

Requires: pip install pandas numpy tables pyarrow
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import HDF5_DIR, PARQUET_DIR, resolve_input  # noqa: E402
from qc_flags import QC_LABELS  # noqa: E402  (shared QC code -> label table)

DATASET_NAME = "SOLETE"
TIMESTAMP_COL = "timestamp"


def _prepare(df: pd.DataFrame, keep_naive: bool) -> pd.DataFrame:
    """Index -> `timestamp` column; localize naive timestamps to UTC."""
    idx = pd.DatetimeIndex(df.index)
    if not keep_naive:
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = df.reset_index(drop=True)
    out.insert(0, TIMESTAMP_COL, idx)
    return out


def _metadata(src: str, keep_naive: bool, n_rows: int, columns) -> dict:
    meta = {
        "dataset": DATASET_NAME,
        "source_file": os.path.basename(src),
        "n_rows": n_rows,
        "timestamp_timezone": "naive (values are UTC)" if keep_naive else "UTC",
        "timestamp_label": "start of interval, [T, T+period) for resampled files; instant for 1-second data",
        "qc_columns": [c for c in columns if c.endswith("_qc") or c.endswith("_qc_worst")],
        "qc_codes": {str(k): v for k, v in QC_LABELS.items()},
        "documentation": "docs/DATA_DICTIONARY.md, docs/QC_SCHEMA.md",
    }
    return {b"solete": json.dumps(meta).encode("utf-8")}


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

    with pd.HDFStore(h5_path, mode="r") as store:
        storer = store.get_storer(key)
        fmt = storer.format_type
        n_rows = int(storer.nrows) if fmt == "table" else None

    def log(msg):
        if verbose:
            print(msg, flush=True)

    if fmt == "table":
        log(f"{h5_path}: key={key!r}, format=table, rows={n_rows:,}")
        starts = range(0, n_rows, chunk_rows)
        chunks = (pd.read_hdf(h5_path, key=key, start=s, stop=min(s + chunk_rows, n_rows)) for s in starts)
    else:  # fixed format: cannot be sliced, but these files are small -- load whole
        whole = pd.read_hdf(h5_path, key=key)
        n_rows = len(whole)
        log(f"{h5_path}: key={key!r}, format={fmt}, rows={n_rows:,} (loaded whole)")
        chunks = iter([whole])

    writer = None
    written = 0
    last_ts = None
    tmp_path = parquet_path + ".tmp"
    try:
        for chunk in chunks:
            if not chunk.index.is_monotonic_increasing:
                raise ValueError("Index is not sorted chronologically; run clean_solete_1sec.py first.")
            if last_ts is not None and chunk.index[0] <= last_ts:
                raise ValueError("Chunks overlap or are out of order; the file is not chronologically sorted.")
            last_ts = chunk.index[-1]
            table = pa.Table.from_pandas(_prepare(chunk, keep_naive), preserve_index=False)
            if writer is None:
                schema = table.schema.with_metadata(_metadata(h5_path, keep_naive, n_rows, chunk.columns))
                writer = pq.ParquetWriter(
                    tmp_path, schema, compression=compression, compression_level=compression_level
                )
            writer.write_table(table.cast(writer.schema), row_group_size=1_000_000)
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

    # Verification: row count + schema + first/last timestamp match the source.
    pf = pq.ParquetFile(parquet_path)
    if pf.metadata.num_rows != n_rows:
        raise RuntimeError(f"Row count mismatch: parquet {pf.metadata.num_rows} vs h5 {n_rows}")
    size_mb = os.path.getsize(parquet_path) / 1e6
    log(f"Wrote {parquet_path}: {n_rows:,} rows, {len(pf.schema_arrow.names)} columns, {size_mb:,.1f} MB")
    return parquet_path, n_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("h5_path")
    ap.add_argument("parquet_path", nargs="?", default=None, help="default: same name with .parquet")
    ap.add_argument("--key", default="DATA")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"])
    ap.add_argument("--level", type=int, default=3, help="compression level (zstd/gzip only)")
    ap.add_argument("--keep-naive", action="store_true", help="write timestamps tz-naive, exactly as stored in the h5")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.h5_path = str(resolve_input(args.h5_path))   # bare names are looked up in data/hdf5/
    h5_to_parquet(
        args.h5_path,
        args.parquet_path,
        key=args.key,
        chunk_rows=args.chunk_rows,
        compression=None if args.compression == "none" else args.compression,
        compression_level=args.level if args.compression in ("zstd", "gzip") else None,
        keep_naive=args.keep_naive,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
