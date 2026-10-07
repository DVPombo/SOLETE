"""Build every SOLETE v4 release artefact with bounded memory.

Run this file directly from Spyder in an external system terminal, or invoke
it from any working directory. Heavy stages are relaunched as subprocesses so
their memory is returned to the operating system between stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_solete_1sec import clean_dataframe  # noqa: E402
from export_parquet import h5_to_parquet  # noqa: E402
from export_parquet import _prepare  # noqa: E402
from resample_solete import resample_dataframe  # noqa: E402
from solete.expansion import expand_physical, iter_hdf_slices  # noqa: E402
from solete.io import import_PV_WT_data  # noqa: E402
from solete.paths import (  # noqa: E402
    DATA_DIR,
    HDF5_DIR,
    PARQUET_DIR,
    data_filename,
    derived_path,
    resolve_input,
)
from solete.qc_codes import MODEL_DERIVED_COLUMNS, QC_MODEL_SUBSTITUTED  # noqa: E402
from solete_report import print_report, save_report  # noqa: E402

KEY = "DATA"
MEASURED_COLUMNS = [
    "TEMPERATURE[degC]",
    "HUMIDITY[%]",
    "WIND_SPEED[m1s]",
    "WIND_DIR[deg]",
    "GHI[kW1m2]",
    "POA Irr[kW1m2]",
    "P_Gaia[kW]",
    "P_Solar[kW]",
    "Pressure[mbar]",
]
RESOLUTIONS = ("1sec", "1min", "5min", "60min")
RULES = {"1min": "1min", "5min": "5min", "60min": "60min"}
STAGES = ("original", "clean", "resample", "expand", "parquet", "verify")
EXPECTED_REAL_COUNTS = {
    "1sec": 39_484_801,
    "1min": 658_081,
    "5min": 131_617,
    "60min": 10_969,
}
EXPECTED_REAL_START = pd.Timestamp("2018-06-01 00:00:00")
EXPECTED_REAL_END = pd.Timestamp("2019-09-01 00:00:00")


def _hdf_path(resolution, *, original=False):
    return HDF5_DIR / data_filename(
        resolution, version="v4", fmt="hdf5", original=original
    )


def _parquet_path(resolution, *, original=False):
    return PARQUET_DIR / data_filename(
        resolution, version="v4", fmt="parquet", original=original
    )


def _scratch_path(name, scratch_dir=None):
    if scratch_dir:
        path = Path(scratch_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path / name
    return derived_path(name)


def _nrows(path):
    with pd.HDFStore(path, mode="r") as store:
        storer = store.get_storer(KEY)
        if storer.nrows is not None:
            return int(storer.nrows)
    with h5py.File(path, mode="r") as h5:
        return len(h5[KEY]["axis1"])


def _write_frames_atomic(path, frames, *, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite or --skip-existing")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    wrote = False
    try:
        with pd.HDFStore(temporary, mode="w", complevel=1, complib="zlib") as store:
            for frame in frames:
                if frame.empty:
                    continue
                store.append(KEY, frame, format="table", index=False)
                wrote = True
        if not wrote:
            raise ValueError(f"Refusing to create empty HDF5 output {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _contiguous_runs(path, chunk_rows):
    """Return positional one-second runs sorted by their first timestamp."""
    runs = []
    run_start = 0
    position = 0
    previous = None
    for frame in iter_hdf_slices(path, key=KEY, chunk_rows=chunk_rows):
        values = pd.DatetimeIndex(frame.index).as_unit("ns").asi8
        if len(values) == 0:
            continue
        if previous is not None and values[0] - previous != 1_000_000_000:
            runs.append((run_start, position, previous))
            run_start = position
        breaks = np.flatnonzero(np.diff(values) != 1_000_000_000) + 1
        for offset in breaks:
            absolute = position + int(offset)
            runs.append((run_start, absolute, int(values[offset - 1])))
            run_start = absolute
        position += len(values)
        previous = int(values[-1])
    if previous is not None:
        runs.append((run_start, position, previous))

    def first_timestamp(run):
        frame = next(iter_hdf_slices(path, key=KEY, chunk_rows=1, start=run[0], stop=run[0] + 1))
        return frame.index[0]

    return sorted(runs, key=first_timestamp)


def build_original(raw, output, *, chunk_rows, overwrite):
    raw_rows = _nrows(raw)
    if Path(raw).name == "SOLETE_Pombo_1sec.h5" and raw_rows != EXPECTED_REAL_COUNTS["1sec"]:
        raise ValueError(
            f"Canonical raw input has {raw_rows:,} rows, but the v4 specification requires "
            f"{EXPECTED_REAL_COUNTS['1sec']:,}. Resolve the input/release-span discrepancy before building; "
            "the builder will not silently drop rows."
        )
    available = next(iter_hdf_slices(raw, key=KEY, chunk_rows=1)).columns.tolist()
    missing = [column for column in MEASURED_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"Raw input is missing required measured columns: {missing}")
    runs = _contiguous_runs(raw, chunk_rows)

    def frames():
        last = None
        for start, stop, _ in runs:
            for frame in iter_hdf_slices(
                raw, key=KEY, chunk_rows=chunk_rows, start=start, stop=stop
            ):
                frame = frame[MEASURED_COLUMNS]
                if last is not None and frame.index[0] <= last:
                    raise ValueError("Raw contiguous runs overlap after chronological ordering")
                last = frame.index[-1]
                yield frame

    _write_frames_atomic(output, frames(), overwrite=overwrite)
    return {"runs_sorted": len(runs), "rows": _nrows(output), "columns": MEASURED_COLUMNS}


def build_clean(
    original, output, *, slice_rows, overlap, overwrite, progress_label=None
):
    total = _nrows(original)

    def frames():
        for slice_number, start in enumerate(
            range(0, total, slice_rows), start=1
        ):
            stop = min(start + slice_rows, total)
            extended_start = max(0, start - overlap)
            extended_stop = min(total, stop + overlap)
            extended = pd.concat(iter_hdf_slices(
                original,
                key=KEY,
                chunk_rows=slice_rows + 2 * overlap,
                start=extended_start,
                stop=extended_stop,
            ))
            cleaned, _ = clean_dataframe(
                extended, solar_chunk_rows=slice_rows, solar_verbose=False
            )
            if progress_label and (
                slice_number == 1 or slice_number % 30 == 0 or stop == total
            ):
                print(f"{progress_label}: {stop:,} / {total:,}", flush=True)
            yield cleaned.iloc[start - extended_start:stop - extended_start]

    _write_frames_atomic(output, frames(), overwrite=overwrite)
    return {"rows": _nrows(output), "slice_rows": slice_rows, "overlap_rows": overlap}


def build_resampled(cleaned, outputs, *, slice_rows, overwrite):
    total = _nrows(cleaned)
    stores = {}
    temporary = {}
    try:
        for resolution, output in outputs.items():
            output = Path(output)
            if output.exists() and not overwrite:
                raise FileExistsError(f"{output} exists; pass --overwrite or --skip-existing")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary[resolution] = output.with_name(output.name + ".tmp")
            temporary[resolution].unlink(missing_ok=True)
            stores[resolution] = pd.HDFStore(
                temporary[resolution], mode="w", complevel=1, complib="zlib"
            )
        for start in range(0, total, slice_rows):
            stop = min(start + slice_rows, total)
            frame = pd.concat(iter_hdf_slices(
                cleaned, key=KEY, chunk_rows=slice_rows, start=start, stop=stop
            ))
            if frame.index[0].normalize() != frame.index[0]:
                raise ValueError("Resampling slices must start at a UTC day boundary")
            for resolution, rule in RULES.items():
                stores[resolution].append(
                    KEY, resample_dataframe(frame, rule), format="table", index=False
                )
        for store in stores.values():
            store.close()
        stores.clear()
        for resolution, output in outputs.items():
            os.replace(temporary[resolution], output)
    finally:
        for store in stores.values():
            store.close()
        for path in temporary.values():
            Path(path).unlink(missing_ok=True)
    return {resolution: _nrows(output) for resolution, output in outputs.items()}


def build_expanded(source, output, *, chunk_rows, overwrite):
    pv_info, _ = import_PV_WT_data()

    def frames():
        for frame in iter_hdf_slices(source, key=KEY, chunk_rows=chunk_rows):
            yield expand_physical(frame, pv_info)

    _write_frames_atomic(output, frames(), overwrite=overwrite)
    return {"rows": _nrows(output)}


def benchmark_parquet_options(hdf_path, scratch_dir):
    frame = next(iter_hdf_slices(hdf_path, key=KEY, chunk_rows=1_000_000))
    table = pa.Table.from_pandas(_prepare(frame, keep_naive=False), preserve_index=False)
    float_columns = [field.name for field in table.schema if pa.types.is_floating(field.type)]
    results = []
    for level in (3, 9, 15):
        for byte_stream_split in (False, True):
            output = _scratch_path(
                f"parquet_trial_zstd{level}_{'bss' if byte_stream_split else 'plain'}.parquet",
                scratch_dir,
            )
            started = time.perf_counter()
            pq.write_table(
                table,
                output,
                compression="zstd",
                compression_level=level,
                use_byte_stream_split=float_columns if byte_stream_split else False,
                row_group_size=1_000_000,
                write_statistics=True,
            )
            results.append({
                "level": level,
                "use_byte_stream_split": byte_stream_split,
                "bytes": output.stat().st_size,
                "seconds": round(time.perf_counter() - started, 3),
            })
            output.unlink()
    chosen = min(results, key=lambda result: (result["bytes"], result["seconds"]))
    return results, chosen


def _assert_frames_equal(left, right, context):
    if list(left.columns) != list(right.columns):
        raise AssertionError(f"{context}: columns differ")
    if not left.index.equals(right.index):
        left_ns = pd.DatetimeIndex(left.index).as_unit("ns")
        right_ns = pd.DatetimeIndex(right.index).as_unit("ns")
        if not left_ns.equals(right_ns):
            raise AssertionError(f"{context}: timestamps differ")
    for column in left.columns:
        if left[column].dtype != right[column].dtype:
            raise AssertionError(
                f"{context}: dtype differs in {column}: "
                f"{left[column].dtype} != {right[column].dtype}"
            )
        a = left[column].to_numpy()
        b = right[column].to_numpy()
        if pd.api.types.is_numeric_dtype(left[column]):
            if not np.array_equal(a, b, equal_nan=True):
                raise AssertionError(f"{context}: values differ in {column}")
        elif not pd.Series(a).equals(pd.Series(b)):
            raise AssertionError(f"{context}: values differ in {column}")


def _compare_hdf(left_path, right_path, *, chunk_rows, context):
    if _nrows(left_path) != _nrows(right_path):
        raise AssertionError(f"{context}: row counts differ")
    total = _nrows(left_path)
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        left = pd.concat(iter_hdf_slices(left_path, chunk_rows=chunk_rows, start=start, stop=stop))
        right = pd.concat(iter_hdf_slices(right_path, chunk_rows=chunk_rows, start=start, stop=stop))
        _assert_frames_equal(left, right, context)


def _verify_original_against_raw(raw, original, *, chunk_rows):
    output_position = 0
    for start, stop, _ in _contiguous_runs(raw, chunk_rows):
        raw_position = start
        while raw_position < stop:
            raw_stop = min(raw_position + chunk_rows, stop)
            expected = pd.concat(iter_hdf_slices(
                raw, chunk_rows=chunk_rows, start=raw_position, stop=raw_stop
            ))[MEASURED_COLUMNS]
            count = len(expected)
            actual = pd.concat(iter_hdf_slices(
                original,
                chunk_rows=chunk_rows,
                start=output_position,
                stop=output_position + count,
            ))
            _assert_frames_equal(actual, expected, "original versus sorted raw")
            raw_position = raw_stop
            output_position += count


def _cleaning_counts(cleaned, *, chunk_rows):
    counts = {}
    pressure_real_days = set()
    p_gaia_active_days = set()
    for frame in iter_hdf_slices(cleaned, chunk_rows=chunk_rows):
        for column in (column for column in frame if column.endswith("_qc")):
            target = counts.setdefault(column, {})
            for code, count in frame[column].value_counts().items():
                target[int(code)] = target.get(int(code), 0) + int(count)
        pressure_real_days.update(
            frame.index[frame["Pressure[mbar]"].notna()].normalize().strftime("%Y-%m-%d")
        )
        p_gaia_active_days.update(
            frame.index[frame["P_Gaia[kW]_qc"] == 10].normalize().strftime("%Y-%m-%d")
        )
    return counts, sorted(pressure_real_days), sorted(p_gaia_active_days)


def verify_release(raw, cleaned, *, chunk_rows, slice_rows, scratch_dir):
    report = {"files": {}, "timestamp_comparison_unit": "ns"}
    for resolution in RESOLUTIONS:
        hdf = _hdf_path(resolution)
        parquet = _parquet_path(resolution)
        rows = _nrows(hdf)
        if _nrows(_hdf_path("1sec")) == EXPECTED_REAL_COUNTS["1sec"] and rows != EXPECTED_REAL_COUNTS[resolution]:
            raise AssertionError(f"{resolution}: expected {EXPECTED_REAL_COUNTS[resolution]:,} rows, got {rows:,}")
        parquet_file = pq.ParquetFile(parquet)
        if parquet_file.metadata.num_rows != rows:
            raise AssertionError(f"{resolution}: Parquet row count differs")
        for hdf_chunk in iter_hdf_slices(hdf, key=KEY, chunk_rows=chunk_rows):
            start = hdf_chunk.index[0].tz_localize("UTC")
            stop = hdf_chunk.index[-1].tz_localize("UTC")
            parquet_chunk = pd.read_parquet(
                parquet,
                filters=[("timestamp", ">=", start), ("timestamp", "<=", stop)],
            ).set_index("timestamp")
            parquet_chunk.index = parquet_chunk.index.tz_convert("UTC").tz_localize(None)
            _assert_frames_equal(hdf_chunk, parquet_chunk, f"{resolution} Parquet round trip")
        report["files"][resolution] = {"rows": rows, "parquet_timestamp": str(parquet_file.schema_arrow.field("timestamp").type)}

    original = _hdf_path("1sec", original=True)
    original_columns = next(iter_hdf_slices(original, chunk_rows=1)).columns.tolist()
    if original_columns != MEASURED_COLUMNS:
        raise AssertionError("Original release column list differs from the nine measured columns")
    index_rows = 0
    first_timestamp = None
    last_timestamp = None
    for frame in iter_hdf_slices(original, chunk_rows=chunk_rows):
        frame_index = pd.DatetimeIndex(frame.index).as_unit("ns")
        if first_timestamp is None:
            first_timestamp = frame_index[0]
        if last_timestamp is not None and frame_index[0] - last_timestamp != pd.Timedelta(seconds=1):
            raise AssertionError("Original release has gaps or duplicates between chunks")
        if len(frame_index) > 1 and not np.all(
            np.diff(frame_index.asi8) == 1_000_000_000
        ):
            raise AssertionError("Original release has gaps or duplicates in its one-second grid")
        index_rows += len(frame_index)
        last_timestamp = frame_index[-1]
    if index_rows == EXPECTED_REAL_COUNTS["1sec"]:
        if first_timestamp != EXPECTED_REAL_START or last_timestamp != EXPECTED_REAL_END:
            raise AssertionError(
                f"Original release span differs from {EXPECTED_REAL_START} through {EXPECTED_REAL_END}"
            )
    _verify_original_against_raw(raw, original, chunk_rows=chunk_rows)
    for frame in iter_hdf_slices(cleaned, chunk_rows=chunk_rows):
        qc_columns = [column for column in frame if column.endswith("_qc")]
        if qc_columns and (frame[qc_columns] == QC_MODEL_SUBSTITUTED).any().any():
            raise AssertionError("Code 6 exists in the pre-expansion cleaned intermediate")
    report["files"]["original"] = {"rows": index_rows, "columns": original_columns}

    pv_info, _ = import_PV_WT_data()
    numeric_model_columns = [
        "Pac", "Pdc", "TempModule", "TempCell", "P_Solar_clean[kW]", "P_hybrid[kW]"
    ]
    resolution_effect = {resolution: {column: 0.0 for column in numeric_model_columns} for resolution in RULES}
    resampled_expanded = {resolution: [] for resolution in RULES}
    for frame in iter_hdf_slices(_hdf_path("1sec"), chunk_rows=slice_rows):
        for resolution, rule in RULES.items():
            resampled_expanded[resolution].append(
                frame[numeric_model_columns].resample(rule, label="left", closed="left").mean()
            )
    for resolution in RULES:
        release = pd.read_hdf(_hdf_path(resolution), key=KEY)
        rerun = expand_physical(release.copy(), pv_info)
        _assert_frames_equal(
            release[list(MODEL_DERIVED_COLUMNS & set(release.columns))],
            rerun[list(MODEL_DERIVED_COLUMNS & set(release.columns))],
            f"{resolution} expansion idempotence",
        )
        averaged = pd.concat(resampled_expanded[resolution]).loc[release.index]
        for column in numeric_model_columns:
            difference = np.abs(release[column].to_numpy() - averaged[column].to_numpy())
            resolution_effect[resolution][column] = float(np.nanmax(difference))
    report["max_abs_difference_from_resample_expanded_1sec"] = resolution_effect

    counts, pressure_real_days, p_gaia_active_days = _cleaning_counts(
        cleaned, chunk_rows=chunk_rows
    )
    report["cleaning_counts"] = counts
    report["cleaning_decision_comparison"] = {
        "wind_dir_wrapped": {
            "expected": 1_343,
            "actual": counts.get("WIND_DIR[deg]_qc", {}).get(1, 0),
        },
        "pressure_real_days": {
            "expected": ["2019-01-16"],
            "actual": pressure_real_days,
        },
        "p_gaia_active_days": {
            "expected": ["2018-08-31", "2019-05-25"],
            "actual": p_gaia_active_days,
        },
    }
    for comparison in report["cleaning_decision_comparison"].values():
        comparison["matches"] = comparison["actual"] == comparison["expected"]
    if index_rows == EXPECTED_REAL_COUNTS["1sec"]:
        mismatches = [
            name
            for name, comparison in report["cleaning_decision_comparison"].items()
            if not comparison["matches"]
        ]
        report["cleaning_decision_mismatches"] = mismatches

    reproduction_clean = _scratch_path("verify_cleaned.h5", scratch_dir)
    reproduction_outputs = {
        resolution: _scratch_path(f"verify_{resolution}_measured.h5", scratch_dir)
        for resolution in RULES
    }
    reproduction_release = {
        resolution: _scratch_path(f"verify_{resolution}_release.h5", scratch_dir)
        for resolution in RESOLUTIONS
    }
    try:
        build_clean(
            original,
            reproduction_clean,
            slice_rows=slice_rows,
            overlap=300,
            overwrite=True,
            progress_label="verification clean rebuild",
        )
        _compare_hdf(cleaned, reproduction_clean, chunk_rows=chunk_rows, context="clean reproducibility")
        build_resampled(
            reproduction_clean,
            reproduction_outputs,
            slice_rows=slice_rows,
            overwrite=True,
        )
        build_expanded(
            reproduction_clean,
            reproduction_release["1sec"],
            chunk_rows=slice_rows,
            overwrite=True,
        )
        for resolution in RULES:
            build_expanded(
                reproduction_outputs[resolution],
                reproduction_release[resolution],
                chunk_rows=slice_rows,
                overwrite=True,
            )
        for resolution in RESOLUTIONS:
            _compare_hdf(
                _hdf_path(resolution),
                reproduction_release[resolution],
                chunk_rows=chunk_rows,
                context=f"{resolution} release reproducibility",
            )
        report["reproducibility"] = "exact"
    finally:
        reproduction_clean.unlink(missing_ok=True)
        for path in [*reproduction_outputs.values(), *reproduction_release.values()]:
            path.unlink(missing_ok=True)
    return report


def write_manifest(summary):
    files = [
        *(_hdf_path(resolution) for resolution in RESOLUTIONS),
        _hdf_path("1sec", original=True),
        *(_parquet_path(resolution) for resolution in RESOLUTIONS),
        _parquet_path("1sec", original=True),
    ]
    checksums = {}
    for path in files:
        print(f"SHA-256: {path.name}", flush=True)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        checksums[path.name] = digest.hexdigest()
    (DATA_DIR / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="ascii",
    )
    manifest = {"dataset": "SOLETE", "version": "v4", "build_date": date.today().isoformat(), "files": []}
    for path in files:
        manifest["files"].append({
            "name": path.name,
            "relative_path": str(path.relative_to(DATA_DIR)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": checksums[path.name],
        })
    manifest["summary"] = summary
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _run_worker(args):
    raw = resolve_input(args.raw)
    scratch = _scratch_path("SOLETE_Pombo_1sec_cleaned_v4.h5", args.scratch_dir)
    slice_rows = args.slice_days * 86_400
    if args.stage == "original":
        return build_original(raw, _hdf_path("1sec", original=True), chunk_rows=slice_rows, overwrite=args.overwrite)
    if args.stage == "clean":
        return build_clean(_hdf_path("1sec", original=True), scratch, slice_rows=slice_rows, overlap=300, overwrite=args.overwrite)
    if args.stage == "resample":
        return build_resampled(scratch, {resolution: _scratch_path(f"SOLETE_Pombo_{resolution}_measured_v4.h5", args.scratch_dir) for resolution in RULES}, slice_rows=slice_rows, overwrite=args.overwrite)
    if args.stage == "expand":
        result = {"1sec": build_expanded(scratch, _hdf_path("1sec"), chunk_rows=slice_rows, overwrite=args.overwrite)}
        for resolution in RULES:
            source = _scratch_path(f"SOLETE_Pombo_{resolution}_measured_v4.h5", args.scratch_dir)
            result[resolution] = build_expanded(source, _hdf_path(resolution), chunk_rows=slice_rows, overwrite=args.overwrite)
        return result
    if args.stage == "parquet":
        trials, chosen = benchmark_parquet_options(_hdf_path("1sec"), args.scratch_dir)
        result = {"compression_trials": trials, "chosen": chosen, "files": {}}
        for resolution, original in [("1sec", True), *((resolution, False) for resolution in RESOLUTIONS)]:
            result["files"][data_filename(resolution, "v4", "parquet", original=original)] = h5_to_parquet(
                _hdf_path(resolution, original=original),
                _parquet_path(resolution, original=original),
                chunk_rows=min(slice_rows, 1_000_000),
                compression="zstd",
                compression_level=chosen["level"],
                overwrite=args.overwrite,
                use_byte_stream_split=chosen["use_byte_stream_split"],
            )[1]
        return result
    if args.stage == "verify":
        lock = _scratch_path("build_release_verify.lock", args.scratch_dir)
        if lock.exists():
            try:
                owner = int(lock.read_text(encoding="ascii"))
            except ValueError:
                owner = None
            if owner and psutil.pid_exists(owner):
                raise RuntimeError(f"Verification is already running in process {owner}")
            lock.unlink(missing_ok=True)
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("Verification lock was acquired by another process") from error
        with os.fdopen(descriptor, "w", encoding="ascii") as lock_file:
            lock_file.write(str(os.getpid()))
        try:
            result = verify_release(
                raw,
                scratch,
                chunk_rows=min(slice_rows, 1_000_000),
                slice_rows=slice_rows,
                scratch_dir=args.scratch_dir,
            )
            save_report(DATA_DIR / "release_verification.json", result)
            write_manifest(result)
            return result
        finally:
            lock.unlink(missing_ok=True)
    raise ValueError(args.stage)


def _stage_outputs(stage, scratch_dir):
    scratch = _scratch_path("SOLETE_Pombo_1sec_cleaned_v4.h5", scratch_dir)
    if stage == "original":
        return [_hdf_path("1sec", original=True)]
    if stage == "clean":
        return [scratch]
    if stage == "resample":
        return [_scratch_path(f"SOLETE_Pombo_{resolution}_measured_v4.h5", scratch_dir) for resolution in RULES]
    if stage == "expand":
        return [_hdf_path(resolution) for resolution in RESOLUTIONS]
    if stage == "parquet":
        return [_parquet_path("1sec", original=True), *(_parquet_path(resolution) for resolution in RESOLUTIONS)]
    return [DATA_DIR / "manifest.json", DATA_DIR / "SHA256SUMS.txt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--raw", default=str(HDF5_DIR / "SOLETE_Pombo_1sec.h5"))
    parser.add_argument("--slice-days", type=int, default=1)
    parser.add_argument("--max-ram-gb", type=float, default=4.0)
    parser.add_argument("--scratch-dir")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.slice_days <= 0 or args.max_ram_gb <= 0:
        parser.error("--slice-days and --max-ram-gb must be positive")
    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing and --overwrite are mutually exclusive")
    if args.worker:
        started = time.perf_counter()
        result = _run_worker(args)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        memory = psutil.Process().memory_info()
        peak_bytes = getattr(memory, "peak_wset", memory.rss)
        result["peak_rss_gb"] = round(peak_bytes / 1024**3, 3)
        if peak_bytes > args.max_ram_gb * 1024**3:
            raise MemoryError(
                f"Stage {args.stage} exceeded --max-ram-gb: "
                f"{peak_bytes / 1024**3:.3f} > {args.max_ram_gb:.3f} GiB"
            )
        print_report(f"build_release_{args.stage}", result)
        return

    summary = {"raw": str(resolve_input(args.raw)), "slice_days": args.slice_days, "max_ram_gb": args.max_ram_gb, "stages": {}}
    for stage in args.stages:
        outputs = _stage_outputs(stage, args.scratch_dir)
        if args.skip_existing and all(path.exists() for path in outputs):
            summary["stages"][stage] = {"status": "skipped", "outputs": [str(path) for path in outputs]}
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", "--stage", stage,
            "--raw", str(resolve_input(args.raw)), "--slice-days", str(args.slice_days),
            "--max-ram-gb", str(args.max_ram_gb),
        ]
        if args.scratch_dir:
            command.extend(["--scratch-dir", args.scratch_dir])
        if args.overwrite:
            command.append("--overwrite")
        print(f"[{stage}] {' '.join(command)}", flush=True)
        if args.dry_run:
            summary["stages"][stage] = {"status": "dry-run"}
            continue
        started = time.perf_counter()
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        summary["stages"][stage] = {"status": "complete", "elapsed_seconds": round(time.perf_counter() - started, 3)}

    if not args.keep_intermediate and not args.dry_run and "verify" in args.stages:
        _scratch_path("SOLETE_Pombo_1sec_cleaned_v4.h5", args.scratch_dir).unlink(missing_ok=True)
        for resolution in RULES:
            _scratch_path(f"SOLETE_Pombo_{resolution}_measured_v4.h5", args.scratch_dir).unlink(missing_ok=True)
    save_report(DATA_DIR / "build_release_summary.json", summary)
    print_report("build_release_summary", summary)


if __name__ == "__main__":
    main()