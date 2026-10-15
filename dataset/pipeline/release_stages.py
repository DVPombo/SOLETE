# -*- coding: utf-8 -*-
"""
release_stages.py -- the stages of the v4 build as functions of explicit paths.

build_release.py calls these (each in its own subprocess) and release_verify.py calls them again, with other
paths and other slice sizes, for the reproducibility check. No rule lives here: the cleaning rules are in
clean_solete_1sec.clean_block, the resampling in resample_solete.resample_dataframe, the model columns in
solete.expansion.expand_physical, the Parquet writing in export_parquet. This module only moves slices
between them.

Stage order (docs: dataset/README.md "Building the v4 release"):
  original  raw v3 1 s            -> SOLETE_Pombo_1sec_original_v4.h5   (make_original.py)
  clean     _original             -> scratch cleaned 1 s (sliced)       (clean_solete_1sec.clean_file)
  resample  scratch cleaned 1 s   -> scratch 1min / 5min / 60min        (resample_solete.resample_file)
  expand    scratch files         -> SOLETE_Pombo_{1sec,1min,5min,60min}_v4.h5 (expand_physical)
  parquet   the five h5 files     -> the five .parquet files            (export_parquet.h5_to_parquet)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from solete.expansion import expand_physical  # noqa: E402
from solete.h5io import TableWriter, h5_info, read_rows, write_fixed  # noqa: E402
from solete.params import import_PV_WT_data  # noqa: E402
import clean_solete_1sec as cl  # noqa: E402
import export_parquet as ep  # noqa: E402
import make_original as mo  # noqa: E402
import resample_solete as rs  # noqa: E402

COARSE = ("1min", "5min", "60min")
KEY = "DATA"
DAY = 86400


def stage_original(raw, out, overwrite=False, chunk_rows=2_000_000):
    rep = mo.build_original(raw, out, KEY, chunk_rows, overwrite)
    rep["verification"] = mo.verify_original(raw, out, KEY)
    return rep


def stage_clean(original, cleaned, slice_days=31, overwrite=False, jobs=1):
    return cl.clean_file(original, cleaned, KEY, slice_rows=max(int(slice_days * DAY), 1), overwrite=overwrite, jobs=jobs)


def stage_resample(cleaned, resampled_by_rule, slice_days=31, overwrite=False, jobs=1):
    """resampled_by_rule: {'1min': path, '5min': path, '60min': path} (fixed-format files)."""
    results = rs.resample_file(cleaned, KEY, list(resampled_by_rule), slice_days, jobs=jobs)
    rep = {}
    for rule, path in resampled_by_rule.items():
        write_fixed(results[rule], path, KEY, overwrite=overwrite)
        rep[rule] = {"rows": len(results[rule]), "columns": len(results[rule].columns)}
    kept = [c for c in h5_info(cleaned, KEY)["columns"] if c not in rs.SKIP_MODEL_COLUMNS]
    rep["_columns_resampled"] = kept
    return rep


def stage_expand_coarse(resampled, out, overwrite=False):
    PV, _ = import_PV_WT_data()
    df = read_rows(resampled, KEY)
    expand_physical(df, PV)
    write_fixed(df, out, KEY, overwrite=overwrite)
    return {"rows": len(df), "columns": len(df.columns)}


def _expand_one_slice(args):
    """Worker body for the `jobs > 1` path: reads only its own row range (never the whole file) and
    expands it. Module-level (not nested) so it is picklable on Windows."""
    cleaned, a, b, PV = args
    block = read_rows(cleaned, KEY, a, b)
    expand_physical(block, PV)          # pointwise: any slicing gives the same rows
    return block


def stage_expand_1s(cleaned, out, slice_days=31, overwrite=False, jobs=1):
    PV, _ = import_PV_WT_data()
    n = h5_info(cleaned, KEY)["nrows"]
    step = max(int(slice_days * DAY), 1)
    bounds = [(a, min(a + step, n)) for a in range(0, n, step)]
    jobs = max(1, min(int(jobs), len(bounds)))
    written = 0
    with TableWriter(out, KEY, overwrite=overwrite) as w:
        if jobs == 1:
            for a, b in bounds:
                block = read_rows(cleaned, KEY, a, b)
                expand_physical(block, PV)          # pointwise: any slicing gives the same rows
                w.append(block)
                written += len(block)
                print(f"  expand 1 s: {written:,} / {n:,}", flush=True)
                del block
        else:
            tasks = [(cleaned, a, b, PV) for a, b in bounds]
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                # pool.map preserves input order: results line up with `bounds` even though workers
                # may finish out of order, so writes below stay in original slice order.
                for block in pool.map(_expand_one_slice, tasks):
                    w.append(block)
                    written += len(block)
                    print(f"  expand 1 s: {written:,} / {n:,} ({jobs} jobs)", flush=True)
                    del block
    return {"rows": written}


def stage_parquet(h5_by_name, parquet_by_name, trial_source=None, overwrite=False, compression=None):
    """Export every file. The compression settings come from a measured trial on `trial_source` (the 1 s
    file) unless `compression` is given."""
    chosen = compression
    trial = None
    if chosen is None and trial_source is not None:
        trial = ep.compression_trial(trial_source, KEY)
        ep.print_trial(trial)
        chosen = ep.choose_settings(trial)
    chosen = chosen or {"compression": "zstd", "compression_level": 3, "byte_stream_split": False}
    print("parquet settings:", chosen, flush=True)
    rep = {"settings": chosen, "trial": trial, "files": {}}
    for name, h5 in h5_by_name.items():
        _, n = ep.h5_to_parquet(h5, parquet_by_name[name], KEY, overwrite=overwrite, **chosen)
        rep["files"][name] = {"rows": n, "bytes": Path(parquet_by_name[name]).stat().st_size}
    return rep
