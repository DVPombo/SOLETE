"""
qc_flags.py -- quality-flag helpers for the SOLETE cleaning pipeline.

The flag VOCABULARY (code numbers, labels, severity order) is not defined here any
more: it lives in `solete/qc_codes.py`, the single module shared with the forecasting
platform. This file re-exports it under the names the pipeline scripts already use and
adds the pipeline-side guard `assert_pipeline_codes`.

Code 6 (QC_MODEL_SUBSTITUTED) is platform-owned: it needs the PV model and is computed
per resolution by `solete/expansion.py`. No rule in this directory may emit it
(`assert_pipeline_codes` enforces that before anything is written). A new pipeline rule
takes the next free number after 11.

One int8 `<column>_qc` column is emitted per treated source column, MERGED into the same
DataFrame as the cleaned data (see clean_solete_1sec.py) -- not a separate companion file,
so they survive a plain `pd.read_hdf()`. Each row/column gets exactly ONE code (not a bitmask).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.qc_codes`
from solete.qc_codes import (  # noqa: E402,F401  (re-exported for the pipeline scripts)
    QC_OK,
    QC_WRAPPED,
    QC_PLACEHOLDER,
    QC_DROPOUT_SHORT_FIXED,
    QC_DROPOUT_LONG_UNTREATED,
    QC_GLITCH_SHORT_FIXED,
    QC_MODEL_SUBSTITUTED,
    QC_UNVERIFIED_PROVENANCE,
    QC_RECOMPUTED,
    QC_GLITCH_LONG_UNTREATED_NAN,
    QC_ACTIVE_DAY,
    QC_UNTREATED_IMPLAUSIBLE,
    QC_LABELS,
    QC_SEVERITY_ORDER,
    PIPELINE_OWNED_CODES,
    PLATFORM_OWNED_CODES,
    MODEL_DERIVED_COLUMNS,
)


def assert_pipeline_codes(flags, name="flag column"):
    """Raise if `flags` holds a code a dataset-pipeline rule is not allowed to emit
    (6 = model-substituted, 11 = legacy v3, or anything unknown). Call before writing."""
    present = set(np.unique(np.asarray(flags)).tolist())
    foreign = sorted(present - set(PIPELINE_OWNED_CODES))
    if foreign:
        raise ValueError(f"{name}: codes {foreign} are not pipeline-owned "
                         f"(allowed: {sorted(PIPELINE_OWNED_CODES)}); code 6 is platform-owned.")


def bool_runs(mask):
    """Start/end (inclusive) positions of maximal runs of True in a 1-D boolean array."""
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return starts, ends


def fix_short_runs(values, bad_mask, max_run, short_code, long_code, nan_long_runs=False):
    """
    values: 1-D float ndarray (NOT modified in place; a new array is returned).
    bad_mask: 1-D boolean ndarray, True = this sample is missing/implausible.
    max_run: runs of bad_mask no longer than this are linearly interpolated
        between the nearest valid (finite, non-bad) samples just outside the
        run. If only one side has a valid neighbour (run touches the start or
        end of the array), that neighbour's value is persisted across the run.
        Runs longer than max_run, or with no valid neighbour on either side,
        are left untreated (see nan_long_runs for what "untreated" means).
    nan_long_runs: if False (default -- used for the WS/HUM dropout pass),
        an untreated run keeps its originally-recorded value; the value is
        still plausible on its own (e.g. a genuine 0), it's the *cause* that's
        unresolved. If True (used for the generic physical-bound glitch pass),
        an untreated run is overwritten with NaN instead: those values are
        physically impossible by construction (outside BOUNDS), so leaving
        the impossible number in the data would be worse than admitting it's
        missing -- NaN is honest, the original value is not.
    Returns (new_values, flag) where flag is an int64 ndarray:
        0            untouched (not in bad_mask)
        short_code   part of a run that was fixed
        long_code    part of a run that was left untreated (value kept or
                     NaN'd, per nan_long_runs)
    """
    n = len(values)
    out = np.asarray(values, dtype=np.float64).copy()
    flag = np.zeros(n, dtype=np.int64)
    bad = np.asarray(bad_mask)
    starts, ends = bool_runs(bad)
    for s, e in zip(starts, ends):
        run_len = e - s + 1
        left, right = s - 1, e + 1
        has_left = left >= 0 and np.isfinite(out[left])
        has_right = right < n and np.isfinite(out[right])
        if run_len > max_run or not (has_left or has_right):
            if nan_long_runs:
                out[s:e + 1] = np.nan
            flag[s:e + 1] = long_code
            continue
        if has_left and has_right:
            out[s:e + 1] = np.linspace(out[left], out[right], run_len + 2)[1:-1]
        elif has_left:
            out[s:e + 1] = out[left]
        else:
            out[s:e + 1] = out[right]
        flag[s:e + 1] = short_code
    return out, flag


def detect_flatline_mask(values, min_run=300):
    """
    Boolean mask marking every sample that's part of a run of `min_run` or
    more CONSECUTIVE bit-for-bit identical values. Real sensor noise almost
    never holds a constant value for many minutes straight, so a long
    flatline is itself evidence of a stuck sensor or placeholder --
    independent of whether the value looks "round". NaNs are never flagged.
    Use this alongside (not instead of) detect_round_sentinels: it catches
    sentinels that happen to not be round numbers (e.g. a plateau at 997.5).
    """
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    same_as_prev = np.zeros(len(v), dtype=bool)
    same_as_prev[1:] = (v[1:] == v[:-1]) & finite[1:] & finite[:-1]
    starts, ends = bool_runs(same_as_prev)
    mask = np.zeros(len(v), dtype=bool)
    for s, e in zip(starts, ends):
        run_len = e - s + 2  # `run_len` identical samples span indices [s-1, e]
        if run_len >= min_run:
            mask[s - 1:e + 1] = True
    return mask


def detect_round_sentinels(series, round_to=100.0, min_count=500):
    """
    Auto-detect placeholder/sentinel values in a numeric series: exact
    multiples of `round_to` that appear far more often than real-world
    variance would produce (more than `min_count` times). Returns
    (sorted_list_of_sentinel_values, {value: count}) -- print/inspect this
    before trusting it; it is a heuristic, not a certainty.
    """
    vc = series.value_counts()
    is_round = np.isclose(np.mod(vc.index.to_numpy(dtype=float), round_to), 0.0, atol=1e-6)
    cand = vc[is_round & (vc.to_numpy() > min_count)]
    return sorted(cand.index.tolist()), {float(k): int(v) for k, v in cand.items()}
