"""
qc_flags.py -- shared quality-flag codes and generic helpers for the SOLETE
cleaning pipeline.

The canonical flag axis is defined in solete.qc_codes and shared with the
forecasting platform. Code 6 belongs to the platform's model-substitution
rule, which is not a property of the raw sensor stream. No pipeline rule in
this module imports or emits it; a future pipeline rule takes the next free
number after 10.

One int8 `<column>_qc` column is emitted per treated source column, MERGED
into the same DataFrame as the cleaned data (see clean_solete_1sec.py) --
not a separate companion file, so they survive a plain `pd.read_hdf()`.

Each row/column gets exactly ONE code (not a bitmask -- simpler to read,
and in this dataset the issues we found don't overlap on the same second).
"""
import numpy as np

from solete.qc_codes import (
    QC_ACTIVE_DAY,
    QC_DROPOUT_LONG_UNTREATED,
    QC_DROPOUT_SHORT_FIXED,
    QC_GLITCH_LONG_UNTREATED_NAN,
    QC_GLITCH_SHORT_FIXED,
    QC_LABELS,
    QC_OK,
    QC_PLACEHOLDER,
    QC_RECOMPUTED,
    QC_SEVERITY_ORDER,
    QC_UNVERIFIED_PROVENANCE,
    QC_WRAPPED,
)


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
