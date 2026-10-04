"""
qc_flags.py -- shared quality-flag codes and generic helpers for the SOLETE
cleaning pipeline.

This is the dataset side's half of a single flag axis shared with the
SOLETE forecasting platform (the SOLETEplatform repo). Code 6 is deliberately left unassigned here --
it belongs to the platform's QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED,
which is not a property of the raw sensor stream and can't be derived from
this repo. Do not repurpose 6 for anything dataset-side; a future rule that
needs a new code takes the next free number after 10, not 6.

One int8 `<column>_qc` column is emitted per treated source column, MERGED
into the same DataFrame as the cleaned data (see clean_solete_1sec.py) --
not a separate companion file, so they survive a plain `pd.read_hdf()`.

Each row/column gets exactly ONE code (not a bitmask -- simpler to read,
and in this dataset the issues we found don't overlap on the same second).
"""
import numpy as np

QC_OK = 0                          # untouched, no issue detected
QC_WRAPPED = 1                      # WIND_DIR only: value was mod-360 wrapped
QC_PLACEHOLDER = 2                  # sentinel value detected and replaced with NaN
QC_DROPOUT_SHORT_FIXED = 3          # WS/HUM joint-zero dropout, run short enough to interpolate
QC_DROPOUT_LONG_UNTREATED = 4       # same, but run too long -- value kept as recorded (still a
                                     # plausible reading, e.g. 0), flagged for manual review
QC_GLITCH_SHORT_FIXED = 5           # isolated out-of-bound value(s), interpolated
# 6 is RESERVED for the platform repo's QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED.
# Not used, not defined, not emitted by anything in this repo.
QC_UNVERIFIED_PROVENANCE = 7        # value is plausible and UNCHANGED, but its cause is not
                                     # established (currently: P_Gaia zero outside the two
                                     # confirmed-active days)
QC_RECOMPUTED = 8                   # value fully replaced by a model computation (Azimuth/Elevation)
QC_GLITCH_LONG_UNTREATED_NAN = 9    # out-of-bound run too long to interpolate -- set to NaN
                                     # (not fabricated, not left physically impossible), flagged
QC_ACTIVE_DAY = 10                  # P_Gaia on a day with confirmed real telemetry

QC_LABELS = {
    QC_OK: "ok",
    QC_WRAPPED: "wrapped_mod_360",
    QC_PLACEHOLDER: "placeholder_set_to_nan",
    QC_DROPOUT_SHORT_FIXED: "dropout_short_run_interpolated",
    QC_DROPOUT_LONG_UNTREATED: "dropout_long_run_untreated",
    QC_GLITCH_SHORT_FIXED: "glitch_short_run_interpolated",
    # 6: platform-owned, deliberately absent here.
    QC_UNVERIFIED_PROVENANCE: "unverified_provenance",
    QC_RECOMPUTED: "recomputed_replacing_measurement",
    QC_GLITCH_LONG_UNTREATED_NAN: "glitch_long_run_untreated_set_to_nan",
    QC_ACTIVE_DAY: "p_gaia_confirmed_active_day",
}

# Severity precedence for aggregating flags to coarser resolutions (see
# resample_solete.py's `<column>_qc_worst`), highest severity first. A
# collapsed bucket reports the single worst code present in it. Code 6 is
# included so a resampled file that later gets a platform-computed
# substitution column merged in can share one precedence list; nothing in
# this repo emits 6 today.
QC_SEVERITY_ORDER = (
    QC_GLITCH_LONG_UNTREATED_NAN,   # data now missing (physically-impossible run, unfixable)
    QC_PLACEHOLDER,                 # data now missing (sentinel run, unfixable)
    QC_DROPOUT_LONG_UNTREATED,      # value kept but reliability unresolved (long dropout)
    6,                               # QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED (platform-owned)
    QC_UNVERIFIED_PROVENANCE,       # value kept, cause of the reading not established
    QC_RECOMPUTED,                  # not a measurement at all, but trustworthy (model output)
    QC_GLITCH_SHORT_FIXED,          # interpolated over a short run
    QC_DROPOUT_SHORT_FIXED,         # interpolated over a short run
    QC_WRAPPED,                     # trivial, fully-determined correction
    QC_ACTIVE_DAY,                  # confirmed good; informational only
    QC_OK,                          # untouched, no issue
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
