# SOLETE — Resolutions and Aggregation Methodology

## Available resolutions

`Control_Var['resolution']` in `Functions.py::import_SOLETE_data()` and
`MLForecasting.py` accepts exactly four values: `'1sec'`, `'1min'`, `'5min'`, `'60min'`.
Each maps to a file named `SOLETE_Pombo_<resolution>.h5` expected at the repo root
(`Functions.py`, line ~73: `name = 'SOLETE_Pombo_'+ Control_Var['resolution'] +'.h5'`).

The paper's abstract states the underlying sensors sample at 1 Hz (1-second) and that
the released dataset is "averaged over 5 min and hourly intervals" [1]. The README
separately discusses a 1-second file explicitly ("Note that the latest version of the
SOLETE dataset includes a 1sec resolution version. The file is quite large..."), and the
CHANGELOG's v2.3 entry ("Execution no longer stops when using the 1-second and 1-minute
resolution versions of the dataset") confirms a 1-minute version exists too. So the full
dataset (as distributed via the separate DOI in the README, not this repo) has (at
least) four resolutions: 1sec, 1min, 5min, 60min.

## Which resolution(s) the files in this repo represent

Only two real files are present at the repo root, and **both are 1-hour resolution**:

- `SOLETE_short.h5` — 24 rows, 1 calendar day (2018-06-01), hourly.
- `SOLETE_Pombo_60min.h5` — 10,969 rows, spanning 2018-06-01 to 2019-09-01 (~15 months,
  once sorted — see caveat below), hourly, with **zero gaps**: after sorting the index,
  every consecutive pair of timestamps is exactly 1 hour apart across all 10,968
  intervals.

No `SOLETE_Pombo_1sec.h5`, `SOLETE_Pombo_1min.h5`, or `SOLETE_Pombo_5min.h5` file is
present in this repo. Everything below about "the aggregation method" is therefore based
on reading the README, the paper's abstract, and the code — **not** on being able to
empirically compare a finer-resolution file against a coarser one, because no finer file
was available to compare against. That empirical spot-check (step 2 of the original task)
could not be performed for lack of a second file at a different resolution; the two real
files found are both hourly.

## How aggregation is actually done

**There is no resampling or aggregation code anywhere in this repo.** A full-text search
of `Functions.py`, `RunMe.py`, and `MLForecasting.py` for `resample`, `groupby`, or any
comparable rolling/aggregation logic applied to raw sensor data turns up nothing that
converts 1-second data into 1-minute/5-minute/60-minute data. (The one hit for
"aggregate" in `Functions.py`, around line 769, is an unrelated comment inside the ML
train/test-set concatenation code, not resolution aggregation.)

This means: the different resolution `.h5` files are **pre-built and delivered as
separate files** (via the dataset's own DOI, per the README), not generated from a
single raw file by this codebase. `import_SOLETE_data()` simply loads whichever
resolution's `.h5` file is asked for — it does not aggregate anything itself.

**Consequence: the aggregation method used to build `SOLETE_Pombo_5min.h5` etc. from the
1-second source cannot be verified against this repo's code, because that code doesn't
exist here.** It can only be described from the paper, and the paper's abstract does not
go into per-variable aggregation methodology (e.g. it doesn't say whether every column
uses a plain arithmetic mean, or whether wind direction gets special circular-mean
handling). The full paper text was not accessible during this phase (ScienceDirect is
paywalled for the body text; the PMC mirror served a bot-check page instead of content).

**This is filed as an ASK-FIRST item:** `<!-- TODO: confirm with maintainer --> `the
exact per-variable aggregation method (especially for `WIND_DIR[deg]`) used to build the
finer-resolution files into the coarser ones is not verifiable from what's in this repo.

## Wind direction: naive mean vs. circular mean

This was the one thing we *could* check empirically, indirectly, using the 60min file
alone (see `DATA_DICTIONARY.md` for the full write-up):

**`WIND_DIR[deg]` in `SOLETE_Pombo_60min.h5` contains 103 of 10,969 rows (0.94%) with
values above 360°, up to a maximum of 639.34°.** A valid compass bearing cannot exceed
360°, so wherever this file's `WIND_DIR[deg]` values come from an averaging step, that
step is not wrapping the result back into `[0, 360)`. This is the exact failure mode a
naive arithmetic mean of angles produces (e.g. averaging two readings near the 0°/360°
seam without unwrapping can push the raw arithmetic mean past 360° before/without a
modulo step, and — separately — a plain arithmetic mean of 359° and 1° gives 180°, which
is compass-wise backwards, even when the result happens to stay under 360°). The
above-360° values are the unambiguous, checkable half of that failure mode; the
"180°-for-359°-and-1°" failure mode is plausible but not independently checkable from a
single already-aggregated file, since we can't see the finer-resolution inputs that went
into any specific hourly value.

**Bottom line: this looks like a real circular-mean bug in whatever produced
`WIND_DIR[deg]` for the 60min file, but it cannot be fully confirmed (attributed to a
specific aggregation step, vs. e.g. a sensor artifact, vs. something else) without either
the source code that built this file or a finer-resolution file to compare it against —
neither of which is in this repo.** Filed as a data-quality item in `KNOWN_ISSUES.md`,
not fixed here (out of scope for Phase 1, which is documentation-only).

## Summary answer to "did you find a circular-mean issue with wind direction?"

**Found strong circumstantial evidence of one** (103 out-of-range values, a pattern
consistent with unwrapped angle averaging), **but could not conclusively confirm the
root cause** because no aggregation code or finer-resolution source file exists in this
repo to check against. Treat as a confirmed *symptom* (invalid angle values exist in the
delivered file) with an unconfirmed *mechanism* (presumed naive/non-circular averaging
upstream, not verified against actual aggregation code).

---

## References
[1] Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a 15-month long holistic
    dataset including: Meteorology, co-located wind and solar PV power from Denmark with
    various resolutions. *Data in Brief*, 42, 108046.
