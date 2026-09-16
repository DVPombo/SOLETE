# `splits/` — canonical, versioned benchmark splits

This directory holds versioned train/val/test split definitions for the SOLETE benchmark
(Phase 5, Task 5.1). Each version is a small, self-documenting JSON file — the exact
boundary timestamps plus enough metadata (row counts, date ranges, and the checks that
were run against those boundaries) that a reader doesn't need to re-derive anything to
trust the numbers.

## `v1.json`

Built from `SOLETE_Pombo_60min.h5` (the full 15-month, 10,969-row real file), loaded
through the real pipeline (`Functions.import_SOLETE_data(..., resolution="60min",
SOLETE_builvsimport="Build")`, then `.sort_index()` — see `KNOWN_ISSUES.md` #8; the file's
rows are **not** stored in chronological order on disk, and `import_SOLETE_data()` does
not sort internally, so this must happen before anything order-sensitive, including a
split).

**Method: chronological time-based blocks, not a random row shuffle.** This dataset has a
real, documented history with exactly that mistake — see the v3.0 corrigendum in
`CHANGELOG.md` ("Train/validation/test data-splitting bug… affected all results computed
with v2.3 or earlier"). A time-series forecasting benchmark that shuffled rows across
train/test would leak future information into training via adjacent timestamps and lagged
features; every block below is a contiguous, non-overlapping calendar range.

### Blocks

| Split | Start | End | Rows | % of total |
|---|---|---|---|---|
| train | 2018-06-01 00:00 | 2019-03-31 23:00 | 7,296 | 66.51% |
| val   | 2019-04-01 00:00 | 2019-04-30 23:00 |   720 |  6.56% |
| test  | 2019-05-01 00:00 | 2019-09-01 00:00 | 2,953 | 26.92% |

(7,296 + 720 + 2,953 = 10,969, the full file — every row is assigned to exactly one
block.)

### Why these particular boundaries, not a plain 70/15/15 cut

A plain proportional cut (train = first 70%, val = next 15%, test = last 15%) was tried
first and **rejected** after running Task 5.1.4's boundary check. The dataset's two
wind-active days (see below) sit at the 19.9% and 78.3% marks of the full timeline. Under
a plain 70/15/15 split, the second active day (2019-05-25) lands in **val**, not test —
which would leave the **test** split with zero rows where `P_Gaia[kW] > 0`. A wind-power
test set that is entirely zero-valued isn't a meaningful benchmark: a naive
all-zero/persistence forecast would score a perfect 0.000 MAE/RMSE, and that number would
reflect the target being degenerate, not any model's skill.

The boundaries above were chosen instead so that:
- **train** contains the first wind-active day, 2018-08-31 (24 active rows) — plenty of
  history before it too, for lag features.
- **test** contains the second wind-active day, 2019-05-25 (24 active rows) — so a wind
  metric computed on test is at least evaluated against *some* real, non-zero readings.
- **val** ends up with zero wind-active rows. This is a real limitation (documented here
  rather than hidden) — there are only two active days in the entire 15-month record, so
  any 3-way split leaves at least one block with none. Hyperparameter tuning against `val`
  should keep this in mind for any wind model; it's a non-issue for PV, which is populated
  throughout.
- Whole contiguous months were moved to hit these targets — no individual rows were
  cherry-picked, and the split is still purely chronological.

The single populated azimuth/elevation day (2019-01-16, `KNOWN_ISSUES.md` #7) falls inside
**train** either way and did not factor into the boundary choice — `Azimuth[deg]` /
`Elevation[deg]` are not benchmark targets.

### Decision — wind-power benchmark scope (Task 5.1.2, ASK FIRST)

**Decided: include wind.** The v1 canonical benchmark covers both PV (`P_Solar[kW]`) and
wind (`P_Gaia[kW]`) as forecast targets, per the maintainer's explicit choice.

**Caveat that must travel with every presentation of a wind result** (leaderboard,
`BENCHMARKS.md`, any future API/docs): `P_Gaia[kW]` is exactly `0.0` for 10,921 of 10,969
rows (99.56%) in the full record. The only non-zero readings are 24 rows on 2018-08-31 and
24 rows on 2019-05-25 — 48 rows total (0.44%) — despite thousands of hours elsewhere with
wind speed above the turbine's 3.5 m/s cut-in. Most plausibly the turbine was out of
service for nearly the entire 15-month record (unconfirmed). **A wind-power score on this
dataset is overwhelmingly a score on an all-zero target.** Near-perfect wind metrics are
expected from that alone and should not be read as forecasting skill until/unless this is
revisited (e.g. if the reason for the near-zero readings is understood and a
better-populated wind file becomes available).

### Decision — `P_Solar[kW]_qc == 6` (model-substituted) rows (Task 5.1.3, ASK FIRST)

**Decided: flagged, not auto-excluded.** Rows where `P_Solar[kW]_qc == 6` (King's PV
model substituted for the raw sensor reading — see `QC_SCHEMA.md` finding #6, ~38% of the
full record) are kept in **train, val, and test alike**. The split does not remove or
relabel them; the `P_Solar[kW]_qc` column is preserved on every row so that each
baseline/model/metric call decides for itself whether to apply a QC-exclusion mask.

This is a deliberate difference from `examples/03_pv_forecasting.ipynb`'s own convention
(which excludes `qc == 6` rows from its evaluation cells) — that notebook's approach still
applies as one *valid way to use* `metrics.py`'s exclusion mask (Task 5.2), it's just not
baked into the split itself. Keeping the split raw/unfiltered means it supports both a
strict "sensor-only" evaluation and a "score everything, note the caveat" evaluation
without needing a second split version for the difference.

| Split | `qc==6` (model-substituted) rows | `qc==0` (valid) rows |
|---|---|---|
| train | 3,232 | 4,064 |
| val   |   221 |   499 |
| test  |   751 | 2,202 |

### Reproducing this split

```python
import pandas as pd
from Functions import import_SOLETE_data, import_PV_WT_data

Control_Var = {
    "resolution": "60min",
    "SOLETE_builvsimport": "Build",
    "SOLETE_save": False,
    "OriginalFeatures": [],
    "PossibleFeatures": [],
}
PVinfo, WTinfo = import_PV_WT_data()
df = import_SOLETE_data(Control_Var, PVinfo, WTinfo).sort_index()

train = df[df.index <= "2019-03-31 23:00:00"]
val   = df[(df.index >= "2019-04-01 00:00:00") & (df.index <= "2019-04-30 23:00:00")]
test  = df[df.index >= "2019-05-01 00:00:00"]
```

Or load the boundary timestamps programmatically from `splits/v1.json` rather than
hardcoding them a second time (see `metrics.py`/`solete/dataset.py` in later tasks, which
do this).
