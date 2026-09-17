# SOLETE — Known Data & Code Issues

This file is a single point of reference for data-quality issues (open and fixed) and
one process/housekeeping gap. It does not duplicate `CHANGELOG.md`'s full detail — it
points there (and to git history) for that — but it's a reference specifically for
someone deciding "can I trust this column / this file as-is."

Each entry: what it is, where it lives, current status, what to do in the meantime.

---

## Already-fixed issues (Phase 0.5 — historical record)

These three are **fixed** as of commit `5eb11d0` ("Fixing bugs related to masking in the
PV_Performance_Model, positional indexing in HUMIDITY and the
Rincon_Pombo_ThermodynamicModel while also adding a flag."). Listed here for a single
point of reference; see `CHANGELOG.md` and git history (`git show 5eb11d0`) for full
detail on each.

### 1. `PV_Performance_Model` DataFrame-wide masking bug — **FIXED**
- **What it was:** In `Functions.py::PV_Performance_Model()`, the inverter-capacity clamp
  used `Results[mask] = value` (boolean-mask assignment on the whole DataFrame) instead
  of `Results.loc[mask, 'Pac_<pv>'] = value`. Because `Results[mask] = value` applies the
  scalar to *every* column of `Results` on the masked rows, it was silently clobbering
  `Tm`, `Tc`, `Pmp_panel`, `Pmp_array`, and `eff_inv` on those rows too, not just the
  intended `Pac_<pv>` column.
- **Status:** Fixed — now uses `.loc[mask, 'Pac_' + pv]`, confined to the intended
  column. See the inline `#NOTE:` comment at `Functions.py` line ~309.
- **What a user should do in the meantime:** Nothing — this is fixed in the current code.
  Anyone who ran an older version of `PV_Performance_Model` and kept the output should be
  aware `Tm`/`Tc`/derived columns could have been silently wrong on inverter-clamped rows.

### 2. Positional-indexing break in `Rincon_Pombo_ThermodynamicModel` — **FIXED**
- **What it was:** The per-timestep loop indexed pandas Series with integer positions
  (`data['HUMIDITY[%]'][i]`, etc.) relying on pandas' old fallback from label-based to
  positional indexing for non-integer (DatetimeIndex) indexes. Modern pandas (3.0+) no
  longer does that fallback and raises `KeyError` instead.
- **Status:** Fixed — the loop now works on plain `numpy` arrays (`.to_numpy()`) so `[i]`
  is unambiguously positional. See the inline comment at `Functions.py` line ~366.
- **What a user should do in the meantime:** Nothing — fixed in current code.

### 3. Silent `P_Solar[kW]`/`Pac` substitution in `ExpandSOLETE` — **FIXED (flag added)**
- **What it was:** `ExpandSOLETE()` replaces `P_Solar[kW]` with the modeled `Pac` on rows
  where `Pac >= 1.5 * P_Solar[kW]`, as a noise/curtailment-cleaning step. This happened
  silently — there was no way to tell, downstream, which rows had been substituted.
- **Status:** Fixed — a new boolean column, `P_Solar_model_substituted`, is now added
  alongside the substitution, so it's traceable. See `Functions.py` line ~208 and the
  `P_Solar_model_substituted` entry in `DATA_DICTIONARY.md`.
- **What a user should do in the meantime:** Nothing to fix, but **do** check this flag
  before treating `P_Solar[kW]` (post-`ExpandSOLETE`) as a pure sensor reading — see the
  real substitution rates below.

---

## Open data-quality issues (found during Phase 1, real-file verified)

### 4. `Pressure[mbar]` mostly pegged to round placeholder values in the 60min file — **OPEN**
- **What it is:** In `SOLETE_Pombo_60min.h5`, 10,477 of 10,969 rows (95.5%) hold exactly
  `1000.000000` mbar, 477 rows (4.3%) hold exactly `2000.0`, and 2 rows hold `3000.0`.
  Only ~13 rows carry a plausible, non-round atmospheric pressure value (992–998 mbar).
  `SOLETE_short.h5`'s pressure column, by contrast, varies smoothly and realistically
  (1013.26–1017.47 mbar) across all 24 rows.
- **Where it lives:** `SOLETE_Pombo_60min.h5`, column `Pressure[mbar]`.
- **Current status:** Open — not fixed, not previously documented anywhere found in the
  repo (README, paper abstract, code comments).
- **What a user should do in the meantime:** Do not use `Pressure[mbar]` from the 60min
  file as a real atmospheric-pressure time series without first filtering or flagging the
  1000/2000/3000 placeholder rows. `Rincon_Pombo_ThermodynamicModel` consumes this column
  directly (`p = data['Pressure[mbar]'] * 100`, converted to Pa for CoolProp calls), so
  its output (`TempModule_RP`) inherits this problem wherever pressure is pegged.
  `<!-- TODO: confirm with maintainer -->` whether 1000/2000/3000 are known sentinel/fill
  values from the data pipeline, or a control-system/logging fault.

### 5. `HUMIDITY[%]` exceeds physically valid range (>1.0, i.e. >100%) in the 60min file — **OPEN**
- **What it is:** `HUMIDITY[%]` is stored as a fraction on [0, 1] (confirmed by
  `Rincon_Pombo_ThermodynamicModel`'s own `if humidity[i] > 1: humidity[i] = 1.0` clip
  before passing it to CoolProp). In `SOLETE_Pombo_60min.h5`, 188 of 10,969 rows (1.7%)
  exceed 1.0, up to 2.7. `SOLETE_short.h5` has no such rows (small sample, though).
- **Where it lives:** `SOLETE_Pombo_60min.h5`, column `HUMIDITY[%]`.
- **Current status:** Open. The existing clip in `Rincon_Pombo_ThermodynamicModel` only
  protects that one function's internal calculation — it does not fix or flag the
  underlying column, so any other code path (or a user reading `HUMIDITY[%]` directly)
  still sees the invalid values.
- **What a user should do in the meantime:** Clip or filter `HUMIDITY[%] > 1` before use
  if working with the 60min file directly. `<!-- TODO: confirm with maintainer -->`
  whether this is a known sensor-saturation artifact (e.g. condensation on the sensor)
  or something else.

### 6. `WIND_DIR[deg]` exceeds valid compass range (>360°) in the 60min file — **OPEN**
- **What it is:** 103 of 10,969 rows (0.94%) in `SOLETE_Pombo_60min.h5` have
  `WIND_DIR[deg]` above 360°, up to 639.34°. `SOLETE_short.h5`'s wind direction stays
  within [0°, 360°) throughout its 24 rows.
- **Where it lives:** `SOLETE_Pombo_60min.h5`, column `WIND_DIR[deg]`.
- **Current status:** Open. See `RESOLUTIONS.md` for the full writeup — this pattern is
  consistent with (but not conclusively proven to be caused by) a non-circular mean used
  when aggregating finer-resolution wind-direction readings into this file, since no
  aggregation code or finer-resolution source file exists in this repo to check against.
- **What a user should do in the meantime:** Apply `% 360` (or discard/flag) rows where
  `WIND_DIR[deg] > 360` before using this column for anything direction-sensitive (e.g.
  wind-rose plots, turbine yaw-alignment features). `<!-- TODO: confirm with maintainer -->`
  the actual aggregation method used upstream, and whether it should be considered a bug
  in the dataset-build pipeline (out of scope to fix in this repo either way, since that
  pipeline isn't part of this codebase).

### 7. `Azimuth[deg]` / `Elevation[deg]` effectively unpopulated in the 60min file — **OPEN**
- **What it is:** Both columns exist only in `SOLETE_Pombo_60min.h5` (not in the short
  file), are not referenced anywhere in `Functions.py`/`RunMe.py`/`MLForecasting.py`, and
  are not in `Control_Var['PossibleFeatures']`. 99.9% of rows in both columns are exactly
  `0.0`; the only non-zero values (10 rows for Azimuth, 9 for Elevation) fall on a single
  calendar day, 2019-01-16, out of the ~15-month file.
- **Where it lives:** `SOLETE_Pombo_60min.h5`, columns `Azimuth[deg]`, `Elevation[deg]`.
- **Current status:** Open — undocumented anywhere, and looks more like a leftover/
  partially-run computation than an intentional column.
- **What a user should do in the meantime:** Don't rely on these two columns as a real
  solar-position time series; they are not one, in this file, except for one day.
  `<!-- TODO: confirm with maintainer -->` whether these were meant to be populated
  dataset-wide (e.g. via a solar-position library run at build time that only completed
  for one day), and whether the maintainer wants them documented as intentionally-partial
  or dropped from future exports.

### 8. Row order in `SOLETE_Pombo_60min.h5` is not chronological on disk — **OPEN (usage caveat, not a data-value problem)**
- **What it is:** `pd.read_hdf('SOLETE_Pombo_60min.h5').index.is_monotonic_increasing`
  is `False` — the file's rows are stored out of timestamp order (the index starts at
  2018-11-17 and ends at 2018-09-12, mid-file). Once sorted, the timestamps are a clean,
  gap-free hourly series from 2018-06-01 to 2019-09-01 (10,968 consecutive 1-hour gaps,
  zero missing), so the *values* are fine — only the on-disk row order is scrambled.
- **Where it lives:** `SOLETE_Pombo_60min.h5` (not observed in the 24-row short file,
  which is too small a window to show reordering either way).
- **Current status:** Open for Python users. `CHANGELOG.md`'s v2.2 entry already notes
  `RunMe_matlab.m` "re-sorts the SOLETE file by timestamp on import" — i.e. this
  reordering is a known, long-standing characteristic of the delivered files, but the
  fix so far only exists on the MATLAB import path. Nothing in `Functions.py`'s
  `import_SOLETE_data()` (the Python import path) sorts the index.
  `<!-- TODO: confirm with maintainer -->` whether a Python-side sort should be added to
  `import_SOLETE_data()` too — that would be a code change, out of scope for this
  documentation-only phase, but worth flagging since it's an easy one to miss.
- **What a user should do in the meantime:** Call `.sort_index()` after loading via
  `pd.read_hdf(...)` in Python, especially before doing anything order-sensitive
  (rolling windows, `MeanPrevH`/`StdPrevH` features, plotting, `.diff()`, etc.) —
  `Control_Var['MeanPrevH']`/`StdPrevH`/etc. features in `ExpandSOLETE` use `.rolling()`,
  which is order-dependent and would be silently wrong on an unsorted DataFrame.

---

## `P_Solar_model_substituted` — real substitution rates (new measurement this phase)

Computed by loading each real file, running `Functions.py::PV_Performance_Model()` on it
with the PV parameters from `import_PV_WT_data()`, and reproducing the exact substitution
condition from `ExpandSOLETE()` (`Pac >= 1.5 * P_Solar[kW]`):

| File | Rows | Rows substituted | Substitution rate |
|---|---|---|---|
| `SOLETE_short.h5` | 24 | 5 | **20.83%** |
| `SOLETE_Pombo_60min.h5` | 10,969 | 4,204 | **38.33%** |

**These rates differ by nearly 2×, and that gap is worth flagging rather than treating as
noise.** The short file is a single low-wind, partly-cloudy June day — too small a sample
(24 rows, 5 events) to be a reliable rate estimate on its own; its 95% CI on the true rate
is wide. But even taking that into account, a headline "~38% of hourly rows in the full
15-month record have their solar-power reading replaced by a physics-model estimate" is a
substantial fraction, and anyone using `P_Solar[kW]` for anything measurement-sensitive
(e.g. validating the physics model against ground truth) should know that well over a
third of the values they'd be comparing against are themselves model output, not sensor
readings. `<!-- TODO: confirm with maintainer -->` whether ~38% substitution on the full
dataset matches expectations from when this cleaning step was originally designed, or is
higher than anticipated.

---

## Wind-direction aggregation (Task 1.2 cross-reference)

See item 6 above and the full discussion in `RESOLUTIONS.md`. Filed as **open /
unconfirmed root cause** — real out-of-range values exist in the delivered file, but
this repo has no aggregation code or finer-resolution source file to pin down why.

---

### 10. `P_Gaia[kW]` near-total zero-degeneracy — root cause unconfirmed (Phase 6) — **OPEN**
- **What it is:** `P_Gaia[kW]` is exactly `0.0` for 10,921 of 10,969 rows (99.56%) across
  the full real 15-month record. The only non-zero readings are 24 rows on 2018-08-31 and
  24 rows on 2019-05-25 (48 rows, 0.44%) — despite thousands of hours elsewhere with wind
  speed above the turbine's 3.5 m/s cut-in. This was already flagged as a benchmark
  caveat in `splits/README.md` (Phase 5); this entry is about *why*.
- **Test-split-specific numbers (Phase 6, Task 6.0, computed directly against
  `splits/v1.json`'s test block, 2,953 rows, 2019-05-01 to 2019-09-01):**
  - Wind-active rows: 24 of 2,953 (**0.81%**) — all on 2019-05-25, the split's only
    wind-active day (by design, see `splits/README.md`'s boundary-choice rationale).
  - Wind's share of `P_hybrid[kW] = P_Solar[kW] + P_Gaia[kW]` total **energy**: **3.86%**.
  - Wind's share of `P_hybrid[kW]` **variance**: **11.36%** — notably higher than its
    energy share, because the handful of active-day readings include some large spikes
    rather than a steady small contribution; this is not evidence of a strong signal, just
    of a spiky small one.
  - Solar↔wind covariance in test: **0.0068** (correlation ≈ 0.007, effectively zero).
    `var(solar) + var(wind) = 3.262` vs. the actual `var(hybrid) = 3.276` — matches a
    near-zero covariance almost exactly, i.e. as close to "no interaction between the two
    signals" as this kind of check can show.
- **Two candidate explanations, both unconfirmed, neither preferred by this repo's code
  or data alone:**
  1. **Turbine out of service.** The original Phase 5 write-up's leading guess — physically
     plausible (11 kW research turbines at a single site can sit idle for maintenance,
     grid-connection, or project reasons for long stretches), but nothing in this repo
     (logbook, maintenance record, status column) confirms it either way.
  2. **Resolution/aggregation pipeline artifact.** `RESOLUTIONS.md` already found a related,
     *confirmed-symptom* problem in the same aggregation pipeline: `WIND_DIR[deg]` in this
     same 60min file holds 103 rows (0.94%) above the physically-valid 360° compass range,
     a pattern consistent with a non-circular-mean bug in whatever built the coarser
     resolution files from the finer ones (see `RESOLUTIONS.md` and finding #6 above). No
     aggregation code or finer-resolution (`1sec`/`1min`/`5min`) source file exists in this
     repo to check directly against, for either `WIND_DIR[deg]` or `P_Gaia[kW]` — so this
     explanation is equally unconfirmed, not equally unlikely. A pipeline step that zeroes
     out (rather than mis-averages) a channel under some condition is a different failure
     mode than the angle-wrapping bug, but the same root uncertainty applies: **this repo
     cannot rule in or rule out either explanation from what's available in it.**
- **Current status:** Open. Filed with the same "confirmed symptom / unconfirmed
  mechanism" framing as finding #6, deliberately — the near-total-zero pattern is real and
  verified directly against the real file; *why* it's that way is not established.
  `<!-- TODO: confirm with maintainer -->` — the maintainer's stated plan (2026-09) is to
  pursue raw/finer-resolution data from the original SYSLAB/DTU project to settle this
  empirically, the same way a finer-resolution file would let `RESOLUTIONS.md`'s
  wind-direction question be checked directly instead of left circumstantial.
- **What a user should do in the meantime:** Treat any wind or hybrid (`P_hybrid[kW]`)
  result on this dataset as **infrastructure/methodology**, not a demonstrated finding
  about wind behavior or wind-solar complementarity, until this is resolved — see
  `examples/05_hybrid_forecasting.ipynb` and `BENCHMARKS.md`'s hybrid section for the full
  scoping. Do not cite a "joint beats independent" or "ramp-smoothing" result from this
  phase as general evidence about SOLETE's wind turbine or Danish wind-solar
  complementarity broadly — at most it's evidence about two specific calendar days.

## Housekeeping (not a data-quality issue — process gap only)

### 9. CHANGELOG entries and regression tests still missing for the three Phase 0.5 fixes
- **What it is:** The three fixes described in items 1–3 above (masking bug, positional
  indexing, substitution flag) are real and already in the code (commit `5eb11d0`), but
  `CHANGELOG.md`'s `[Unreleased]` section currently only documents the Phase 0 packaging/
  dependency-bump changes and the `sklearn` `squared=` removal — it does not yet have
  entries for these three. No regression tests exist for any of the three either.
- **Status:** Known, separate gap. Explicitly **not addressed in this phase** — Phase 1
  is dataset documentation, and CHANGELOG/test hygiene is a distinct piece of work for
  whoever picks it up next.
- **Suggested entries for that future CHANGELOG update** (for whoever does that work):
  one line each for the masking-bug fix, the positional-indexing fix, and the
  `P_Solar_model_substituted` flag addition — see items 1–3 above for the exact wording
  basis, and `git show 5eb11d0` for the diff.

---

## Summary table

| # | Issue | File(s) | Status |
|---|---|---|---|
| 1 | `PV_Performance_Model` masking bug | code (all resolutions) | Fixed |
| 2 | `Rincon_Pombo_ThermodynamicModel` positional indexing | code (all resolutions) | Fixed |
| 3 | Silent `P_Solar`/`Pac` substitution | code (all resolutions) | Fixed (now flagged) |
| 4 | `Pressure[mbar]` pegged to round placeholders | `SOLETE_Pombo_60min.h5` | Open |
| 5 | `HUMIDITY[%]` > 1.0 | `SOLETE_Pombo_60min.h5` | Open |
| 6 | `WIND_DIR[deg]` > 360° | `SOLETE_Pombo_60min.h5` | Open |
| 7 | `Azimuth`/`Elevation[deg]` effectively unpopulated | `SOLETE_Pombo_60min.h5` | Open |
| 8 | Unsorted row order on disk | `SOLETE_Pombo_60min.h5` | Open (usage caveat) |
| 9 | CHANGELOG/tests missing for Phase 0.5 fixes | n/a (process) | Housekeeping, open |
| 10 | `P_Gaia[kW]` near-total zero-degeneracy, root cause unconfirmed | `SOLETE_Pombo_60min.h5` | Open |
