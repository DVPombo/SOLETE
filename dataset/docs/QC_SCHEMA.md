# QC_SCHEMA.md — SOLETE quality-control flag layer

Reference for the `<column>_qc` flag columns: code values, precedence, which
rule flags which column, and how flags behave when resampled. The canonical
values and severity order are defined once in `../../solete/qc_codes.py`.
Implementation is in `pipeline/qc_flags.py` and `pipeline/clean_solete_1sec.py`; the reasoning
behind each rule is in `CLEANING_DECISIONS.md` — read that first if you want
the *why* rather than the lookup table.

This dataset is the single source of truth for "is this raw measurement
trustworthy, and if not, what was done about it". The SOLETE forecasting
platform consumes these flags and keeps only one rule of its own (code 6),
because that one needs a forecasting model to compute.

## 1. Flag value set

```
 0 = QC_OK                          untouched, no issue detected
 1 = QC_WRAPPED                     WIND_DIR only: value was mod-360 wrapped
 2 = QC_PLACEHOLDER                 sentinel/flatline value replaced with NaN
 3 = QC_DROPOUT_SHORT_FIXED         WS/HUM joint dropout, short run, interpolated
 4 = QC_DROPOUT_LONG_UNTREATED      same, long run — value kept as recorded
 5 = QC_GLITCH_SHORT_FIXED          out-of-bound value(s), short run, interpolated
 6 = QC_MODEL_SUBSTITUTED           model substituted for measured PV power
 7 = QC_UNVERIFIED_PROVENANCE       value plausible & unchanged, cause not established
 8 = QC_RECOMPUTED                  value fully replaced by a model computation
 9 = QC_GLITCH_LONG_UNTREATED_NAN   out-of-bound, long run, too long to fix — set to NaN
10 = QC_ACTIVE_DAY                  confirmed-good value (currently: P_Gaia)
```

Code 6 is platform-owned and evaluated independently at every released
resolution by `solete.expansion.expand_physical`. No dataset-pipeline rule
emits it, and the resampler explicitly removes it before aggregating
pipeline-owned flags. A future dataset-side rule takes the next free number
after 10.

## 2. Bitmask vs. mutually exclusive — **decision: mutually exclusive**

Checked against the
real 1-second file, every one of the seven detection rules below targets a
different column, each with a single-purpose rule, and no cell needs two
flags at once. A plain `int8` column stays trivially readable
(`value_counts()` gives clean per-category counts) and a documented
precedence order (below) resolves any future collision deterministically.

**Precedence order** (`QC_SEVERITY_ORDER` in `solete/qc_codes.py`), highest
severity first — used when aggregating to coarser resolutions (§7) and
available for any future same-cell tie-break:

`QC_GLITCH_LONG_UNTREATED_NAN` (9) > `QC_PLACEHOLDER` (2) >
`QC_DROPOUT_LONG_UNTREATED` (4) > *(platform's 6)* >
`QC_UNVERIFIED_PROVENANCE` (7) > `QC_RECOMPUTED` (8) >
`QC_GLITCH_SHORT_FIXED` (5) > `QC_DROPOUT_SHORT_FIXED` (3) >
`QC_WRAPPED` (1) > `QC_ACTIVE_DAY` (10) > `QC_OK` (0).

Rationale: a value now missing (NaN'd) is a stronger claim than a value kept
but of unresolved reliability, which is stronger than a fully-trustworthy
non-measurement (a model computation), which is stronger than a routine
interpolation or a trivial, fully-determined correction. `QC_ACTIVE_DAY`
ranks just above `QC_OK` because it's informational (confirms a good value)
rather than a warning.

## 3. Column → flag → detection rule

| Column(s) | Flag(s) | Detection rule |
|---|---|---|
| `WIND_DIR[deg]` | 1 `QC_WRAPPED` | `value != value mod 360` (extra sensor revolutions, not different bearings) |
| `Pressure[mbar]` | 2 `QC_PLACEHOLDER` | exact multiple of 1000 (any count) **OR** part of a run ≥300 consecutive bit-identical samples |
| `WIND_SPEED[m1s]`, `HUMIDITY[%]` | 3/4 `QC_DROPOUT_*` | both `WIND_SPEED == 0` and `HUMIDITY < 0.05` in the same second; run ≤5s → 3 (interpolated), run >5s → 4 (kept, flagged) |
| `TEMPERATURE[degC]`, `HUMIDITY[%]`, `Pressure[mbar]`, `GHI[kW1m2]`, `POA Irr[kW1m2]` | 5/9 `QC_GLITCH_*` | value outside the per-column physical bound (see `CLEANING_DECISIONS.md` §6 for the bound table); run ≤3s → 5 (interpolated), run >3s → 9 (**NaN'd**, flagged) |
| `P_Gaia[kW]` | 7/10 `QC_UNVERIFIED_PROVENANCE` / `QC_ACTIVE_DAY` | day in `{2018-08-31, 2019-05-25}` → 10; every other day → 7. **No value is ever changed.** |
| `Azimuth[deg]`, `Elevation[deg]` | 8 `QC_RECOMPUTED` | every row — published values dropped entirely, replaced by pvlib (NREL SPA), UTC, south-referenced azimuth, elevation not clipped at night |
| `P_Solar[kW]` | 6 `QC_MODEL_SUBSTITUTED` | after pipeline cleaning, independently at each resolution: `Pac >= 1.5 * P_Solar[kW]` |

Row order has no `_qc` column — it's not a per-cell value problem, every value
is correct once sorted, so it doesn't fit the `<column>_qc` pattern. It's
fixed by construction (the cleaned file is written pre-sorted) rather than
flagged.

## 4. Naming convention

`<column>_qc`, e.g. `Pressure[mbar]_qc`, `HUMIDITY[%]_qc`, `WIND_DIR[deg]_qc`,
`P_Gaia[kW]_qc`, `Azimuth[deg]_qc`, `Elevation[deg]_qc` — identical
convention to the SOLETE platform's `Pressure[mbar]_qc` etc., chosen
deliberately so a consumer that already knows the platform's convention
doesn't have to learn a second one. Every column with a detection rule in §3
gets one; `WIND_DIR[deg]_qc` is kept even though post-wrap the value is
unambiguously correct (code 1 either way), for transparency about what was
touched.

## 5. Delivery format: merged, not a companion file

Every `<column>_qc` column ships **merged into the same HDF5 file** as the
data (`SOLETE_Pombo_1sec_v4.h5` and everything regenerated from it via
`resample_solete.py`), dtype `int8`. An earlier draft of this pipeline wrote
a separate `*_qcflags.h5` companion file; that's been dropped because a companion
file is one more thing to keep in sync and one more thing a consumer can
forget to load, while a merged column survives a plain `pd.read_hdf()`. The
size cost — one `int8` column per QC'd source column, 9 of the 11 raw
columns — is well under 10% of the main file's size even at 1-second
resolution.

## 6. Provenance notes

- **Pressure sentinel set**: `{1000.0, 2000.0}` confirmed as blocks,
  `3000.0` as isolated glitches, plus non-round flatline plateaus (300+
  consecutive identical samples). Whether this set is complete — e.g. a
  real but brief excursion to a round number — is not proven; see
  `CLEANING_DECISIONS.md` §4.
- **Azimuth/Elevation**: every delivered value is pvlib-recomputed; no
  original values are shipped. An old, unreconciled discrepancy between two
  checks of the *original* column (~4.72° vs. ~11–12°) is not pursued because
  those values are never delivered; see `CLEANING_DECISIONS.md`. Flag 8 marks these
  rows as *recomputed*, not as *validated to a specific accuracy*.
- **P_Gaia**: flag 7 records that the cause of a zero reading is unknown,
  not that the reading itself is suspect — most zeros may be genuinely zero
  output on a calm day, or genuinely un-logged; the pattern (not any
  individual value) is the open question. Do not treat flag 7 as
  "implausible" — it means "unresolved provenance," which is a weaker claim.

## 7. Behavior under resample/export

`resample_solete.py` aggregates `<column>_qc` columns separately from data
columns — an arithmetic mean of integer codes is meaningless. Each
`<column>_qc` becomes two fields per output bucket:

- **`<column>_qc_worst`**: the single highest-severity code present in the
  bucket, by the precedence order in §2. Lets a consumer see at a glance
  whether *any* second in the bucket was compromised, and how badly.
- **`<column>_qc_frac_flagged`**: fraction of the bucket's 1-second samples
  that were not `QC_OK`. Lets a consumer pick their own tolerance threshold
  (e.g. "drop buckets >10% flagged") instead of inheriting one baked into
  the resampler.

A bucket with zero source samples at all — a genuine gap in the 1-second
file, not just a period that happened to be `QC_OK` — gets `NaN` in both
fields, distinguishable from a genuinely all-`QC_OK` bucket (`_worst == 0`,
`_frac_flagged == 0.0`).

Only the named pipeline-owned source flags in
`solete.qc_codes.PIPELINE_QC_COLUMNS` are resampled. Model-derived columns
and code 6 are never averaged or carried upward. After measured inputs and
pipeline flags have been resampled, `expand_physical` recomputes `Pac`,
`Pdc`, temperatures, clean PV and hybrid power, then evaluates code 6 at
that target resolution. A 1-second substitution flag and an hourly
substitution flag therefore answer different questions.

**Known degenerate case, confirmed on the real 39.48M-row file (not a
bug):** for any column whose detection rule never produces `QC_OK` at all —
currently `P_Gaia[kW]` (every row is 7 or 10), `Azimuth[deg]` and
`Elevation[deg]` (every row is 8) — `<column>_qc_frac_flagged` is a constant
`1.0` across every single bucket in the whole file. It is not wrong, just
uninformative: with no `QC_OK` rows to ever divide against, "fraction
flagged" can't distinguish anything. For these three columns,
`<column>_qc_worst` is the only one of the two fields that carries any
signal (e.g. telling an "active day" `P_Gaia` bucket, code 10, apart from an
"unverified provenance" one, code 7). A consumer who filters on
`_frac_flagged < threshold` as a blanket rule across all `<column>_qc` pairs
should special-case these three rather than conclude the whole file is
unusable.

**Also confirmed on the real file:** `WIND_DIR[deg]_qc_frac_flagged` can run
as high as ~0.78 in a single one-minute bucket — i.e. most of that bucket's
60 seconds were wrapped, not just one or two isolated seconds. Don't assume
a wind-direction wrap is always a momentary, single-sample blip; it can
cluster into a run that spans most of a bucket at 1-minute resolution.

## 8. Relationship to the forecasting platform

The platform reads release flags without replacing them and adds only code
6. It builds a derived column's flag from its constituents' flags
(`P_hybrid[kW]_qc` = whichever of `P_Solar[kW]_qc` / `P_Gaia[kW]_qc` ranks higher
in the single canonical severity order above), written generically against whatever
`<constituent>_qc` columns exist, so `P_Gaia[kW]_qc` is picked up with no
platform code change. The platform's own raw-value QC rules (hourly-only
Pressure/Humidity/WindDir/Azimuth/Elevation checks) are superseded by this schema.
