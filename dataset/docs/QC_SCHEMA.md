# QC_SCHEMA.md — SOLETE quality-control flag layer

Reference for the `<column>_qc` flag columns: code values, precedence, which
rule flags which column, and how flags behave when resampled. Implementation
is in `pipeline/qc_flags.py` and `pipeline/clean_solete_1sec.py`; the reasoning
behind each rule is in `CLEANING_DECISIONS.md` — read that first if you want
the *why* rather than the lookup table.

This dataset is the single source of truth for "is this raw measurement
trustworthy, and if not, what was done about it". The SOLETE forecasting
platform consumes these flags and keeps only one rule of its own (code 6),
because that one needs a PV model to compute. **There is one vocabulary:**
the code numbers, labels and severity order are defined once, in
`solete/qc_codes.py`, and imported by `pipeline/qc_flags.py` and by the platform.

## 1. Flag value set

```
 0 = QC_OK                          untouched, no issue detected
 1 = QC_WRAPPED                     WIND_DIR only: value was mod-360 wrapped
 2 = QC_PLACEHOLDER                 sentinel/flatline value replaced with NaN
 3 = QC_DROPOUT_SHORT_FIXED         WS/HUM joint dropout, short run, interpolated
 4 = QC_DROPOUT_LONG_UNTREATED      same, long run — value kept as recorded
 5 = QC_GLITCH_SHORT_FIXED          out-of-bound value(s), short run, interpolated
 6 = QC_MODEL_SUBSTITUTED           PLATFORM-OWNED: measured P_Solar[kW] replaced by the PV model
                                    (Pac >= 1.5 * P_Solar[kW]); computed per resolution, see below
 7 = QC_UNVERIFIED_PROVENANCE       value plausible & unchanged, cause not established
 8 = QC_RECOMPUTED                  RESERVED, not emitted by any v4 column (decision D1; see 3b). Never reuse the number.
 9 = QC_GLITCH_LONG_UNTREATED_NAN   out-of-bound, long run, too long to fix — set to NaN
10 = QC_ACTIVE_DAY                  confirmed-good value (currently: P_Gaia)
11 = QC_UNTREATED_IMPLAUSIBLE       LEGACY v3 only: implausible value detected, kept as recorded
```

**Ownership.** Codes 0-5 and 7-10 are *pipeline-owned*: written by `pipeline/*.py` from the
raw sensor stream. Code 6 is *platform-owned*: it is a property of a PV-model comparison, not of
the sensor stream. No rule in `pipeline/` may emit it (`assert_pipeline_codes` in `qc_flags.py`
is called before the cleaned file is written, and a test enforces it), and it appears only in
`P_Solar[kW]_qc` of the expanded ("full") files, written by `solete/expansion.py`; there is no separate boolean column (`P_Solar[kW]_qc == 6` is the substitution). Code 11 exists only for the
original v3 files, which were never cleaned by this pipeline: the platform's old raw-value checks
(implausible pressure/humidity/wind direction, azimuth/elevation == 0) are kept for them and now
write 11. A v4 file never contains 11. A new pipeline rule takes the next free number after 11.

**Code 6 is evaluated per resolution.** Whether a row is "substituted" depends on
`Pac >= 1.5 * P_Solar[kW]` evaluated on *that resolution's own cleaned inputs*. The flag at 1 s,
1 min, 5 min and 1 h are four different computations; they are never derived from one another and
code 6 is never carried upward by resampling (`resample_solete.py` drops every model-derived column
and neutralises a 6 found inside a pipeline flag column). A 1-second code 6 is not comparable with
an hourly one. See `METHODOLOGY.md`, "Model columns are per resolution".

**What code 6 means.** On a given row: the model estimate is at least 1.5 times the measurement *and* the model is producing
(stored `Pac > 0`, i.e. above 0.001 kW). Without the second condition a dark row (measured 0, model 0) would satisfy `0 >= 1.5 * 0`;
that was the behaviour up to the v3 platform and it flagged 4,204 of 10,969 hourly rows (38.33 %), every one a night row, none a real
substitution. The condition was added in v4; `P_Solar_clean[kW]` and `P_hybrid[kW]` are bit-identical either way, only the flag changed.
On the v3 hourly file and the sample files no row carries code 6 under the current rule.

## 2. Bitmask vs. mutually exclusive — **decision: mutually exclusive**

Checked against the
real 1-second file, every one of the seven detection rules below targets a
different column, each with a single-purpose rule, and no cell needs two
flags at once. A plain `int8` column stays trivially readable
(`value_counts()` gives clean per-category counts) and a documented
precedence order (below) resolves any future collision deterministically.

**Precedence order** (`QC_SEVERITY_ORDER` in `solete/qc_codes.py`, re-exported by `qc_flags.py`), highest
severity first — used when aggregating to coarser resolutions (§7) and
available for any future same-cell tie-break:

`QC_GLITCH_LONG_UNTREATED_NAN` (9) > `QC_PLACEHOLDER` (2) >
`QC_DROPOUT_LONG_UNTREATED` (4) > `QC_UNTREATED_IMPLAUSIBLE` (11, v3 only) >
`QC_MODEL_SUBSTITUTED` (6) > `QC_UNVERIFIED_PROVENANCE` (7) > `QC_RECOMPUTED` (8) >
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
| `WIND_DIR[deg]` | 1 `QC_WRAPPED` | `value != value mod 360` (extra sensor revolutions, not different bearings; in the real file 1,153 values above 360° and 190 values of exactly 360.0, which become 0.0) |
| `Pressure[mbar]` | 2 `QC_PLACEHOLDER` | exact multiple of 1000 (any count) **OR** part of a run ≥300 consecutive bit-identical samples |
| `WIND_SPEED[m1s]`, `HUMIDITY[%]` | 3/4 `QC_DROPOUT_*` | both `WIND_SPEED == 0` and `HUMIDITY < 0.05` in the same second; run ≤5s → 3 (interpolated), run >5s → 4 (kept, flagged) |
| `TEMPERATURE[degC]`, `HUMIDITY[%]`, `Pressure[mbar]`, `GHI[kW1m2]`, `POA Irr[kW1m2]` | 5/9 `QC_GLITCH_*` | value outside the per-column physical bound (see `CLEANING_DECISIONS.md` §6 for the bound table); run ≤3s → 5 (interpolated), run >3s → 9 (**NaN'd**, flagged) |
| `P_Gaia[kW]` | 7/10 `QC_UNVERIFIED_PROVENANCE` / `QC_ACTIVE_DAY` | day in `{2018-08-31, 2019-05-25}` → 10; every other day → 7. **No value is ever changed.** |
| `Azimuth[deg]`, `Elevation[deg]` | *(no flag column in v4)* | any published values are dropped entirely and every row is replaced by pvlib (NREL SPA), UTC, south-referenced azimuth, elevation not clipped at night. The flag would be a constant 8, so it is not stored (decision D1); code 8 stays reserved and **not emitted by any v4 column** |
| `P_Solar[kW]` (expanded files only) | 6 `QC_MODEL_SUBSTITUTED` | **platform-owned**, not computed by `pipeline/`: `Pac >= 1.5 * P_Solar[kW]` per resolution, by `solete/expansion.py` |

### 3b. Is every code useful? (audit, v4)

| Code | Emitted by | Distinct meaning? | Verdict |
|---|---|---|---|
| 0 | all | "nothing found" | keep |
| 1 | WIND_DIR | value changed (wrapped); the only "trivial correction" | keep |
| 2 | Pressure | value now NaN, sentinel/flatline | keep |
| 3 / 4 | WIND_SPEED, HUMIDITY | same cause (joint dropout), short run fixed vs long run kept; the split tells the user whether the value was touched | keep |
| 5 / 9 | TEMPERATURE, HUMIDITY, Pressure, GHI, POA | same cause (out-of-bound), short run fixed vs long run NaN'd | keep |
| 6 | platform (`P_Solar[kW]_qc`) | measured value replaced by the model; only record of it (no boolean column) | keep |
| 7 / 10 | P_Gaia | 10 = the two confirmed-active days, 7 = every other day (cause of the zeros unresolved); no value changed | keep: they separate the days that can be trusted from the rest |
| 8 | *(none: reserved)* | was Azimuth, Elevation: **constant** 8 on every row, so the two columns carried no per-row information; the provenance is stated in §3, `DATA_DICTIONARY.md`, the figshare README and the Parquet metadata | **dropped in v4 (decision D1)**; the code stays defined, not emitted by any v4 column |
| 11 | platform, v3 files only | "implausible value kept as recorded"; no other code means that | keep until the v3 loading path is retired |

Code 8 was the only one that did not distinguish rows. **Decision D1 (confirmed): `Azimuth[deg]_qc` and `Elevation[deg]_qc` are not part of the v4 release.**
The angles are still recomputed by the pipeline and the fact is documented instead. The number 8 stays defined in `solete/qc_codes.py` (marked "not emitted by
any v4 column"), so files from earlier development runs still load, and it is never reused. The platform's v3 legacy rules are unaffected (they create
their own `Azimuth[deg]_qc` / `Elevation[deg]_qc`, code 11, for the old v3 file only).

Row order has no `_qc` column — it's not a per-cell value problem, every value
is correct once sorted, so it doesn't fit the `<column>_qc` pattern. It's
fixed by construction (the cleaned file is written pre-sorted) rather than
flagged.

## 4. Naming convention

`<column>_qc`, e.g. `Pressure[mbar]_qc`, `HUMIDITY[%]_qc`, `WIND_DIR[deg]_qc`,
`P_Gaia[kW]_qc` — identical
convention to the SOLETE platform's `Pressure[mbar]_qc` etc., chosen
deliberately so a consumer that already knows the platform's convention
doesn't have to learn a second one. Every column with a detection rule in §3
gets one; `WIND_DIR[deg]_qc` is kept even though post-wrap the value is
unambiguously correct (code 1 either way), for transparency about what was
touched.

## 5. Delivery format: merged, not a companion file

Every `<column>_qc` column ships **merged into the same HDF5 file** as the
data (`SOLETE_Pombo_1sec_v4.h5`; the 1 min / 5 min / 60 min files carry `_qc_worst` / `_qc_frac_flagged`, `SOLETE_Pombo_{1min,5min,60min}_v4.h5`), dtype `int8`. An earlier draft of this pipeline wrote
a separate `*_qcflags.h5` companion file; that's been dropped because a companion
file is one more thing to keep in sync and one more thing a consumer can
forget to load, while a merged column survives a plain `pd.read_hdf()`. The
size cost — one `int8` column per QC'd source column, 8 of the 11 raw
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
  those values are never delivered; see `CLEANING_DECISIONS.md`. These
  rows are recomputed, not as validated to a specific accuracy (there is no flag column for it in v4, D1).
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

**Known degenerate case, confirmed on the real 39.48M-row file (not a
bug):** for any column whose detection rule never produces `QC_OK` at all —
currently `P_Gaia[kW]` only (every row is 7 or 10; verified in the build: `release_verify` and the tests) —
`P_Gaia[kW]_qc_frac_flagged` is a constant
`1.0` across every single bucket in the whole file. It is not wrong, just
uninformative: with no `QC_OK` rows to ever divide against, "fraction
flagged" can't distinguish anything. For this column,
`P_Gaia[kW]_qc_worst` is the only one of the two fields that carries any
signal (e.g. telling an "active day" `P_Gaia` bucket, code 10, apart from an
"unverified provenance" one, code 7). A consumer who filters on
`_frac_flagged < threshold` as a blanket rule across all `<column>_qc` pairs
should special-case `P_Gaia` rather than conclude the whole file is
unusable.

**Also confirmed on the real file:** `WIND_DIR[deg]_qc_frac_flagged` can run
as high as ~0.78 in a single one-minute bucket — i.e. most of that bucket's
60 seconds were wrapped, not just one or two isolated seconds. Don't assume
a wind-direction wrap is always a momentary, single-sample blip; it can
cluster into a run that spans most of a bucket at 1-minute resolution.

## 8. Relationship to the forecasting platform

The platform reads the `<column>_qc` columns of a v4 file as they are (it never recomputes or overwrites
them) and adds exactly one flag of its own, code 6 on `P_Solar[kW]_qc`, together with the other
deterministic model columns, by `solete.expansion.expand_physical` (see `DATA_DICTIONARY.md`, "Model-derived
columns"). A derived column's flag is inherited from its constituents by the single severity order of §2:
`P_hybrid[kW]_qc` is the more severe of `P_Solar[kW]_qc` and `P_Gaia[kW]_qc` (ties go to `P_Solar`), with
`P_hybrid[kW]_qc_source` (int8) saying which one. Because every `P_Gaia[kW]_qc` is 7 or 10 and 6 outranks both, a row with
substitution carries 6 and every other row carries `P_Gaia`'s 7/10. If a 6 is already present in a file,
it is treated as platform-owned and recomputed (expansion is idempotent). The resampled files carry
`P_Gaia[kW]_qc_worst` instead of `P_Gaia[kW]_qc` (the pipeline never writes a per-second flag column at a coarser resolution), so at 1 min,
5 min and 1 h the expansion reads `P_Gaia[kW]_qc_worst` (the worst code in the bucket, by the same severity order) as the wind flag; a bucket
with no source seconds (NaN) counts as `QC_OK`. `P_hybrid[kW]_qc_source` is an `int8` column:

| value | name (`qc_codes.py`) | meaning |
|---|---|---|
| 0 | `SOURCE_NONE` | the hybrid flag is 0 (`QC_OK`): both constituents are fine, nothing to attribute |
| 1 | `SOURCE_SOLAR` | the hybrid flag was taken from `P_Solar[kW]_qc` (also when both constituents are equally severe) |
| 2 | `SOURCE_WIND` | the hybrid flag was taken from `P_Gaia[kW]_qc` |

The same table is `solete.qc_codes.SOURCE_LABELS`. It is deliberately a different, three-value vocabulary: it is not a QC code and must not be read with `QC_LABELS`.

The platform's own v3 raw-value checks (pressure, humidity, wind direction, azimuth, elevation) apply to
the original v3 files only, and write code 11; they are superseded by this schema for v4.
