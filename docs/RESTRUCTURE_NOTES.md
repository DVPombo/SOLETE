# Restructure notes (merge of `SOLETEdataset` into the platform repository)

Working notes for the maintainer. Safe to delete once the open items below are settled.

## 1. Old -> new locations

| Before | Now |
|---|---|
| `Functions.py` (shim) | removed; import from `solete.*` |
| `solete_pipeline/{io,qc,physics,preprocessing,modeling,postprocess}.py` | `solete/{io,qc,physics,preprocessing,modeling,postprocess}.py` |
| `metrics.py` | `solete/metrics.py` |
| `bench_common.py` | `solete/benchmark/common.py` (`from solete.benchmark import common as bc`) |
| `solete/dataset.py` | unchanged location, imports updated |
| `baseline_persistence.py`, `baseline_climatology_ar.py`, `baseline_gbm.py` | `benchmarks/` (same names) |
| `task5_6_lstm_cnn_harness.py` | `benchmarks/lstm_cnn_harness.py` |
| `task6_2_hybrid_joint_vs_independent.py` | `benchmarks/hybrid_joint_vs_independent.py` |
| `task6_3_ramp_rate_analysis.py` | `benchmarks/ramp_rate_analysis.py` |
| `task7_6_probabilistic_forecast.py` | `benchmarks/probabilistic_forecast.py` |
| `splits/`, `results/`, `BENCHMARKS.md` | `benchmarks/splits/`, `benchmarks/results/`, `benchmarks/BENCHMARKS.md` |
| `RunMe.py`, `MLForecasting.py` | `scripts/quickstart/` |
| `RunMe_matlab.m` | `matlab/RunMe_matlab.m` |
| `SOLETE_short.h5` | `examples/SOLETE_short.h5` |
| `SOLETE_Pombo_60min.h5` (was tracked in git) | `data/hdf5/` (untracked; from figshare) |
| `availability_report.csv` | `docs/legacy/availability_report_v3.csv` |
| `QC_SCHEMA.md`, `DATA_DICTIONARY.md` (platform) | `docs/legacy/*_platform_v3.md` |
| `RESOLUTIONS.md` | `docs/RESOLUTIONS.md` |
| `SOLETEdataset/{pipeline,diagnostics,docs}` | `dataset/{pipeline,diagnostics,docs}` |
| trained models, `Results_*.h5`, training plots (cwd) | `outputs/` |
| `*_Expanded.h5` cache (cwd) | `data/derived/` |

Historical text (CHANGELOG entries, `KNOWN_ISSUES.md` history, the legacy documents) still uses the old names in places; they describe
the past and were left as written, except for mechanical path updates in the active docs.

## 2. RESOLVED: one QC vocabulary, and where model columns come from

**Before.** The platform (`solete/qc.py`) and the dataset pipeline (`dataset/pipeline/qc_flags.py`) used different meanings for the same numbers
(platform: 1 missing, 2 sensor error, 3 implausible, 4 interpolated, 5 gap-affected, 6 model-substituted; pipeline: 1 wrapped, 2 placeholder,
3 dropout fixed, 4 dropout untreated, 5 glitch fixed, 7-10 provenance/recomputed/glitch NaN/active day), and reading a v4 file would have overwritten
the pipeline's flags. `data_version='v4'` therefore raised `NotImplementedError`.

**Decision (maintainer-confirmed).**
- One module, `solete/qc_codes.py`, holds the v4 code set and severity order; `dataset/pipeline/qc_flags.py` and `solete/` import it.
- Code 6 = `QC_MODEL_SUBSTITUTED`, platform-owned; no pipeline rule may emit it (`assert_pipeline_codes`, enforced before writing and by a test).
- `solete/qc.py` is reduced to reading the `<col>_qc` columns present, adding the substitution flag, and a legacy path for v3 files. The v3 files
  have no flags, so the old platform raw-value checks are kept for them (same detectors, same rows) but write a new code **11**
  (`QC_UNTREATED_IMPLAUSIBLE`, "implausible value kept as recorded"), since old 1 and 3 now mean something else. Added by this change; revisit
  if a different mapping is preferred.
- `data_version='v4'` works and never overwrites the file's flags. `P_hybrid[kW]_qc` inherits by the single severity order (on v3 results are
  unchanged because only 0 and 6 occur there).
- Expansion is split: `solete/expansion.py::expand_physical` adds the deterministic, row-wise, horizon-independent columns;
  `ExpandSOLETE` keeps `TempModule_RP` (carries the previous row's state, so not row-wise) and the `Control_Var`/horizon features, and calls it first.
- **Released files never overwrite measured `P_Solar[kW]`.** They add `P_Solar_clean[kW]`; the platform sets its working `P_Solar[kW]` from it on load, which reproduces the old in-memory overwrite exactly.
- **Resolution rule.** Measured columns and pipeline-owned flags are resampled from the cleaned 1-second data. Model-derived columns (incl. code 6) are computed
  at each resolution by `expand_physical` from that resolution's own cleaned inputs, never averaged up. *Reason:* the model is nonlinear (cell temperature, clipping at 0 and
  at the inverter maximum) and the rule `Pac >= 1.5 * P_Solar` is a threshold, so model-of-means differs from mean-of-model and the flag fires on different instants
  at different scales; averaging a 1 s flag or `Pac` upward would present a derived quantity as if it were the hourly model. Quantified in `dataset/docs/METHODOLOGY.md`.

**Decision (maintainer): code 6 requires `Pac > 0` as well.** The old test `Pac >= 1.5 * P_Solar[kW]` is also true for 0 vs 0. On the v3 hourly file all 4,204
flagged rows (38.33 %) were night rows with measured and modelled power both 0; no daytime row was flagged, so the rule never substituted a value there.
The rule is now `Pac >= 1.5 * P_Solar[kW]` and stored `Pac > 0`. `P_Solar_clean[kW]`, `P_hybrid[kW]` and every other column are bit-identical to before; only
`P_Solar[kW]_qc`, `P_hybrid[kW]_qc` and its source code change (4,204 -> 0 rows on the hourly file; 1,272 -> 0 and 358 -> 0 on the two
samples; 5 -> 0 on `SOLETE_short.h5`). **Consequence for the benchmarks:** `metrics.qc_mask` excludes code 6, so `qc_excluded` now equals `qc_included` on the v3
hourly file (before: 751 test rows fewer, n = 2202 instead of 2953). The benchmark results that can run without TensorFlow (persistence, smart persistence, climatology, AR, gradient boosting, probabilistic PV/wind, hybrid
and ramp-rate) were **regenerated** on 2026-10-08 with approval: every number outside the `qc_excluded` blocks is unchanged (checked key by key against the
previous JSON), and `qc_excluded` now equals `qc_included` (n = 2953). `benchmarks/splits/v1.json` was not touched. The LSTM/CNN smoke test needs TensorFlow and was not re-run;
its inputs are bit-identical. `BENCHMARKS.md` shows one line per model (the duplicate `qc_excluded` lines were removed).

**Other redundancies, reviewed.** Code 11 is kept: no other code can mean "implausible value left in place" (2 and 9 mean the value was set to NaN, 1 means corrected, 4 and 7 mean plausible),
it is one code for five columns, and it exists only in v3 frames; it and `legacy_v3_raw_value_rules` can be deleted together when the v3 path is retired. The boolean column `P_Solar_model_substituted`
was identical to `P_Solar[kW]_qc == 6` (P_Solar has no other flag rule, so no more severe code can hide a 6) and has been **removed**; so has the alias
`QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED` and `build_substitution_qc_rule`. `P_hybrid[kW]_qc_source` is now an `int8` code (0 none, 1 P_Solar, 2 P_Gaia) instead of text, always present.

## 3. Other open items

- **Benchmark results are v3.** `benchmarks/results/*.json` and `BENCHMARKS.md` were computed on the v3 hourly file. All of them except the LSTM/CNN smoke test (needs TensorFlow) were regenerated on 2026-10-08 after the `Pac > 0` change. Re-running on the v4 `60min` file is still open (needs that file).
- **Splits.** `benchmarks/splits/v1.json` is defined on the v3 hourly index. Check that the v4 `60min` file has the same boundaries before reusing it.
- **v4 `60min` timestamps** are labelled "start of interval"; confirm the platform's lag/feature code assumes the same convention.
- **Figshare description / `LANDING.txt`** must carry the final repository URL (`github.com/DVPombo/SOLETE` assumed here) and the placeholder `.v4` DOI must be replaced after publishing.
- **Docker**: the Dockerfile was updated but, as before, not built (no Docker in the authoring environment).
- **`examples/` Colab setup cell** clones the `main` branch of `DVPombo/SOLETE`; it only works once this restructure is merged there.
- **Line numbers** quoted in `KNOWN_ISSUES.md` for `Functions.py` were already stale after the earlier split; they are now labelled "formerly".

## 4. What was verified, and how

- `pytest`: 62 passed with the v3 file in `data/hdf5/`; 30 passed / 32 skipped without it; 62 passed with the data in a folder pointed to by `SOLETE_DATA_DIR`.
- Dataset pipeline (clean -> resample -> export_parquet -> a diagnostic) run end-to-end from a different working directory with bare file names, on a **synthetic** 3-hour 1-second file (the real 3.5 GB file was not available here). This verifies path handling, not the cleaning results.
- `baseline_persistence`, `baseline_climatology_ar`: re-run from another folder; outputs identical to the stored results.
- `RunMe.py`, `inspect_dataset.py`, `availability_report.py` run from another folder. The five notebooks execute end to end.
- **Not run**: `MLForecasting.py` and `lstm_cnn_harness.py` (need keras/TensorFlow, not installed here; their imports were checked up to that point), `baseline_gbm`/`probabilistic_forecast`/`hybrid`/`ramp` (import-checked only), the MATLAB script, the Docker build.

- **Code 8 (`Azimuth/Elevation _qc`, constant).** Candidate for removal at the next 1-second pipeline run; see `dataset/docs/QC_SCHEMA.md` §3b. Not done: it changes the pipeline output schema.
- **Real 1-second file:** `expand_physical`, the resampler changes and `scripts/expansion_checks.py effect --input` were only run on synthetic 1 s data.

## 5. The v4 release build (`dataset/pipeline/build_release.py`)

### 5a. Old -> new file names (implemented once, in `solete/paths.py`)

| old / stale name | new |
|---|---|
| `SOLETE_clean_1sec.h5`, `SOLETE_Pombo_1sec_clean.h5` | `SOLETE_Pombo_1sec_v4.h5` (cleaned + expanded) |
| `SOLETE_clean_1min.h5`, `SOLETE_clean_5min.h5` | `SOLETE_Pombo_1min_v4.h5`, `SOLETE_Pombo_5min_v4.h5` |
| `SOLETE_clean_1h.h5`, `SOLETE_resampled_1h.h5` | `SOLETE_Pombo_60min_v4.h5` (`1h` stays only as an input alias of `60min`) |
| `SOLETE_clean_<res>.parquet` | `SOLETE_Pombo_<res>_v4.parquet` |
| *(new)* | `SOLETE_Pombo_1sec_original_v4.h5` / `.parquet`: raw 1 s data, sorted, nine measured columns |
| raw v3 input | unchanged: `SOLETE_Pombo_1sec.h5` (`--raw`) |
| `clean_solete_1sec.py` default `--out-prefix SOLETE_clean_1sec` | `SOLETE_Pombo_1sec_cleaned` (a pre-expansion intermediate; the build keeps it in scratch) |
| `resample_solete.py` default `--out-prefix SOLETE_clean`, rule `1h`, output `<prefix>_<rule>.h5` | `SOLETE_resampled`, rule `60min` (`1h` accepted), intermediates only |

Hits left on purpose: `1h` as a pandas frequency string (diagnostics) and as the platform's forecast `horizon="1h"`; `CHANGELOG.md` history lines; the legacy v3 docs.

### 5b. Rule inventory (one line per rule; where it lives; duplicates flagged)

| stage | rule | lives in |
|---|---|---|
| clean | `WIND_DIR` wrapped `mod 360`, code 1 | `clean_solete_1sec.clean_block` |
| clean | pressure placeholder: exact multiple of 1000 (`pressure_multiples_of_1000`) or flatline run >= 300 (`qc_flags.detect_flatline_mask`) -> NaN, code 2 | `apply_pressure_sentinels` |
| clean | WS/HUM joint dropout (`dropout_bad_mask`): run <= 5 s interpolated (3), longer kept (4) | `apply_dropout_fix` + `qc_flags.fix_short_runs` |
| clean | glitch bounds per column (`BOUNDS`, `glitch_bad_mask`): run <= 3 s interpolated (5), longer NaN (9) | `apply_glitch_fix` + `fix_short_runs` |
| flag | `P_Gaia` day flag: active days 10, others 7, values untouched | `flag_p_gaia` |
| clean | Azimuth/Elevation recomputed (pvlib, UTC, 0 = south), no flag column (D1) | `solar_position.compute_solar_position_bulk` |
| resample | mean; circular mean for `WIND_DIR` and `Azimuth`; flags -> `_qc_worst`, `_qc_frac_flagged`; model columns dropped, foreign codes -> 0 | `resample_solete.resample_dataframe` |
| expand | `Pac`, `Pdc`, `TempModule`, `TempCell`, `P_Solar[kW]_qc` (6 if `Pac >= 1.5 P_Solar` and `Pac > 0`), `P_Solar_clean`, `P_hybrid`, `P_hybrid_qc`, `_qc_source` | `solete.expansion.expand_physical` (flag helper `solete.qc.add_substitution_flag`) |
| export | index -> `timestamp` (ns, UTC), metadata, round trip | `export_parquet.py`, `release_meta.py` |
| slicing | safe cut = no dropout/out-of-bound row and no identical non-sentinel pressure pair across the cut | `clean_solete_1sec.safe_boundaries`, built from the rule masks above |

Rules that exist twice, and why they were left:
1. **v3 raw-value flags** (`solete.qc.legacy_v3_raw_value_rules`, code 11) re-implement, differently, the pressure / humidity / wind-direction detections of the pipeline. Intentional: they serve the v3 hourly file that has no flags of its own, and are deleted together with code 11 when the v3 path is retired. Not touched.
2. `qc_flags.detect_round_sentinels` (an auditing heuristic that finds candidate sentinels by frequency) overlaps in purpose with the multiple-of-1000 rule. It is used by diagnostics only; the build uses `pressure_multiples_of_1000`. Left.
3. `resample_solete.circular_mean_deg` is a dead stub that raises `NotImplementedError` (the real work is `resample_angular_column`). Left to avoid changing a public name.
4. Before this change the multiple-of-1000 test and the dropout/glitch masks were written inline in the rule functions; the cut finder needs the same masks, so each now has ONE function used by both the rule and the cut finder (no copy).

### 5c. What the build adds beyond the prompt's list (so you can veto it)
- `solete/h5io.py`, `solete/params.py` (`import_PV_WT_data` moved out of `solete/io.py`, which still re-exports it), lazy CoolProp import in `solete/physics.py`.
- Azimuth in the 1 min / 5 min / 60 min files is now a **circular** mean (south-referenced, [-180, 180)); a plain mean was wrong in the bucket containing solar midnight. Elevation is unchanged (plain mean).
- Column order of the cleaned frame is fixed: input measured columns, `Azimuth[deg]`, `Elevation[deg]`, 8 `_qc` columns (before: Azimuth/Elevation kept their input position if present).
- `_qc_worst` columns are `int64` when a file has no empty bucket and `float64` (NaN) when it has one (unchanged pandas behaviour; the real file has no gaps).

