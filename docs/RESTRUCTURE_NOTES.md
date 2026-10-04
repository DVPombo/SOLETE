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

## 2. Open item: the two QC code sets (the one real design decision)

The platform (`solete/qc.py`) and the dataset pipeline (`dataset/pipeline/qc_flags.py`) use **different meanings for the same numbers**:

| code | platform (`solete/qc.py`) | dataset v4 (`qc_flags.py`) |
|---|---|---|
| 0 | valid | ok |
| 1 | missing | wrapped (WIND_DIR) |
| 2 | sensor error (reserved) | placeholder sentinel -> NaN |
| 3 | physically implausible | dropout, short, fixed |
| 4 | interpolated (reserved) | dropout, long, untreated |
| 5 | aggregation affected by gaps (reserved) | glitch, short, fixed |
| 6 | suspected curtailment / model-substituted | (unused) |
| 7-10 | - | unverified provenance / recomputed / glitch long NaN / active day |

The v4 files already contain `<column>_qc` columns. `apply_qc_flags()` in the platform writes columns with the same names, so reading a v4
file through `import_SOLETE_data` would silently overwrite the dataset's flags with platform codes. That is why `data_version='v4'`
currently raises `NotImplementedError`. Suggested resolution: keep the v4 code set as the only vocabulary (one module, e.g. `solete/qc_codes.py`,
imported by both halves), give "model-substituted" a new v4 code (code 6 is free), and reduce the platform's `qc.py` to
(a) reading the `_qc` columns that are present and (b) adding only the substitution flag. `metrics.qc_mask()` and `benchmarks/` assume
code 6 for substitution, so they need no change if 6 is kept for that meaning.

## 3. Other open items

- **Benchmark results are v3.** `benchmarks/results/*.json` and `BENCHMARKS.md` were computed on the v3 hourly file. The persistence, climatology and AR
  results were regenerated after the move and their parsed JSON content is identical to the stored files; the others were not re-run. They must be re-run on v4 once item 2 is done.
- **Splits.** `benchmarks/splits/v1.json` is defined on the v3 hourly index. Check that the v4 `1h` file has the same boundaries before reusing it.
- **v4 `1h` timestamps** are labelled "start of interval"; confirm the platform's lag/feature code assumes the same convention.
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
