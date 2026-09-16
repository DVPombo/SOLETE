# SOLETE benchmark leaderboard

**Split:** `splits/v1.json` ("v1") — chronological, non-overlapping train / val / test
blocks over `SOLETE_Pombo_60min.h5`. Test block: **2019-05-01 00:00 → 2019-09-01 00:00**,
2,953 rows. See `splits/README.md` for the full boundary rationale and reproduction
recipe.

**Horizon:** 1 step ahead (1h) for every row in every table below.

**Metrics:** MAE and RMSE (target units, kW) from `metrics.py`, plus nRMSE under both
supported normalizations — `nrmse(method="capacity")` (**headline**, fixed installed-
capacity denominator) and `nrmse(method="mean")` (secondary, mean-of-target denominator).
Per the Task 5.2 decision, no bare "nRMSE" is presented anywhere below without saying
which normalization it is.

**QC setting:** every row below is run under **both** QC settings (`qc_included` = all
test rows scored as-is; `qc_excluded` = rows with `P_Solar[kW]_qc == 6`,
model-substituted PV readings, removed before scoring — wind has no QC column to exclude,
see caveat below). **`qc_included` is shown as the headline row per model below, with
`qc_excluded` given as a second line**, so a reader can see both without the table being
read as endorsing only one setting.

All numbers below are copied as-is from the `results/*.json` files produced in the prior
session (`persistence.json`, `smart_persistence.json`, `climatology.json`, `ar.json`,
`gradient_boosting.json`, `lstm_cnn_smoke_test.json`) — **nothing here was recomputed or
re-derived by hand.**

---

## PV (`P_Solar[kW]`)

| Model | QC setting | n | MAE (kW) | RMSE (kW) | nRMSE (capacity) | nRMSE (mean) |
|---|---|---:|---:|---:|---:|---:|
| Persistence `ŷ(t)=y(t-1)` | qc_included | 2,953 | 0.434 | 0.699 | 0.0939 | 0.539 |
| Persistence `ŷ(t)=y(t-1)` | qc_excluded | 2,202 | 0.581 | 0.809 | 0.1087 | 0.465 |
| Smart persistence `ŷ(t)=y(t-24)` | qc_included | 2,953 | 0.562 | 1.132 | 0.1522 | 0.873 |
| Smart persistence `ŷ(t)=y(t-24)` | qc_excluded | 2,202 | 0.754 | 1.311 | 0.1762 | 0.754 |
| Climatology (hour-of-day mean) | qc_included | 2,953 | 0.629 | 1.046 | 0.1406 | 0.807 |
| Climatology (hour-of-day mean) | qc_excluded | 2,202 | 0.843 | 1.211 | 0.1628 | 0.697 |
| AR(p=48), lagged target only | qc_included | 2,953 | 0.291 | 0.524 | 0.0704 | 0.404 |
| AR(p=48), lagged target only | qc_excluded | 2,202 | 0.382 | 0.606 | 0.0815 | 0.349 |
| Gradient boosting (LightGBM) †same-timestamp weather† | qc_included | 2,953 | 0.0113 | 0.0254 | 0.0034 | 0.0196 |
| Gradient boosting (LightGBM) †same-timestamp weather† | qc_excluded | 2,202 | 0.0150 | 0.0294 | 0.0039 | 0.0169 |
| LSTM ‡reduced-epoch, not final‡ | qc_included | 2,953 | 0.277 | 0.539 | 0.0725 | 0.416 |
| CNN ‡own unmodified default epochs‡ | qc_included | 2,953 | 0.356 | 0.567 | 0.0762 | 0.437 |
| CNN-LSTM ‡reduced-epoch, not final‡ | qc_included | 2,953 | 0.288 | 0.550 | 0.0739 | 0.424 |

**† Gradient-boosting caveat (must travel with this row):** the GBM feature set uses
**same-timestamp** meteorological readings (GHI, POA irradiance, wind speed,
temperature) as features, not weather forecast values or lagged-only inputs. This treats
the *measured* weather at time *t* as a stand-in for a perfect short-horizon weather
forecast — a simplifying assumption flagged in `baseline_gbm.py`'s own docstring. **The
GBM PV row is not a fair apples-to-apples comparison against the persistence /
smart-persistence / climatology / AR / LSTM-CNN rows above and below it**, all of which
see only lagged target (and, for LSTM/CNN, a lag window) — not same-timestamp weather.
Its very low RMSE (0.025 kW, qc_included) reflects that near-determinism, not superior
forecasting skill under a comparable information set.

**‡ LSTM / CNN / CNN-LSTM caveat (must travel with these rows):** the sandbox this ran in
had a hard ~300-second wall-clock limit per command, which made `MLForecasting.py`'s own
original `epo_num=1000` (LSTM, CNN_LSTM) infeasible. What is reported above is **LSTM and
CNN_LSTM trained for 20 of their original 1000 epochs**; **CNN** ran at its own original,
already-small default of **3 epochs** (unchanged, not reduced). Loss curves plateau at 20
epochs in the run log, but this is explicitly **not** the full original run — see
`results/lstm_cnn_smoke_test.json`'s own `"status"` field. Treat these three rows as
indicative, not final, until re-run at (near-)full epochs. Only the `qc_included` setting
is available for these rows — the harness did not separately score a `qc_excluded` pass.

---

## Wind (`P_Gaia[kW]`)

> **Wind caveat (Task 5.1, repeated here directly above the table, per instruction):**
> `P_Gaia[kW]` is exactly `0.0` for 10,921 of 10,969 rows (99.56%) in the full record —
> only 48 rows total, on two isolated calendar days (2018-08-31 and 2019-05-25), are
> non-zero, despite thousands of hours elsewhere with wind speed above the turbine's
> 3.5 m/s cut-in. Most plausibly the turbine was out of service for nearly the entire
> 15-month record (unconfirmed). **A wind-power score on this dataset is overwhelmingly a
> score on an all-zero target.** Near-perfect wind metrics below reflect that, not
> forecasting skill, and should not be read as such. See `splits/README.md` for the full
> discussion. There is also **no QC column for `P_Gaia[kW]`** in this dataset (the weather
> sensor QC columns don't cover wind power output itself), so every wind row below has
> only one QC setting (`qc_excluded` is not applicable / `null` in the source JSON).

| Model | n | MAE (kW) | RMSE (kW) | nRMSE (capacity) | nRMSE (mean) |
|---|---:|---:|---:|---:|---:|
| Persistence `ŷ(t)=y(t-1)` | 2,953 | 0.0123 | 0.224 | 0.0203 | 4.305 |
| Smart persistence `ŷ(t)=y(t-24)` | 2,953 | 0.1040 | 0.866 | 0.0787 | 16.644 |
| Climatology (hour-of-day mean) | 2,953 | 0.0553 | 0.612 | 0.0556 | 11.765 |
| AR(p=48), lagged target only | 2,953 | 0.0334 | 0.298 | 0.0271 | 5.733 |
| Gradient boosting (LightGBM) †same-timestamp weather† | 2,953 | 0.0476 | 0.518 | 0.0470 | 9.953 |
| LSTM ‡reduced-epoch, not final‡ | 2,953 | 0.0454 | 0.523 | 0.0476 | 10.063 |
| CNN ‡own unmodified default epochs‡ | 2,953 | 0.0497 | 0.571 | 0.0519 | 10.978 |
| CNN-LSTM ‡reduced-epoch, not final‡ | 2,953 | 0.0533 | 0.612 | 0.0556 | 11.767 |

Note the very large `nRMSE (mean)` values on wind (up to ~16.6): this is a direct symptom
of the caveat above — the mean of an almost-all-zero target is tiny, so dividing by it
blows up the ratio. `nRMSE (capacity)` is the more usable of the two normalizations for
wind for exactly this reason, though neither should be read as a meaningful skill score
given how degenerate the target is on this split.

†/‡ carry the same meanings as in the PV table above (same-timestamp-weather GBM caveat;
reduced/default-epoch LSTM-CNN-family caveat) and apply identically here.

---

## Findings worth keeping as-is (not bugs)

- **Plain persistence beats smart (seasonal) persistence on PV**, qc_included (RMSE 0.699
  vs. 1.132) — a real result for this dataset/split, not an error.
- **Climatology is worse than plain persistence on PV** (RMSE 1.046 vs. 0.699,
  qc_included).
- **AR(p=48) beats both persistence baselines on PV** (RMSE 0.524, qc_included) — the
  best lag-only PV result on this leaderboard.
- **Gradient boosting's wind result is worse than persistence/AR on wind** (RMSE 0.518 vs.
  0.224 / 0.298) — despite GBM's very strong (same-timestamp-weather-assisted) PV result,
  it does not generalize to an advantage on the near-degenerate wind target.

## Historical / pre-corrigendum numbers

None included. This repository's `CHANGELOG.md` documents a v3.0 corrigendum
("Train/validation/test data-splitting bug… affected all results computed with v2.3 or
earlier," plus a related RMSE-calculation bug fixed in the same release) — any numbers
from v2.3 or earlier releases or the original SOLETE papers would need to be clearly
separated and labeled pre-corrigendum/non-comparable per that note, and none were
available to this session to include on that basis. All numbers above are v1-split,
current-codebase results only.
