# -*- coding: utf-8 -*-
"""
solete/dataset.py -- SOLETE Phase 5 continuation, Task 5.8

Minimal `SOLETE` wrapper class exposing:

    dataset = SOLETE(resolution="60min", target="pv_power", split="v1")
    X_train, y_train, X_test, y_test = dataset.forecasting(horizon="1h")

    # resolution, horizon and data_version all vary independently, e.g.:
    dataset = SOLETE(resolution="5min", data_version="v4")   # v4-only install
    X_train, y_train, X_test, y_test = dataset.forecasting(horizon="1h")  # == 12 steps at 5min

Wraps:
    - benchmarks/splits/v1.json's train/val/test boundaries (via solete/benchmark/common.py,
      which itself just reads the JSON -- no boundary is hardcoded twice)
    - metrics.py's qc_mask() exclusion logic (via the `qc_exclude` flag)
    - solete.io.import_SOLETE_data's real HDF5 loading (via
      bench_common.load_full_df(), the same canonical recipe used by every
      other Phase 5 script -- not a second/parallel loading path)

Kept additive: does not change import_SOLETE_data's own behaviour, and does
not touch scripts/quickstart/MLForecasting.py / solete/.

Scope (documented rather than silently assumed):
    - resolution: "1sec", "1min", "5min" or "60min" (anything solete.paths
      knows about -- see solete.paths.RESOLUTIONS/normalize_resolution).
      "1sec" loads the full 1-second file, which is large and slow (a few
      minutes); prefer 5min/60min unless second-resolution really is needed.
    - data_version: "v3" (default, matching every existing caller -- the
      legacy figshare files the benchmarks were built against) or "v4" (the
      current figshare release -- solete/paths.py). An installation that only
      downloaded the v4 release needs data_version="v4" explicitly; v3 and v4
      files use different on-disk names (solete.paths.data_filename) and
      neither is auto-detected, to avoid silently mixing file versions.
    - split: "v1" only (the only split version that exists right now --
      v1.json's boundaries are plain calendar timestamps, so they slice
      cleanly at every resolution above; what's still genuinely missing is
      a *second split version*, e.g. a different train/val/test calendar
      carve-up, which is a maintainer data decision -- ASK FIRST -- not a
      parameter this class can safely default its way into).
    - horizon: an int number of native steps at the chosen resolution, or a
      pandas.Timedelta-parseable string (e.g. "1h", "30min", "1D") that is an
      exact multiple of that resolution's step (e.g. "1h" at 5min resolution
      = 12 steps). A horizon that doesn't divide evenly raises ValueError
      rather than silently rounding.
    - Feature set for `forecasting()` matches Task 5.5's gradient-boosting
      baseline (lagged target + same-timestamp weather + calendar), with
      `lag_<horizon_steps>` deliberately kept as its own named column: at the
      default horizon ("1h" at 60min resolution, i.e. 1 step) it is
      `lag_1`, which doubles as the Task 5.3 plain-persistence prediction --
      tests/test_dataset_api.py's exact-reproduction round-trip check uses
      it for exactly that, and still passes unchanged (horizon_steps=1
      reproduces the original fixed LAGS=(1,2,3,24) exactly).
"""

import numpy as np
import pandas as pd

from solete.benchmark import common as bc
from solete import metrics as M
from solete.paths import RESOLUTIONS, normalize_resolution

SUPPORTED_RESOLUTIONS = RESOLUTIONS  # ("1sec", "1min", "5min", "60min")
SUPPORTED_SPLITS = ("v1",)

# Seconds per native sample at each resolution (same values as solete/synthetic.py's _SPS).
_STEP_SECONDS = {"1sec": 1, "1min": 60, "5min": 300, "60min": 3600}
# A horizon of 1 step reproduces these exactly (the original, fixed LAGS=(1,2,3,24) at 60min):
# a few of the most recent native-resolution samples, plus "this time yesterday" (a seasonal/
# daily lag, scaled to each resolution's own step count rather than left as a fixed "24").
_RECENT_LAG_STEPS = (1, 2, 3)
_DAILY_LAG_SECONDS = 24 * 3600

MET_COLS = ["GHI[kW1m2]", "POA Irr[kW1m2]", "WIND_SPEED[m1s]", "TEMPERATURE[degC]"]


def _horizon_to_steps(horizon, step_seconds):
    """Translate a `forecasting(horizon=...)` argument into a native-step count at the
    dataset's resolution. Accepts an int (steps directly) or anything
    pandas.Timedelta understands (e.g. "1h", "30min", "1D"); the latter must be an
    exact multiple of `step_seconds` -- raises rather than silently rounding."""
    if isinstance(horizon, (int, np.integer)) and not isinstance(horizon, bool):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1 step, got {horizon!r}")
        return int(horizon)
    try:
        total_seconds = pd.Timedelta(horizon).total_seconds()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"horizon={horizon!r} is neither an int (native steps) nor a "
            "pandas.Timedelta-parseable string (e.g. '1h', '30min', '1D')."
        ) from exc
    steps, remainder = divmod(round(total_seconds, 6), step_seconds)
    if remainder != 0 or steps < 1:
        raise ValueError(
            f"horizon={horizon!r} ({total_seconds:.0f}s) is not a whole, positive "
            f"multiple of this dataset's {step_seconds}s step -- pass an exact "
            "multiple (e.g. horizon=12 for '1h' at 5min resolution) or an int step "
            "count directly."
        )
    return int(steps)


class SOLETE:
    def __init__(self, resolution="60min", target="pv_power", split="v1", qc_exclude=True, data_version="v3"):
        resolution = normalize_resolution(resolution)
        if target not in bc.TARGETS:
            raise ValueError(f"target must be one of {list(bc.TARGETS)}, got {target!r}")
        if split not in SUPPORTED_SPLITS:
            raise NotImplementedError(f"split={split!r} not supported yet -- only {SUPPORTED_SPLITS}")
        if data_version not in ("v3", "v4"):
            raise ValueError(f"data_version must be 'v3' or 'v4', got {data_version!r}")

        self.resolution = resolution
        self.target = target
        self.target_col = bc.TARGETS[target]
        self.split = split
        self.qc_exclude = qc_exclude
        self.data_version = data_version

        self._df = bc.load_full_df(resolution, data_version)
        self._train, self._val, self._test = bc.split_df(self._df)

    def _qc_keep_mask(self, index):
        if not self.qc_exclude or self.target not in bc.QC_COLUMNS:
            return None
        qc_col = self._df[bc.QC_COLUMNS[self.target]].loc[index]
        return M.qc_mask(qc_col)

    def _build_features(self, lags):
        df = self._df
        col = self.target_col
        feats = {f"lag_{l}": df[col].shift(l) for l in lags}
        for m in MET_COLS:
            feats[m] = df[m]
        idx = df.index
        hour = idx.hour.values
        doy = idx.dayofyear.values
        feats["hour"] = hour
        feats["month"] = idx.month.values
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        return pd.DataFrame(feats, index=idx)

    def forecasting(self, horizon="1h", include_val=False):
        """
        Returns (X_train, y_train, X_test, y_test) -- or, with
        include_val=True, (X_train, y_train, X_val, y_val, X_test, y_test).

        `horizon` is an int step count at this dataset's resolution, or a
        pandas.Timedelta-parseable string ("1h", "30min", "1D", ...) that is an
        exact multiple of the resolution's native step (see
        `_horizon_to_steps`). It shifts every lag feature back by
        `horizon_steps - 1` extra steps relative to the horizon=1 case, so a
        longer horizon means less recent autoregressive history is available
        -- same-timestamp weather/calendar features (an implicit "forecast
        valid for this timestamp" assumption) are unaffected.

        Each X is a DataFrame indexed by timestamp (the row's target
        timestamp, i.e. the same timestamp as the corresponding y). Rows
        with any NaN feature (from lag lookback at the very start of the
        record) are dropped. If qc_exclude=True (default) and this target
        has a QC column (currently only pv_power), rows flagged
        model-substituted (qc==6, per metrics.qc_mask()'s default) are
        dropped from every block.
        """
        step_seconds = _STEP_SECONDS[self.resolution]
        horizon_steps = _horizon_to_steps(horizon, step_seconds)
        daily_lag_steps, remainder = divmod(_DAILY_LAG_SECONDS, step_seconds)
        assert remainder == 0, f"{self.resolution} step does not divide a day evenly"
        lags = tuple(horizon_steps - 1 + l for l in (*_RECENT_LAG_STEPS, daily_lag_steps))

        X_full = self._build_features(lags).dropna()
        y_full = self._df[self.target_col]

        def _block(block_df):
            idx = block_df.index.intersection(X_full.index)
            keep = self._qc_keep_mask(idx)
            if keep is not None:
                idx = idx[keep.to_numpy()]
            return X_full.loc[idx], y_full.loc[idx]

        X_train, y_train = _block(self._train)
        X_val, y_val = _block(self._val)
        X_test, y_test = _block(self._test)

        if include_val:
            return X_train, y_train, X_val, y_val, X_test, y_test
        return X_train, y_train, X_test, y_test

    def installed_capacity_kw(self):
        return M.installed_capacity_kw(self.target)
