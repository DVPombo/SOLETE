# -*- coding: utf-8 -*-
"""
solete/dataset.py -- SOLETE Phase 5 continuation, Task 5.8

Minimal `SOLETE` wrapper class exposing:

    dataset = SOLETE(resolution="60min", target="pv_power", split="v1")
    X_train, y_train, X_test, y_test = dataset.forecasting(horizon="1h")

Wraps:
    - splits/v1.json's train/val/test boundaries (via bench_common.py,
      which itself just reads the JSON -- no boundary is hardcoded twice)
    - metrics.py's qc_mask() exclusion logic (via the `qc_exclude` flag)
    - Functions.import_SOLETE_data's real HDF5 loading (via
      bench_common.load_full_df(), the same canonical recipe used by every
      other Phase 5 script -- not a second/parallel loading path)

Kept additive: does not change import_SOLETE_data's own behaviour, and does
not touch MLForecasting.py / Functions.py.

Scope of this first version (documented rather than silently assumed):
    - resolution: "60min" only (the only resolution splits/v1.json covers).
    - split: "v1" only (the only split version that exists right now).
    - horizon: "1h" only (1 step ahead at 60min resolution -- matches the
      horizon used throughout Tasks 5.3-5.6 for comparability). Other
      values raise NotImplementedError rather than silently doing the
      wrong thing.
    - Feature set for `forecasting()` matches Task 5.5's gradient-boosting
      baseline (lagged target + same-timestamp weather + calendar), with
      `lag_1` deliberately kept as its own named column: it doubles as the
      Task 5.3 plain-persistence prediction, which is what
      tests/test_dataset_api.py's exact-reproduction round-trip check
      uses it for.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_common as bc
import metrics as M

SUPPORTED_RESOLUTIONS = ("60min",)
SUPPORTED_SPLITS = ("v1",)
SUPPORTED_HORIZONS = ("1h",)

LAGS = (1, 2, 3, 24)
MET_COLS = ["GHI[kW1m2]", "POA Irr[kW1m2]", "WIND_SPEED[m1s]", "TEMPERATURE[degC]"]


class SOLETE:
    def __init__(self, resolution="60min", target="pv_power", split="v1", qc_exclude=True):
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise NotImplementedError(
                f"resolution={resolution!r} not supported yet -- only {SUPPORTED_RESOLUTIONS} "
                "(splits/v1.json only covers 60min). Extending this requires a new splits/ "
                "version for that resolution first -- ASK FIRST before adding one."
            )
        if target not in bc.TARGETS:
            raise ValueError(f"target must be one of {list(bc.TARGETS)}, got {target!r}")
        if split not in SUPPORTED_SPLITS:
            raise NotImplementedError(f"split={split!r} not supported yet -- only {SUPPORTED_SPLITS}")

        self.resolution = resolution
        self.target = target
        self.target_col = bc.TARGETS[target]
        self.split = split
        self.qc_exclude = qc_exclude

        self._df = bc.load_full_df()
        self._train, self._val, self._test = bc.split_df(self._df)

    def _qc_keep_mask(self, index):
        if not self.qc_exclude or self.target not in bc.QC_COLUMNS:
            return None
        qc_col = self._df[bc.QC_COLUMNS[self.target]].loc[index]
        return M.qc_mask(qc_col)

    def _build_features(self):
        df = self._df
        col = self.target_col
        feats = {f"lag_{l}": df[col].shift(l) for l in LAGS}
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

        Each X is a DataFrame indexed by timestamp (the row's target
        timestamp, i.e. the same timestamp as the corresponding y). Rows
        with any NaN feature (from lag lookback at the very start of the
        record) are dropped. If qc_exclude=True (default) and this target
        has a QC column (currently only pv_power), rows flagged
        model-substituted (qc==6, per metrics.qc_mask()'s default) are
        dropped from every block.
        """
        if horizon not in SUPPORTED_HORIZONS:
            raise NotImplementedError(f"horizon={horizon!r} not supported yet -- only {SUPPORTED_HORIZONS}")

        X_full = self._build_features().dropna()
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
