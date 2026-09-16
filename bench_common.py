# -*- coding: utf-8 -*-
"""
bench_common.py -- SOLETE Phase 5 continuation (Tasks 5.3-5.8)

Shared data-loading / split / scoring harness so every baseline in 5.3-5.6
goes through exactly one loading path and exactly one scoring path
(metrics.py), per the continuation prompt's instruction not to write a
parallel scoring path per task.

Canonical loading recipe is copied verbatim from splits/README.md
("Reproducing this split").
"""

import json
import os

import numpy as np
import pandas as pd

import metrics as M
from Functions import import_SOLETE_data, import_PV_WT_data

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(REPO_ROOT, "splits", "v1.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

TARGETS = {
    "pv_power": "P_Solar[kW]",
    "wind_power": "P_Gaia[kW]",
}
QC_COLUMNS = {
    # Only PV has a QC column that carries the model-substitution flag
    # (Task 5.1.3). Wind (P_Gaia[kW]) has no per-row QC column in this
    # dataset, so there is nothing to exclude/include there -- see
    # Functions.import_SOLETE_data's printed QC flag set.
    "pv_power": "P_Solar[kW]_qc",
}

WIND_CAVEAT = (
    "CAVEAT: P_Gaia[kW] is exactly 0.0 for 99.56% of the full record "
    "(only 48 of 10,969 rows, on two isolated calendar days, are non-zero). "
    "A wind-power score on this dataset is overwhelmingly a score on an "
    "all-zero target; near-perfect wind metrics reflect that, not "
    "forecasting skill. See splits/README.md."
)

_df_cache = None


def load_full_df():
    """Load the full, sorted SOLETE_Pombo_60min.h5 via the canonical recipe
    in splits/README.md. Cached in-process since this reruns the Build
    pipeline (King's PV model, QC flagging) each call otherwise."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache.copy()

    Control_Var = {
        "resolution": "60min",
        "SOLETE_builvsimport": "Build",
        "SOLETE_save": False,
        "OriginalFeatures": [],
        "PossibleFeatures": [],
    }
    PVinfo, WTinfo = import_PV_WT_data()
    df = import_SOLETE_data(Control_Var, PVinfo, WTinfo).sort_index()
    _df_cache = df
    return df.copy()


def load_split_boundaries():
    with open(SPLITS_PATH) as f:
        spec = json.load(f)
    b = spec["blocks"]
    return {
        "train": (pd.Timestamp(b["train"]["start"]), pd.Timestamp(b["train"]["end"])),
        "val": (pd.Timestamp(b["val"]["start"]), pd.Timestamp(b["val"]["end"])),
        "test": (pd.Timestamp(b["test"]["start"]), pd.Timestamp(b["test"]["end"])),
    }


def split_df(df):
    """Return (train, val, test) DataFrames sliced per splits/v1.json."""
    bounds = load_split_boundaries()
    out = {}
    for name, (start, end) in bounds.items():
        out[name] = df[(df.index >= start) & (df.index <= end)]
    return out["train"], out["val"], out["test"]


def score_block(y_true_full, y_pred_full, block_index, qc_col_full=None, exclude_flags=(6,)):
    """
    Score a forecast restricted to `block_index` (e.g. the test block's
    DatetimeIndex), once with and once without the default QC exclusion,
    per Task 5.3's requirement to show both settings.

    y_true_full / y_pred_full : pandas Series aligned on the FULL df's index
        (so lookback/lag construction can freely reach outside the block).
    qc_col_full : pandas Series of the `<col>_qc` flags aligned on the full
        df's index, or None if this target has no QC column (wind).

    Returns dict: {"qc_included": {...}, "qc_excluded": {...} or None}
    """
    y_true = y_true_full.loc[block_index]
    y_pred = y_pred_full.loc[block_index]

    def _one(mask):
        return {
            "n": int(mask.sum()) if mask is not None else int(len(y_true)),
            "mae": M.mae(y_true, y_pred, mask=mask),
            "rmse": M.rmse(y_true, y_pred, mask=mask),
        }

    included = _one(None)
    excluded = None
    if qc_col_full is not None:
        qc_block = qc_col_full.loc[block_index]
        mask = M.qc_mask(qc_block, exclude_flags=exclude_flags)
        if mask.sum() > 0:
            excluded = _one(mask)

    return {"qc_included": included, "qc_excluded": excluded}


def add_nrmse(score_dict, y_true_full, y_pred_full, block_index, qc_col_full, capacity, exclude_flags=(6,)):
    """Add nrmse (capacity + mean) to an already-computed score_dict from score_block,
    for both the qc_included and qc_excluded rows."""
    y_true = y_true_full.loc[block_index]
    y_pred = y_pred_full.loc[block_index]

    if score_dict["qc_included"] is not None:
        score_dict["qc_included"]["nrmse_capacity"] = M.nrmse(y_true, y_pred, capacity=capacity, method="capacity")
        score_dict["qc_included"]["nrmse_mean"] = M.nrmse(y_true, y_pred, method="mean")

    if score_dict["qc_excluded"] is not None and qc_col_full is not None:
        qc_block = qc_col_full.loc[block_index]
        mask = M.qc_mask(qc_block, exclude_flags=exclude_flags)
        score_dict["qc_excluded"]["nrmse_capacity"] = M.nrmse(y_true, y_pred, capacity=capacity, method="capacity", mask=mask)
        score_dict["qc_excluded"]["nrmse_mean"] = M.nrmse(y_true, y_pred, method="mean", mask=mask)

    return score_dict


def score_target_on_test(df_full, target_key, y_pred_full, test_index):
    """Convenience wrapper: score one target's predictions on the test block,
    with both QC settings and both nRMSE normalizations, matching the
    reporting shape every Task 5.3-5.6 baseline uses."""
    col = TARGETS[target_key]
    qc_col = df_full[QC_COLUMNS[target_key]] if target_key in QC_COLUMNS else None
    capacity = M.installed_capacity_kw(target_key)

    result = score_block(df_full[col], y_pred_full, test_index, qc_col_full=qc_col)
    result = add_nrmse(result, df_full[col], y_pred_full, test_index, qc_col, capacity)
    return result


def save_result(model_name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, model_name + ".json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_result(model_name):
    path = os.path.join(RESULTS_DIR, model_name + ".json")
    with open(path) as f:
        return json.load(f)
