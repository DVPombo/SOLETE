# -*- coding: utf-8 -*-
"""
Part of the solete_pipeline package -- split out of the original
solete/ (Phase 7, Session 8) for independent testability.
See solete/ (kept as a re-export shim) and CONTRIBUTING.md for
why this split happened and how the modules relate to each other.

The standalone "serving" path: given a model bundle loaded with
solete.modeling.load_model_bundle() and a DataFrame already expanded the same
way training data is (solete.preprocessing.ExpandSOLETE), build the model's
expected input tensor and score it -- without running PreProcessDataset's
train/val/test split or refitting a new scaler. See
scripts/quickstart/predict_with_saved_model.py for a worked example.

PreProcessDataset (solete/preprocessing.py) remains the only place a model is
trained; build_inference_features() below intentionally duplicates only its
X-side windowing (via the same series_to_forecast() helper) rather than being
threaded into it, so this stays an additive, low-risk sidecar instead of a
refactor of that already-delicate function. If PreProcessDataset's X-side
windowing/reshape ever changes, check here too.

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).
"""

import numpy as np
import pandas as pd

from .preprocessing import series_to_forecast

_ANN_TYPES = ('LSTM', 'CNN', 'CNN_LSTM')


def build_inference_features(data, bundle):
    """
    Builds the model's expected input tensor from new/held-out rows, applying the bundle's
    already-fitted X scaler (transform only, never fit).

    Parameters
    ----------
    data : DataFrame
        Must contain every column in bundle['possible_features'], already expanded the same way
        training data was (e.g. via solete.preprocessing.ExpandSOLETE). Ordered by time, one row
        per sample at the model's native resolution.
    bundle : dict
        As returned by solete.modeling.load_model_bundle().

    Returns
    -------
    X : ndarray
        Shaped (samples, PRE+1, n_features) for LSTM/CNN/CNN_LSTM, or (samples, n_features*(PRE+1))
        for RF/SVM -- exactly what ``bundle['model'].predict(X)`` expects.
    timestamps : Series
        The row index of `data` each sample in X corresponds to (the "t" / most-recent-known-value
        timestamp), so forecasts can be re-aligned to time. Rows are dropped (and counted in a
        printed message) when their PRE-length lookback window isn't fully available: the first PRE
        rows of `data` always, plus any later row whose window happens to contain a NaN from a gap
        in the raw data (same completeness PreProcessDataset's own dropna() requires for training).
    """
    additions = bundle['possible_features']
    pre = bundle['PRE']
    mltype = bundle['mltype']

    missing = [col for col in additions if col not in data.columns]
    if missing:
        raise KeyError(
            "build_inference_features: data is missing columns this model was trained on: "
            + ", ".join(missing))

    timestamps = pd.Series(data.index).reset_index(drop=True)
    X = data[additions].reset_index(drop=True)

    X_dict = {}
    for col in additions:
        shifted = series_to_forecast(X[col], pre, 0, dropnan=False)  # columns: col_(t-pre)...col_(t)
        if mltype in _ANN_TYPES:
            #each column is its own single-feature (samples, pre+1, 1) tensor here; concatenated
            #below along the feature axis, same order as PreProcessDataset's ANN branch builds it
            X_dict[col] = np.ravel(shifted).reshape(len(X), pre + 1, 1)
        else:
            X_dict[col] = shifted

    if mltype in _ANN_TYPES:
        X_arr = np.concatenate([X_dict[col] for col in additions], axis=2)  # (samples, pre+1, n_feat)
    else:
        X_arr = pd.concat([X_dict[col] for col in additions], axis=1)  # DataFrame, same xcols/order
        #as PreProcessDataset's RF/SVM branch, so the fitted scaler's feature names line up

    #the first `pre` rows never get a full lookback window (series_to_forecast left them NaN)
    timestamps = timestamps.iloc[pre:].reset_index(drop=True)
    if mltype in _ANN_TYPES:
        X_arr = X_arr[pre:]
        valid = ~np.isnan(X_arr).any(axis=(1, 2))
    else:
        X_arr = X_arr.iloc[pre:].reset_index(drop=True)
        valid = ~X_arr.isna().any(axis=1).to_numpy()

    #rows can still be incomplete here for a reason unrelated to `pre`: real gaps in the raw data
    #(e.g. a column that is NaN for long stretches) land inside a lookback window too. Dropping them
    #mirrors PreProcessDataset's own dropna() on the training split -- predicting through a NaN would
    #otherwise silently poison the model's output instead of raising or being filtered out.
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"build_inference_features: dropping {n_dropped} of {len(valid)} rows with an "
              f"incomplete (NaN) feature lookback window.")
    timestamps = timestamps[valid].reset_index(drop=True)
    X_arr = X_arr[valid] if mltype in _ANN_TYPES else X_arr.loc[valid]

    xscaler = bundle['scaler']['X_data']
    if mltype in _ANN_TYPES:
        samples, steps, n_feat = X_arr.shape
        X_scaled = xscaler.transform(X_arr.reshape(samples * steps, n_feat)).reshape(samples, steps, n_feat)
    else:
        X_scaled = xscaler.transform(X_arr)

    return X_scaled, timestamps


def predict_with_bundle(data, bundle):
    """
    The actual "load a trained model + new raw rows -> forecast" entry point: builds the input
    tensor with build_inference_features(), runs bundle['model'].predict(), and inverse-transforms
    the result back to physical units with the bundle's own Y scaler (same approach TestMLmodel
    uses on the held-out test set, applied here to arbitrary new rows instead).

    Returns
    -------
    DataFrame indexed by the timestamp each forecast was made from (see build_inference_features),
    one column per step of the horizon ('t+1', 't+2', ... 't+H').
    """
    X, timestamps = build_inference_features(data, bundle)
    predictions = bundle['model'].predict(X)
    forecasted = bundle['scaler']['Y_data'].inverse_transform(predictions)
    columns = ["t+%d" % (step + 1) for step in range(forecasted.shape[1])]
    return pd.DataFrame(forecasted, index=pd.Index(timestamps, name=data.index.name), columns=columns)
