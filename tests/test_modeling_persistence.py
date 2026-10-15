# -*- coding: utf-8 -*-
"""
Round-trip coverage for the model-persistence/serving path (save -> load ->
predict), closing the zero-coverage gap noted against solete/modeling.py:
PrepareMLmodel's save/load branches and TestMLmodel had no functional test
before this file.

SYNTHETIC: unlike most of this repo's tests (see test_solete_pipeline_units.py's
docstring), this one is deliberately synthetic rather than grounded in a real
SOLETE row. The thing under test here is the save/load/predict *mechanics*
(does the reloaded model+scaler+manifest reproduce the in-memory one exactly,
does build_inference_features() window/scale new rows the same way
PreProcessDataset does), not forecasting accuracy on real PV/wind data -- a
small, deterministic, hand-checkable DataFrame is the right tool for that.
RF is used throughout: it is the cheap, exact (no training-randomness-from-
epochs) model family, so float equality assertions are meaningful.

Run with: pytest tests/test_modeling_persistence.py -v  (from the repo root)
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# test_metrics.py (collected before this file alphabetically) installs lightweight fake
# sys.modules entries for "keras"/"keras.models"/"keras.layers" (but not "keras.callbacks") to
# avoid loading real keras/tensorflow for tests that never touch the ML code. In a single pytest
# session those fakes stay cached in sys.modules, which breaks solete.modeling's own
# `from keras.callbacks import EarlyStopping` here. This file genuinely needs the real package, so
# drop any such stand-ins first and force a fresh import, regardless of collection order.
for _mod_name in list(sys.modules):
    if _mod_name == "keras" or _mod_name.startswith("keras."):
        del sys.modules[_mod_name]

import solete.paths as paths
from solete.preprocessing import PreProcessDataset
from solete.modeling import PrepareMLmodel, load_model_bundle
from solete.inference import build_inference_features, predict_with_bundle


N_ROWS = 240
PRE = 2
H = 3
FEATURES = ["feat_a", "feat_b", "target"]


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(0)
    #PreProcessDataset's reset_index()/set_index('index', ...) round trip assumes an unnamed index
    #(pandas' default reset_index() column name), matching how the real pipeline's DataFrames arrive
    index = pd.date_range("2020-01-01", periods=N_ROWS, freq="h")
    feat_a = np.sin(np.linspace(0, 20, N_ROWS)) + rng.normal(scale=0.01, size=N_ROWS)
    feat_b = np.cos(np.linspace(0, 20, N_ROWS)) + rng.normal(scale=0.01, size=N_ROWS)
    target = 0.5 * feat_a + 0.25 * feat_b + rng.normal(scale=0.01, size=N_ROWS)
    return pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "target": target}, index=index)


@pytest.fixture
def control():
    return {
        'MLtype': 'RF',
        'trainVSimport': True,
        'saveMLmodel': True,
        'random_seed': None,
        'Train_Val_Test': [70, 20, 10],
        'RF_SVM_use_val_as_train': True,
        'Scaler': 'MinMax01',
        'IntrinsicFeature': 'target',
        'PossibleFeatures': FEATURES,
        'H': H,
        'PRE': PRE,
        'resolution': '60min',
        'RF': {'n_trees': 5, 'random_state': 32},
    }


@pytest.fixture
def trained_bundle(tmp_path, monkeypatch, synthetic_data, control):
    """Trains+saves an RF model (with manifest) into a tmp outputs/ dir, then reloads it through
    load_model_bundle() -- the save->load half of the round trip."""
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)

    ml_data, scaler = PreProcessDataset(synthetic_data, control)
    ml = PrepareMLmodel(control, ml_data, scaler=scaler)

    bundle = load_model_bundle(control)
    return ml, ml_data, scaler, bundle


def test_manifest_round_trips_metadata(control, trained_bundle):
    _ml, _ml_data, _scaler, bundle = trained_bundle
    assert bundle['possible_features'] == control['PossibleFeatures']
    assert bundle['intrinsic_feature'] == control['IntrinsicFeature']
    assert bundle['H'] == control['H']
    assert bundle['PRE'] == control['PRE']
    assert bundle['resolution'] == control['resolution']
    assert bundle['mltype'] == 'RF'


def test_loaded_model_predicts_identically_to_in_memory_model(trained_bundle):
    ml, ml_data, _scaler, bundle = trained_bundle
    expected = ml.predict(ml_data['X_TEST'])
    actual = bundle['model'].predict(ml_data['X_TEST'])
    np.testing.assert_array_equal(actual, expected)


def test_loaded_scaler_is_the_exact_fitted_scaler(trained_bundle):
    _ml, _ml_data, scaler, bundle = trained_bundle
    # MinMaxScaler's fitted state; comparing it (rather than re-deriving it) is the point: a
    # refit-on-next-run scaler would only coincidentally match these.
    np.testing.assert_array_equal(bundle['scaler']['X_data'].data_min_, scaler['X_data'].data_min_)
    np.testing.assert_array_equal(bundle['scaler']['X_data'].data_max_, scaler['X_data'].data_max_)
    np.testing.assert_array_equal(bundle['scaler']['Y_data'].data_min_, scaler['Y_data'].data_min_)
    np.testing.assert_array_equal(bundle['scaler']['Y_data'].data_max_, scaler['Y_data'].data_max_)


def test_load_model_bundle_rejects_incompatible_control(tmp_path, monkeypatch, synthetic_data, control):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    ml_data, scaler = PreProcessDataset(synthetic_data, control)
    PrepareMLmodel(control, ml_data, scaler=scaler)

    bad_control = dict(control, H=H + 1)
    with pytest.raises(ValueError, match="H"):
        load_model_bundle(bad_control)


def test_load_model_bundle_without_manifest_raises_clear_error(tmp_path, monkeypatch, synthetic_data, control):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    ml_data, scaler = PreProcessDataset(synthetic_data, control)
    # Save the model but withhold the scaler -> no manifest sidecar is written (old behavior).
    PrepareMLmodel(control, ml_data, scaler=None)

    with pytest.raises(FileNotFoundError):
        load_model_bundle(control)


@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_build_inference_features_matches_a_hand_built_window(synthetic_data, trained_bundle):
    """Independent correctness check of build_inference_features()'s windowing/scaling for the
    dataset's very last row, built by hand from the raw DataFrame rather than by re-deriving it
    through PreProcessDataset (whose train/val/test split makes a row-for-row comparison awkward)."""
    _ml, _ml_data, _scaler, bundle = trained_bundle

    X, timestamps = build_inference_features(synthetic_data, bundle)
    assert timestamps.iloc[-1] == synthetic_data.index[-1]
    assert X.shape == (N_ROWS - PRE, len(FEATURES) * (PRE + 1))

    # Hand-build the last row's lag vector: for each feature (in PossibleFeatures order), the
    # values at t-PRE ... t, oldest to newest -- exactly what series_to_forecast(..., PRE, 0)
    # produces per column before concatenation.
    hand_built = np.concatenate([
        synthetic_data[col].iloc[-(PRE + 1):].to_numpy() for col in FEATURES
    ]).reshape(1, -1)
    hand_built_scaled = bundle['scaler']['X_data'].transform(hand_built)
    np.testing.assert_allclose(X[-1:], hand_built_scaled)


def test_predict_with_bundle_matches_manual_predict_for_last_row(synthetic_data, trained_bundle):
    _ml, _ml_data, _scaler, bundle = trained_bundle

    forecasts = predict_with_bundle(synthetic_data, bundle)
    assert list(forecasts.columns) == ["t+1", "t+2", "t+3"]
    assert forecasts.index[-1] == synthetic_data.index[-1]

    X, _timestamps = build_inference_features(synthetic_data, bundle)
    manual_prediction = bundle['model'].predict(X[-1:])
    manual_forecast = bundle['scaler']['Y_data'].inverse_transform(manual_prediction)
    np.testing.assert_allclose(forecasts.iloc[-1].to_numpy(), manual_forecast[0])


def test_build_inference_features_reports_missing_columns(synthetic_data, trained_bundle):
    _ml, _ml_data, _scaler, bundle = trained_bundle
    with pytest.raises(KeyError, match="feat_a"):
        build_inference_features(synthetic_data.drop(columns=["feat_a"]), bundle)
