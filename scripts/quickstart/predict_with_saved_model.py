# -*- coding: utf-8 -*-
"""
Load a previously trained+saved model and score new rows with it -- the
"serving" path scripts/quickstart/MLForecasting.py doesn't have: even with
Control_Var['trainVSimport'] = False, that script still re-runs the whole
import_SOLETE_data -> ExpandSOLETE -> PreProcessDataset pipeline (including
PreProcessDataset silently *refitting* a new scaler on whatever data happens
to be passed to it) just to assemble ML_DATA/Scaler for TestMLmodel.

This script instead:
    1) loads the model + its manifest sidecar (the scaler it was actually
       trained with, plus PossibleFeatures/IntrinsicFeature/H/PRE/resolution)
       with solete.modeling.load_model_bundle(), which also checks the bundle
       is compatible with the Control_Var below before anything is predicted;
    2) builds new rows the same way ExpandSOLETE always has (no retraining,
       no train/val/test split, no scaler refit);
    3) scores them with solete.inference.predict_with_bundle(), which applies
       the *loaded* scaler (transform only) and inverse-transforms the result
       back to physical units.

Prerequisite: run scripts/quickstart/MLForecasting.py once first, with
Control_Var['trainVSimport'] = True and Control_Var['saveMLmodel'] = True, so
outputs/Trained_<MLtype>.joblib (or .h5) and outputs/Trained_<MLtype>_manifest.joblib
exist. The Control_Var below must describe that same run (MLtype, PossibleFeatures,
IntrinsicFeature, H, PRE, resolution, data_version) -- load_model_bundle() raises a
clear error if it doesn't.

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).
"""

import sys as _sys  # _REPO_ROOT_BOOT
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: `import solete` works from any cwd / Spyder

from solete.io import import_SOLETE_data, import_PV_WT_data
from solete.modeling import load_model_bundle
from solete.inference import predict_with_bundle

#%% Control The Script
# Must match the Control_Var that produced the saved model/manifest you want to load (see
# scripts/quickstart/MLForecasting.py's own Control_Var doc for what each key means).

Control_Var = {
    'resolution': '60min',
    'SOLETE_builvsimport': 'Build',
    'SOLETE_save': False,
    'MLtype': 'RF',  # must match the MLtype of the saved Trained_<MLtype>.joblib/.h5 you want to load
    'IntrinsicFeature': 'P_Solar[kW]',
    'PossibleFeatures': ['TEMPERATURE[degC]', 'HUMIDITY[%]', 'WIND_SPEED[m1s]', 'WIND_DIR[deg]',
                        'GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Gaia[kW]', 'P_Solar[kW]', 'Pressure[mbar]',
                        'Pac', 'Pdc', 'TempModule', 'TempCell',
                        'HoursOfDay', 'MeanPrevH', 'StdPrevH', 'MeanWindSpeedPrevH', 'StdWindSpeedPrevH',
                        ],
    'H': 10,
    'PRE': 5,
}

#%% Import Data
# Same data-preparation step MLForecasting.py uses (ExpandSOLETE runs King's PV model etc.) --
# only PreProcessDataset's train/val/test split + scaler-refit is skipped, see module docstring.

PVinfo, WTinfo = import_PV_WT_data()
DATA = import_SOLETE_data(Control_Var, PVinfo, WTinfo)

#%% Load the saved model + its own scaler/metadata (validated against Control_Var above)
bundle = load_model_bundle(Control_Var)

#%% Score the most recent rows (here: the whole imported file; in real use this would be new,
# just-arrived rows). predict_with_bundle() drops the first PRE rows (not enough lookback yet).
forecasts = predict_with_bundle(DATA, bundle)

print("\nLast 5 forecasts (one row per 't', columns are the horizon steps t+1..t+H):")
print(forecasts.tail())
