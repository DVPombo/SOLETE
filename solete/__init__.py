# -*- coding: utf-8 -*-
"""
solete -- the SOLETE project as one importable package.

Layout:
    paths.py          WHERE everything lives (data folder, examples, results).
                      Every script and module resolves files through here.
    io.py             import_SOLETE_data, import_SOLETE_sample, import_PV_WT_data
    qc_codes.py       THE single QC vocabulary (codes, labels, severity order); shared with dataset/pipeline
    qc.py             read present <col>_qc flags, add code 6, legacy v3 raw-value checks
    expansion.py      expand_physical: deterministic row-wise model columns (Pac, P_Solar_clean[kW], P_hybrid[kW], ...)
    synthetic.py      synthetic SOLETE-like table for tests/diagnostics
    physics.py        PV_Performance_Model, Rincon_Pombo_ThermodynamicModel
    preprocessing.py  ExpandSOLETE, PreProcessDataset, series_to_forecast
    modeling.py       PrepareMLmodel, train_LSTM/CNN/CNN_LSTM, TestMLmodel (needs keras)
    postprocess.py    post_process, error_msg
    metrics.py        forecast metrics (deterministic and probabilistic)
    dataset.py        the `SOLETE` convenience class used by the benchmarks
    benchmark/        shared benchmark harness (splits, loaders, result files)

Deliberately NOT re-exported here: importing this __init__.py only pulls in
pandas (for the warnings-filter line below), nothing else. Import from the
specific submodule you need (e.g. `from solete.qc import apply_qc_flags`) to
get exactly that module's own dependencies and nothing more. qc.py, physics.py
and paths.py need no keras/tensorflow at all, and the dataset cleaning scripts
under dataset/ rely on that (they import solete.paths only).

Original author: Daniel Vazquez Pombo
Licensed under the MIT License -- see LICENSE at the repo root. If you use
this work, please give credit (see CITATION.cff).
"""

import warnings

import pandas as pd

__version__ = "4.0.0.dev0"

# Carried over from the original Functions.py -- suppresses a pandas
# PerformanceWarning that fires from a call pattern elsewhere in this
# pipeline. The filter is still load-bearing (removing it reintroduces the
# warning), so it is kept. Applied here, at the top-level package, so it is
# active regardless of which submodule(s) a caller imports.
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
