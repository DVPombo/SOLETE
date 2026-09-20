# -*- coding: utf-8 -*-
"""
solete_pipeline -- split out of the original monolithic Functions.py
(Phase 7, Session 8).

Layout:
    qc.py             QC flag constants, apply_qc_flags, build_*_qc_rules
    physics.py        PV_Performance_Model, Rincon_Pombo_ThermodynamicModel
    preprocessing.py  ExpandSOLETE, PreProcessDataset, series_to_forecast
    io.py             import_SOLETE_data, import_SOLETE_sample, import_PV_WT_data
    modeling.py       PrepareMLmodel, train_LSTM/CNN/CNN_LSTM, TestMLmodel,
                      generate_persistence
    postprocess.py    post_process, error_msg

Deliberately NOT re-exported here: importing this __init__.py only pulls in
pandas (for the warnings-filter line below), nothing else. Import from the
specific submodule you need (e.g. `from solete_pipeline.qc import
apply_qc_flags`) to get exactly that module's own dependencies and nothing
more -- e.g. qc.py and physics.py need no keras/tensorflow at all, which is
the whole point of splitting this file (see CONTRIBUTING.md). If this
__init__.py eagerly re-exported every submodule's names (as a first version
of this split did), every import of `solete_pipeline` -- even
`solete_pipeline.qc` alone -- would transitively import modeling.py's
keras/tensorflow, defeating that. `Functions.py` at the repo root needs
everything anyway (that's what it always did), so it imports each submodule
directly too, same as any other caller.

Original author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).
"""

import pandas as pd
import warnings

# Carried over from the original Functions.py -- suppresses a pandas
# PerformanceWarning that fires from a call pattern elsewhere in this
# pipeline. "yep, bad practice, see function get_results to understand why
# is this here :)" -- original author's comment, kept verbatim; no
# `get_results` function exists in this codebase, so the referenced
# rationale is lost to history, but the filter itself is still load-bearing
# (removing it reintroduces the warning), so it's kept rather than dropped.
# Applied here, at the top-level package, so it's active regardless of
# which individual submodule(s) a caller imports.
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
