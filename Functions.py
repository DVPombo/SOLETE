# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:35:08 2021
Latest edit: Phase 7, Session 8 (split into solete_pipeline/)

author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com
LinkedIn: https://www.linkedin.com/in/dvp/
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo

Historically this module collected every function called by the rest of
the scripts. As of Phase 7 Session 8, the actual implementations have
moved into the solete_pipeline/ package (io.py, physics.py, qc.py,
preprocessing.py, modeling.py, postprocess.py) so each piece can be
imported and unit-tested in isolation -- see CONTRIBUTING.md for the
rationale and solete_pipeline/__init__.py for the module map.

This file is now a thin re-export shim, kept so every existing
`from Functions import X` (RunMe.py, MLForecasting.py, the baseline
scripts, notebooks, tests, etc.) keeps working unmodified, with exactly
the same import cost it always had (this file still needs everything,
including keras/tensorflow for the ML training functions). New code that
only needs e.g. the QC layer or the physics models should import from the
specific solete_pipeline submodule directly instead -- see that package's
__init__.py docstring for why that matters.
"""

from solete_pipeline.qc import (  # noqa: F401  (re-export shim, see module docstring)
    QC_VALID,
    QC_MISSING,
    QC_SENSOR_ERROR,
    QC_PHYSICALLY_IMPLAUSIBLE,
    QC_INTERPOLATED,
    QC_AGGREGATION_AFFECTED_BY_GAPS,
    QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED,
    QC_FLAG_PRECEDENCE,
    KNOWN_PRESSURE_SENTINELS,
    PRESSURE_PLAUSIBLE_RANGE,
    apply_qc_flags,
    build_raw_value_qc_rules,
    build_substitution_qc_rule,
)
from solete_pipeline.physics import (  # noqa: F401
    PV_Performance_Model,
    Rincon_Pombo_ThermodynamicModel,
)
from solete_pipeline.preprocessing import (  # noqa: F401
    ExpandSOLETE,
    PreProcessDataset,
    series_to_forecast,
)
from solete_pipeline.io import (  # noqa: F401
    import_SOLETE_data,
    import_SOLETE_sample,
    import_PV_WT_data,
)
from solete_pipeline.modeling import (  # noqa: F401
    PrepareMLmodel,
    train_LSTM,
    train_CNN,
    train_CNN_LSTM,
    TestMLmodel,
    generate_persistence,
)
from solete_pipeline.postprocess import (  # noqa: F401
    post_process,
    error_msg,
)
