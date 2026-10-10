# -*- coding: utf-8 -*-
"""
Part of the solete_pipeline package -- split out of the original
solete/ (Phase 7, Session 8) for independent testability.
See solete/ (kept as a re-export shim) and CONTRIBUTING.md for
why this split happened and how the modules relate to each other.

Original author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).
"""

import pandas as pd

from .paths import data_filename, find_data_file, derived_path, resolve_sample
from .preprocessing import ExpandSOLETE
from .qc import apply_qc_flags, legacy_v3_raw_value_rules, check_qc_vocabulary, present_qc_columns
from .postprocess import error_msg


def _add_raw_value_flags(df, data_version, reapply=False):
    """v3: add the legacy raw-value flags (code 11). v4: read-only check of the shared vocabulary."""
    if data_version == 'v3':
        _, qc_counts = apply_qc_flags(df, legacy_v3_raw_value_rules(df))
        print("QC flags (re)applied on Import:" if reapply else "QC flags applied:", qc_counts, "\n")
    else:
        unknown = check_qc_vocabulary(df)
        if unknown:
            raise ValueError(f"Unknown QC codes in a v4 file: {unknown} (vocabulary: solete/qc_codes.py)")
        print("v4 file: QC flags read as present:", present_qc_columns(df), "\n")


def import_SOLETE_data(Control_Var, PVinfo, WTinfo):
    """
    Imports different versions of SOLETE depending on the inputs:
        -resolution -SOLETE_builvsimport
    if built it has the option to save the expanded dataset

    Parameters
    ----------
    Control_Var : dict
        Holds information regarding what to do
    PVinfo : dict
        Holds data regarding the PV string in SYSLAB 715
    WTinfo : dict
        Holds data regarding the Gaia wind turbine

    Returns
    -------
    df : DataFrame
        The one, the only, the almighty SOLETE dataset

    """    
    
    print("___The SOLETE Platform___\n")
    
    if Control_Var['resolution'] not in ['1sec', '1min', '5min', '60min']:            
        error_msg(key = "resolution")
    else:
        # Which file version to read: 'v3' (default, the original SOLETE_Pombo_<res>.h5
        # files the platform and benchmarks were built on) or 'v4' (cleaned figshare
        # v4 files). Where the files live is decided in solete/paths.py (data/hdf5/).
        data_version = Control_Var.get('data_version', 'v3')
        if data_version not in ('v3', 'v4'):
            raise ValueError(f"Control_Var['data_version'] must be 'v3' or 'v4', got {data_version!r}")
        # v3: the original files, no flags of their own -> the legacy raw-value checks add them.
        # v4: files from the dataset pipeline (clean + flag + az/el) and, optionally, already carrying
        #     the platform's model columns. Their <column>_qc flags are read as they are and never
        #     recomputed or overwritten; the model columns are (re)computed by expand_physical, which
        #     reproduces them exactly if present. The working P_Solar[kW] is then P_Solar_clean[kW].
        name_stem = data_filename(Control_Var['resolution'], data_version)[:-3]
        name_import = name_stem + '_Expanded.h5'   # cached under data/derived/, see paths.derived_path
        
    
    if Control_Var["SOLETE_builvsimport"]=='Build':
        
        df=pd.read_hdf(find_data_file(Control_Var['resolution'], data_version)) #import the Raw SOLETE based on the selected resolution
        print("SOLETE was imported:")
        print("    -resolution: ", Control_Var['resolution'])
        print("    -version: Original (" + data_version + "). \n")
        
        Control_Var['OriginalFeatures']=list(df.columns)
        
        print("SOLETE was imported with a resolution of: ", Control_Var['resolution'], "\n")
        
        _add_raw_value_flags(df, data_version)
        
        ExpandSOLETE(df, [PVinfo, WTinfo], Control_Var)
        
        if Control_Var["SOLETE_save"]==True:
            df.to_hdf(derived_path(name_import), key='name', mode='w')
            
    elif Control_Var["SOLETE_builvsimport"]=='Import':
        
        try:
            df=pd.read_hdf(derived_path(name_import))
        except FileNotFoundError:
            error_msg(key = "missing_expanded_SOLETE")
        
        print("SOLETE was imported:")
        print("    -resolution: ", Control_Var['resolution'])
        print("    -version: Expanded. ")
        
        #v3 only: raw-value flags are recomputed from the raw columns rather than trusted from disk
        #(self-healing if a cached file lost its _qc columns, see docs/legacy/QC_SCHEMA_platform_v3.md
        #section 7; it only sticks if PossibleFeatures lists them). v4: the flags in the file are
        #authoritative and are left exactly as they are.
        _add_raw_value_flags(df, data_version, reapply=True)
        
        for col in Control_Var['PossibleFeatures']: #if the possiblefeature includes
        #something that was not in the import file, execution is killed with an error message
            if col not in df.columns: 
                error_msg(key = "missing_feature_expanded_SOLETE")
        
        print("")
        for col in df.columns: #the undesired columns are removed
        #undersired columns are those within the imported file not appearing in Control_Var['PossibleFeatures']
            if col not in Control_Var['PossibleFeatures']: 
                df = df.drop(col, axis=1)
                print("Dropped col: ", col)           
        
        print("\n")
        
    return df


def import_SOLETE_sample(path, Control_Var, PVinfo, WTinfo):
    """
    Phase 3 convenience wrapper (see examples/) -- lets the lightweight
    notebook sample files (e.g. SOLETE_sample.h5) go through the same
    intended entry point as import_SOLETE_data()'s 'Build' branch (raw-value
    QC flags, then ExpandSOLETE()'s PV-model expansion and substitution
    flag) without needing to match the SOLETE_Pombo_<resolution>.h5 naming
    convention import_SOLETE_data() assumes for the full-size real files.

    Does not modify import_SOLETE_data() or its behavior for the real
    files -- this is an additive wrapper for an explicit file path.

    Parameters
    ----------
    path : str
        Path to the sample .h5 file to load (e.g. 'SOLETE_sample.h5').
    Control_Var : dict
        Same Control_Var dict used elsewhere; only 'OriginalFeatures' is
        set/overwritten here, mirroring import_SOLETE_data()'s 'Build' branch.
    PVinfo, WTinfo : dict
        As returned by import_PV_WT_data().

    Returns
    -------
    df : DataFrame
        The sample dataset, expanded exactly as import_SOLETE_data()'s
        'Build' branch would (QC flags + King's PV performance model).
    """
    path = resolve_sample(path)  # bare names are looked up in examples/ then data/
    df = pd.read_hdf(path)
    print(f"SOLETE sample was imported from: {path}")
    print(f"    {len(df)} rows, {df.index.min()} .. {df.index.max()}\n")

    Control_Var['OriginalFeatures'] = list(df.columns)

    #Same raw-value QC flags as import_SOLETE_data()'s 'Build' branch -- computed here so the
    #notebooks demonstrate the real entry point. A sample that already carries pipeline flags
    #(a v4 sample) keeps them.
    _add_raw_value_flags(df, 'v4' if present_qc_columns(df) else 'v3')

    ExpandSOLETE(df, [PVinfo, WTinfo], Control_Var)

    return df


# the parameters live in solete/params.py (light imports); re-exported here for existing callers
from .params import import_PV_WT_data  # noqa: E402,F401
