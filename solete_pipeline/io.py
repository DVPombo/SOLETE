# -*- coding: utf-8 -*-
"""
Part of the solete_pipeline package -- split out of the original
Functions.py (Phase 7, Session 8) for independent testability.
See Functions.py (kept as a re-export shim) and CONTRIBUTING.md for
why this split happened and how the modules relate to each other.

Original author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).
"""

import pandas as pd

from .preprocessing import ExpandSOLETE
from .qc import apply_qc_flags, build_raw_value_qc_rules
from .postprocess import error_msg


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
        name = 'SOLETE_Pombo_'+ Control_Var['resolution'] +'.h5' 
        name_import = name[:-3]+'_Expanded.h5'
        
    
    if Control_Var["SOLETE_builvsimport"]=='Build':
        
        df=pd.read_hdf(name) #import the Raw SOLETE based on the selected resolution
        print("SOLETE was imported:")
        print("    -resolution: ", Control_Var['resolution'])
        print("    -version: Original. \n")
        
        Control_Var['OriginalFeatures']=list(df.columns)
        
        print("SOLETE was imported with a resolution of: ", Control_Var['resolution'], "\n")
        
        #QC flags for raw-value issues (Phase 2, see QC_SCHEMA.md). Computed here,
        #directly on the raw columns, rather than deferred into ExpandSOLETE --
        #none of these rules depend on anything ExpandSOLETE derives.
        _, qc_counts = apply_qc_flags(df, build_raw_value_qc_rules(df))
        print("QC flags applied:", qc_counts, "\n")
        
        ExpandSOLETE(df, [PVinfo, WTinfo], Control_Var)
        
        if Control_Var["SOLETE_save"]==True:
            df.to_hdf(name_import, key='name', mode='w')
            
    elif Control_Var["SOLETE_builvsimport"]=='Import':
        
        try:
            df=pd.read_hdf(name_import)
        except FileNotFoundError:
            error_msg(key = "missing_expanded_SOLETE")
        
        print("SOLETE was imported:")
        print("    -resolution: ", Control_Var['resolution'])
        print("    -version: Expanded. ")
        
        #QC flags (Phase 2): recomputed from the raw columns rather than trusted
        #from disk. This is deliberately self-healing -- if a saved _Expanded.h5
        #file ever had its _qc columns dropped below (because they weren't listed
        #in Control_Var['PossibleFeatures'], the same registration gap that
        #already affects P_Solar_model_substituted, see QC_SCHEMA.md section 7),
        #this regenerates them here from the still-present raw columns instead of
        #silently going without. Note this only sticks if the caller's
        #PossibleFeatures *also* lists the _qc columns -- otherwise the drop loop
        #a few lines down removes them again immediately after.
        _, qc_counts = apply_qc_flags(df, build_raw_value_qc_rules(df))
        print("QC flags (re)applied on Import:", qc_counts)
        
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
    df = pd.read_hdf(path)
    print(f"SOLETE sample was imported from: {path}")
    print(f"    {len(df)} rows, {df.index.min()} .. {df.index.max()}\n")

    Control_Var['OriginalFeatures'] = list(df.columns)

    #Same raw-value QC flags as import_SOLETE_data()'s 'Build' branch
    #(see QC_SCHEMA.md) -- computed here so the notebooks demonstrate the
    #real entry point rather than bypassing it with a bare pd.read_hdf.
    _, qc_counts = apply_qc_flags(df, build_raw_value_qc_rules(df))
    print("QC flags applied:", qc_counts, "\n")

    ExpandSOLETE(df, [PVinfo, WTinfo], Control_Var)

    return df


def import_PV_WT_data():
    """
    Returns
    -------
    PV : dict
        Holds data regarding the PV string in SYSLAB 715
    WT : dict
        Holds data regarding the Gaia wind turbine

    """
    
    PV={
        "Type": "Poly-cristaline",
        "Az": 60,#deg
        "Estc": 1000, #W/m**2
        "Tstc": 25,#C
        'Pmp_stc' : [165, 125], #W
        'ganma_mp' : [-0.478/100, -0.45/100], #1/K
        'Ns':[18, 6], #int
        'Np':[2, 2], #int
        'a' : [-3.56, -3.56], #module material construction parameters a, b and D_T
        'b' : [-0.0750, -0.0750],
        'D_T' : [3, 3],# represents the difference between the module and cell temperature
                        #these three parameters correspond to glass/cell/polymer sheet with open rack
                        #they are extracted from Sandia document King, Boyson form 2004 page 20
        'eff_P' : [[0, 250, 400, 450, 500, 600, 650, 750, 825, 1000, 1200, 1600, 2000, 3000, 4000,  6000, 8000, 10000],
                   [0, 250, 400, 450, 500, 600, 650, 750, 825, 1000, 1200, 1600, 2000, 3000, 4000,  6000, 8000, 10000]],
        'eff_%' : [[0, 85.5, 90.2, 90.9, 91.8, 92, 92.3, 94, 94.4, 94.8, 95.6, 96, 97.3, 97.7, 98, 98.1, 98.05, 98],
                   [0, 85.5, 90.2, 90.9, 91.8, 92, 92.3, 94, 94.4, 94.8, 95.6, 96, 97.3, 97.7, 98, 98.1, 98.05, 98]],
       "index": ['A','B'], #A and B refer to each channel of the inverter, which has connected a different string.
       "L": 10, # array characteristic length
       "W": 1.5, # array width
       "d": 0.1, # array thickness
       "k_r": 350, # PV conductive resistance W/(m*K)
       # "": ,
        }
    
    WT={
        "Type": "Asynchronous",
        "Mode": "Passive, downwind vaning",
        "Pn": 11,#kW
        "Vn": 400,#V
        'CWs' : [3.5, 6, 8, 10, 10.5, 11, 12, 13, 13.4, 14, 16, 18, 20, 22, 24, 25,],#m/s
        'CP' : [0, 5, 8.5, 10.9, 11.2, 11.3, 11.2, 10.5, 10.5, 10, 8.8, 8.7, 8, 7.3, 6.6, 6.3,],#kW
        "Cin": 3.5,#m/s
        "Cout": 25,#m/s
        "HH": 18,#m
        "D": 13,#m
        "SA": 137.7,#m**2
        "B": 2,#int       
        }
    
    return PV, WT


