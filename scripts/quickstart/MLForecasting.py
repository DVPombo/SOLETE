# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:35:08 2021
Latest edit July 2023

author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com
LinkedIn: https://www.linkedin.com/in/dvp/
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo

The purpose of this script is to give you a running example importing the dataset
from [1] and the physics-informed methodology described in [2, 3, 4]:
    
    [1] Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a 15-month  
    long holistic dataset including: Meteorology, co-located wind and solar PV
    power from Denmark with various resolutions. Data in Brief, 42, 108046.
        
    [2] Pombo, D. V., Bindner, H. W., Spataru, S. V., Sørensen, P. E., & Bacher, P. 
    (2022). Increasing the accuracy of hourly multi-output solar power forecast  
    with physics-informed machine learning. Sensors, 22(3), 749.
    
    [3] Pombo, D. V., Bacher, P., Ziras, C., Bindner, H. W., Spataru, S. V., & 
    Sørensen, P. E. (2022). Benchmarking physics-informed machine learning-based 
    short term PV-power forecasting tools. Energy Reports, 8, 6512-6520.
    
    [4] Pombo, D. V., Rincón, M. J., Bacher, P., Bindner, H. W., Spataru, S. V., 
    & Sørensen, P. E. (2022). Assessing stacked physics-informed machine learning 
    models for co-located wind–solar power forecasting. Sustainable Energy, Grids 
    and Networks, 32, 100943.

It should run without errors simply by placing all the files in the same location.
We have checked the hdf5 file (the actual dataset) compatibility with Matlab, Python and R.
If you encounter any troubles using it with other software let me know and I will see what I can do.

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).


Using SOLETE for the first time:
    1- Put all the files in the same folder and click run. If you get the message "Done!" in the console that is it.
    2- If it didn't work:
        a) Check the dependencies: 
            -Python 3.9.12          -Pandas 1.5.0 
            -Numpy 1.23.1           -Matplotlib 3.6.0
            -Scikit-Learn 1.1.      -Keras 2.10.0
            -TensorFlow 2.10.0      -CoolProp 6.4.3   
            -The SOLETE dataset [1] -> https://doi.org/10.11583/DTU.17040767 
        b) If that didn't solve it... You have a problem my friend ¯\_(ツ)_/¯ 
    
How to use the second and subsequent times:
    1- Configure it. Running with no arguments at all (including Spyder's "Run file") reproduces
        today's defaults exactly -- pick whichever of these three is most convenient, mix as needed:
            a) Edit scripts/quickstart/configs/default_control_var.json directly. It holds
               everything that used to be the hardcoded Control_Var dict + the 5 per-model
               hyperparameter dicts (RF/SVM/LSTM/CNN/CNN_LSTM). Field meanings are documented in
               scripts/quickstart/configs/README.md. This is the only place to edit ANN topology
               (Neurons/Dense are lists, not scalar CLI flags) or EarlyStopping settings.
            b) Copy that JSON file, change what you want, and run with
               `python MLForecasting.py --config path/to/your_copy.json`.
            c) Pass CLI flags for the handful of scalar options you're most likely to want to
               change without touching JSON, e.g. `--resolution 5min --mltype RF -H 10 --pre 5`.
               Run `python MLForecasting.py -h` for the full list. CLI flags win over --config,
               which wins over the shipped default JSON.
        Modify "SOLETE_builvsimport" from "Build" to "Import" (--import-solete) to avoid expanding
        the dataset every time and simply import the last built version. Note each version is
        exclusive to a resolution; the default is 60min, you can try 5min as well.
    2- After setting those parameters just run it. Take into account the size of 
    the data, it might take a while for the algorithm to finish.
    3- Now that you have some time to spare while it runs: 
        a) Drink some water, you are probably not hydrated enough.
        b) If you found SOLETE useful, and want to send some kudos, do so.

General comments:
    -Note that 'TempModule_RP' is commented out within 'PossibleFeatures'. This makes the 
    algorithm expand the dataset with a new thermodynamic model for the panels that gives
    you another panel temperature estimation. This is commented out because its code is 
    poorly optimized and takes some time to run. I strongly recomend you to run the builder
    once with all the possible expansions, save it and then simply import it afterwards.
    -Why does the original SOLETE dataset not include all these features? Because part of
    the point of this code is to show you how I did it. Also, it makes no sense for me to 
    upload a much larger file to a repository when you can pull it and obtain everything 
    you need.
    -What to do if you are not able to obtain the exact same results as we presented in
    the papers? Well you can cry, you can lift your fist in the air and scream hateful
    words addressed to my mother. Or you can realise that there is more to those papers
    than what I share here (an HPC for once). So pick up books and papers and keep working.
    
DISCLAIMERS:
1- Regarding papers, I was using a regular PC only for coding, models were run in an HPC.
    This means that this is obviously not the exact code I used for those. Only the dataset
    is identical. So don't expect to obtain the exact same results directly.
2- There are better ways to configure the ML-models, have fun playing with it.
3- SVM will take for ever if run in this way for the whole set. An alternative programming 
    based on this -> https://stackoverflow.com/questions/31681373/making-svm-run-faster-in-python    
    was employed in when researching what ended up becoming [4]. 
4- I put together the dataset and these scripts in a couple of days, so do not expect them to be pretty or 100% error free.
"""

import sys as _sys  # _REPO_ROOT_BOOT
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: `import solete` works from any cwd / Spyder
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # this dir: `import ml_config` works from any cwd / Spyder

from solete.io import import_SOLETE_data, import_PV_WT_data
from solete.preprocessing import PreProcessDataset
from solete.modeling import PrepareMLmodel, TestMLmodel
from solete.postprocess import post_process
import ml_config

#%% Control The Script:
# Control_Var holds every value that defines the behaviour of the algorithm: horizon to forecast,
# train/validation/test split, which feature to predict, which to use as extrinsic features, and
# the 5 per-model hyperparameter dicts (RF/SVM/LSTM/CNN/CNN_LSTM). It is built from
# scripts/quickstart/configs/default_control_var.json, optionally layered with a --config file and/or
# CLI flags (run `python MLForecasting.py -h` for the list) -- see that file's docstring-equivalent,
# scripts/quickstart/configs/README.md, for what each key means. Running with no arguments at all
# (e.g. Spyder's "Run file") reproduces the shipped JSON defaults exactly.

Control_Var = ml_config.get_control_var()

#%% Import Data

PVinfo, WTinfo = import_PV_WT_data()
DATA=import_SOLETE_data(Control_Var, PVinfo, WTinfo)

#%% Generate Time Periods
ML_DATA, Scaler = PreProcessDataset(DATA, Control_Var)
# ML_DATA, Scaler = TimePeriods(DATA, Control_Var) 

#%% Train, Evaluate, Test
ML = PrepareMLmodel(Control_Var, ML_DATA) #train or import model
results = TestMLmodel(Control_Var, ML_DATA, ML, Scaler)

#%% Post-Processing
analysis = post_process(Control_Var, results)























