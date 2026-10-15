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

from .paths import output_path

import matplotlib.pyplot as plt
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error


def post_process(control, RESULTS):
    """
    Computes errors and plots RMSE.

    Parameters
    ----------
    control : dict
        Control_Var
    RESULTS : dict of DataFrames
        The dict containing Forecasted, Observed, and Persistence dataframes

    Returns
    -------
    analysis : dict of DataFrame
        Contains the results in terms of MAE, MSE, and RMSE per horizon sample, but also the residual

    """
    
    print("Post-processing results...")
    print("    -Computing errors:")
    
    residual=RESULTS["Observed"] - RESULTS["Forecasted"] 
    
    #raw values gives us the value per horizon time step
    rmse = pd.DataFrame(root_mean_squared_error(RESULTS["Observed"], RESULTS["Forecasted"], multioutput='raw_values'), columns=["Forecaster"])
    mae = pd.DataFrame(mean_absolute_error(RESULTS["Observed"], RESULTS["Forecasted"], multioutput='raw_values'), columns=["Forecaster"])
    mse = pd.DataFrame(mean_squared_error(RESULTS["Observed"], RESULTS["Forecasted"], multioutput='raw_values'), columns=["Forecaster"])
    
    for error in ["Persistence"]: #["Persistence", "Persistence24"]
        rmse[error] = root_mean_squared_error(RESULTS["Observed"], RESULTS[error], multioutput='raw_values')
        mae[error] = mean_absolute_error(RESULTS["Observed"], RESULTS[error], multioutput='raw_values')
        mse[error] = mean_squared_error(RESULTS["Observed"], RESULTS[error], multioutput='raw_values')
    
        
    #these enable the autoscaling of the plot
    ymin = min([rmse.min().min()*1.1, 0])
    ymax = max([rmse.max().max()*1.1, 0])
        
    fig = plt.figure()
    plt.plot(rmse)
    plt.grid()
    plt.xlim((rmse.index[0], rmse.index[-1]))
    plt.ylim(ymin, ymax) 
    plt.ylabel( "RMSE" )
    plt.xlabel( "Time Horizon" )
    plt.title("RMSE=" + str(round(rmse.mean().iloc[0], 3)) + " MAE = " + str(round(mae.mean().iloc[0], 3))\
              +" MSE = "+ str(round(mse.mean().iloc[0], 3)))
    plt.legend(rmse.columns)
    
    filename = "RMSE_" + control["MLtype"]
    print("    -Saving RMSE plot as: ", filename)
    plt.savefig(str(output_path(filename)), dpi=500)
    print("...Done")
    
    analysis ={'_description_' : 'Holds different statistics related to prediction accuracy',
            'RMSE' : rmse, #root mean squared error
            'MAE' : mae, #mean absolute error     
            'MSE' : mse, #mean squared error
            'Residual': residual, #residual error
            }
    
    print("\n\nThe End!")
    
    return analysis



def error_msg(key):
    """
    This function collects error messages to help you fix common errors I imagined
    could occur while using the default SOLETE.

    Parameters
    ----------
    key : str
        Keyword that selects error messages

    Returns
    -------
    Kills the execution and prints an error message and help to solve it.

    """
    
    if key == "missing_SOLETE_datafile": 
        print("ERROR: SOLETE dataset not found.\n")
        print("Worry not, it is easy to fix:")
        print("     1-Make sure you have downloaded the dataset from:  https://doi.org/10.11583/DTU.17040767.v3")
        print("     2-Extract the .zip in the same directory as the scripts/quickstart/RunMe.py and scripts/quickstart/MLForecasting.py files")
        print("     3-Try running the script again. If it fails, double check the spelling of the file's name")
        print("     Available options: 1sec, 1min, 5min or 60min")
        
    elif key == "resolution":
        print("ERROR: You have selected a resolution that is not available.\n")
        print("Worry not, it is easy to fix:")
        print("     1-Go to the code section: Control The Script")
        print("     2-Edit the dict Control_Var[resolution]")
        print("     Available options: 1sec, 1min, 5min or 60min")
    
    elif key == "missing_expanded_SOLETE":
        print("ERROR: You have selected Import a expanded SOLETE dataset.")
        print("Unfortunately, you don't have such a file in the current directory")
        print("The most likely error is that you have never run the Build and Save option for the selected resolution\n")
        print("Worry not, it is easy to fix:")
        print("     1-Go to the code section: Control The Script")
        print("     2-Edit the dict Control_Var[SOLETE_builvsimport] select Build")
        print("     3-Edit the dict Control_Var[SOLETE_save] select True")
        print("     4-Run it once like this (no need to train any ML model)")
        print("     5-Revert Control_Var[SOLETE_builvsimport] back to Import")
        print("It should be fixed now.\n\n\n")        
    
    elif key == "missing_feature_expanded_SOLETE":
        print("ERROR: You have imported a version of the expanded SOLETE dataset, which does not include one of the features you would like to employ.")
        print("The most likely error is that you have never run the Build and Save option including the desired feature\n")
        print("Worry not, it is easy to fix:")
        print("     1-Go to the code section: Control The Script")
        print("     2-Edit the dict Control_Var[SOLETE_builvsimport] select Build")
        print("     3-Edit the dict Control_Var[SOLETE_save] select True")
        print("     4-Run it once like this (no need to train any ML model)")
        print("     5-Revert Control_Var[SOLETE_builvsimport] back to Import")
        print("It should be fixed now, try to import again.\n\n\n")        

    elif key == "config_file_not_found":
        print("ERROR: The --config JSON file you pointed MLForecasting.py at does not exist.\n")
        print("Worry not, it is easy to fix:")
        print("     1-Double check the path you passed to --config")
        print("     2-See scripts/quickstart/configs/README.md for the expected file shape")
        print("     3-Or drop --config entirely to use scripts/quickstart/configs/default_control_var.json")

    elif key == "invalid_config_json":
        print("ERROR: The config JSON file could not be parsed (or is not a JSON object).\n")
        print("Worry not, it is easy to fix:")
        print("     1-Make sure the file is valid JSON (no trailing commas, matching braces/quotes)")
        print("     2-Compare it against scripts/quickstart/configs/default_control_var.json")
        print("     3-See scripts/quickstart/configs/README.md for the expected file shape")

    elif key == "invalid_train_val_test":
        print("ERROR: Control_Var['Train_Val_Test'] must be three percentages that sum to 100.\n")
        print("Worry not, it is easy to fix:")
        print("     1-Edit Train_Val_Test in your config JSON, or pass --train-val-test TRAIN VAL TEST")
        print("     2-Make sure the three numbers add up to 100, e.g. 70 20 10")

    print("\n")    
    sys.exit("Did I do that? ¯\\_(ツ)_/¯ \n\n\nCheck at the top for hints on what went wrong!!!")
    #yes, that was a reference to good old Steve Urkel
    pass











