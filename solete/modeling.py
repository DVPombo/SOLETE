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
import numpy as np

from .paths import output_path  # trained models / results / figures go to outputs/

import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed


def _early_stopping_callback(cfg):
    """Builds the EarlyStopping callback list for ML.fit() from a model's 'EarlyStopping' sub-dict
    (see MLForecasting.py Control_Var doc). Returns [] when missing/disabled, so existing configs
    train the full epo_num epochs exactly as before."""
    if not cfg or not cfg.get('enabled', False):
        return []
    return [EarlyStopping(monitor='val_loss',
                           patience=cfg.get('patience', 20),
                           restore_best_weights=cfg.get('restore_best_weights', True))]


#Keys copied verbatim from Control_Var into the manifest sidecar (see _save_manifest/load_model_bundle):
#enough to tell, at load time, whether a saved model was trained for the feature set/horizon/lookback/
#resolution/scaling the caller is about to use it for.
_MANIFEST_CONTROL_KEYS = ('PossibleFeatures', 'IntrinsicFeature', 'H', 'PRE', 'resolution', 'Scaler')


def _manifest_path(control):
    return output_path("Trained_" + control['MLtype'] + "_manifest.joblib")


def _save_manifest(control, scaler):
    """Persists the fitted Scaler (PreProcessDataset's Xscaler/Yscaler, fit on the training split
    only) alongside the small bits of Control_Var needed to tell whether a saved model is being
    reused in a compatible way later (see load_model_bundle). Saved as one joblib dict next to
    Trained_<MLtype>.joblib/.h5 -- a new, optional sidecar, it does not change the existing model
    filename/format."""
    manifest = {key: control[key] for key in _MANIFEST_CONTROL_KEYS if key in control}
    manifest['MLtype'] = control['MLtype']
    manifest['data_version'] = control.get('data_version', 'v3')
    manifest['scaler'] = scaler
    path = _manifest_path(control)
    print("Saving model manifest (scaler + feature/horizon metadata) with name: " + path.name)
    joblib.dump(manifest, str(path))


def load_model_bundle(control):
    """
    Loads a model saved by PrepareMLmodel(..., saveMLmodel=True) together with the manifest
    sidecar _save_manifest wrote next to it (the fitted Scaler the model was trained with, plus
    PossibleFeatures/IntrinsicFeature/H/PRE/resolution/Scaler-type), and checks that the two agree
    with whatever of those keys are present in `control` -- this is the compatibility check that
    plain joblib.load/keras.load_model can't give you (a shape-mismatch exception at best, or a
    silent wrong-semantics "success" at worst, if two configs happen to have matching shapes).

    Parameters
    ----------
    control : dict
        Must contain at least 'MLtype'. Any of _MANIFEST_CONTROL_KEYS present here are compared
        against the manifest; mismatches raise ValueError before anything is predicted.

    Returns
    -------
    dict with keys: 'model' (the loaded keras/sklearn object), 'scaler' (the Scaler dict PreProcess-
    Dataset produced, i.e. {'X_data':..., 'Y_data':...}), 'possible_features', 'intrinsic_feature',
    'H', 'PRE', 'resolution', 'scaler_type', 'mltype', 'data_version'.
    """
    manifest_path = _manifest_path(control)
    if not manifest_path.exists():
        raise FileNotFoundError(
            "No manifest found at " + str(manifest_path) + " -- this model was saved before the "
            "manifest sidecar existed, or saveMLmodel/scaler were not set when it was trained. "
            "Re-train and save it with PrepareMLmodel(control, ml_data, scaler=Scaler) to get one.")
    manifest = joblib.load(str(manifest_path))

    mismatches = []
    for key in _MANIFEST_CONTROL_KEYS:
        if key in control and key in manifest and control[key] != manifest[key]:
            mismatches.append("%s: requested=%r, saved model was trained with=%r" % (key, control[key], manifest[key]))
    if mismatches:
        raise ValueError(
            "load_model_bundle: control is NOT compatible with the saved " + control['MLtype'] +
            " model/manifest:\n  " + "\n  ".join(mismatches))

    model_control = dict(control, trainVSimport=False) #reuse the existing load branch, below
    model = PrepareMLmodel(model_control, ml_data=None)

    return {
        'model': model,
        'scaler': manifest['scaler'],
        'possible_features': manifest.get('PossibleFeatures'),
        'intrinsic_feature': manifest.get('IntrinsicFeature'),
        'H': manifest.get('H'),
        'PRE': manifest.get('PRE'),
        'resolution': manifest.get('resolution'),
        'scaler_type': manifest.get('Scaler'),
        'mltype': manifest['MLtype'],
        'data_version': manifest['data_version'],
    }


def PrepareMLmodel(control, ml_data, scaler=None):
    """
    Parameters
    ----------
    control : dict
        Control_Var
    ml_data : dict of dataframes
        It contains the training, validation and testing sets. Ignored (may be None) when
        control['trainVSimport'] is False, since importing a model needs no training data.
    scaler : dict, optional
        The Scaler dict PreProcessDataset returned alongside ml_data (fit on the training split).
        Only used when trainVSimport and saveMLmodel are both True: if given, it is persisted in a
        manifest sidecar next to the model (see _save_manifest) so a later load_model_bundle() call
        can reuse the *exact* fitted scaler instead of silently refitting a new one on whatever data
        happens to be passed through PreProcessDataset next time. Omitting it reproduces the old
        behavior exactly (model saved, no manifest).

    Returns
    -------
    ML : keras object or sci-kit learn object
        This is the trained model, it depends on which specific ML-method you are requesting

    """
    
    filename = "Trained_" + control['MLtype']

    #random_seed (Control_Var doc): None leaves training fully random (default); an int makes weight
    #initialization/shuffling reproducible across runs (seeds Python, NumPy and the Keras backend together,
    #but is not a GPU-determinism guarantee -- some backend ops stay nondeterministic there regardless).
    if control.get('random_seed') is not None:
        set_random_seed(control['random_seed'])
    
    if control['trainVSimport'] == True: #then, lets train a model
        
        
        if control['MLtype'] in ['RF', "SVM"]:
            filename = filename + ".joblib"
            if control['MLtype'] == 'RF':
                ML = RandomForestRegressor(n_estimators = control['RF']['n_trees'], random_state = control['RF']['random_state']) #initialize ML
            elif control['MLtype'] == 'SVM':
                ML = SVR(kernel = control['SVM']['kernel'], degree = control['SVM']['degree'],
                         gamma = control['SVM']['gamma'], coef0 = control['SVM']['coef0'],
                         C = control['SVM']['C'], epsilon = control['SVM']['epsilon']) #initialize ML
                if control["H"] > 1:
                    ML = MultiOutputRegressor(ML) #This is necessary for multioutput as SVR only support SO
            # It trains a separate SVR for each output whereas RF can inherently handle multiple classes and hence perform better
            
            print("Training " + control['MLtype'] + "...")
            ML.fit(X=ml_data['X_TRAIN'], y=ml_data['Y_TRAIN']) #train
            print("...Done")
        
        elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]: 
            filename = filename + ".h5"
            
            print("Training " + control['MLtype'] + "...\n")
        
            if control['MLtype'] == 'LSTM':
                ML, ANN_training = train_LSTM(ml_data, control)
            elif control['MLtype'] == 'CNN':    
                ML, ANN_training = train_CNN(ml_data, control)
            elif control['MLtype'] == 'CNN_LSTM':
                ML, ANN_training = train_CNN_LSTM(ml_data, control)
            print("...Done")
            
            #Plot the train vs validation loss and save the Figure
            fig = plt.figure()
            plt.plot(ANN_training.history[ "loss" ])
            plt.plot(ANN_training.history[ "val_loss" ])
            plt.grid()
            plt.title("Model train vs validation loss")
            plt.ylabel( "Loss" )
            plt.xlabel( "Epoch" )
            plt.legend([ "Train" , "Validation"], loc= "upper right" )
            plt.savefig(str(output_path('Training_Evaluation_' + control['MLtype'])), dpi=500)
            plt.show() 

        if control['trainVSimport'] and control['saveMLmodel']:
            if control['MLtype'] in ['RF', "SVM"]:
                print("Saving Trained Model with name: " + filename)
                joblib.dump(ML, str(output_path(filename)))
                
            elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]:
                print("Saving Trained Model with name: " + filename)
                ML.save(str(output_path(filename)))
            print("...Done\n")

            if scaler is not None:
                _save_manifest(control, scaler)
            else:
                print(control['MLtype'] + " manifest was NOT saved (no scaler passed to "
                      "PrepareMLmodel): load_model_bundle() will not work for it.\n")
        else:
            print(control['MLtype'] + " was NOT saved.\n")

    else: #we dont train but import the ML
        if control['MLtype'] in ['RF', "SVM"]:
            filename = filename + ".joblib"
            print("Importing " + filename + "...")
            ML=joblib.load(str(output_path(filename)))
            
        elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]: 
            filename = filename + ".h5"
            print("Importing " + filename + "...")
            ML=load_model(str(output_path(filename)))
        print("...Done\n")
    
        
    return ML




def train_LSTM(data, control):
    
    """
    data -> includes ML_DATA, that is the train, validation and test sets, without applying reshapes as a dict of dataframes
    Control_Var -> generic control variables as it brings stuff from LSTM as a dictionary
    pre -> previous number of samples to use in the prediction as an int
    hor -> prediction horizon as an int
    features -> basic feature and complements as a list of strings
    
    returns 
    ML -> the trained LSTM model
    ANN_training -> the history of the fit, can be used to plot loss function
    """
    pre=control["PRE"]
    hor=control["H"]

    train_data = data['X_TRAIN']#already a correctly-shaped (samples, pre+1, n_features) array from PreProcessDataset
    validation_data = data['X_VAL']

    train_target = data['Y_TRAIN']#already a correctly-shaped (samples, hor) array from PreProcessDataset
    validation_target = data['Y_VAL']
    
    ##### Designing Neuronal Network #######
    ML = Sequential() #initialize
    #no Masking layer: PreProcessDataset drops NaN rows outright, no 999 placeholder ever reaches training
    if control['LSTM']['Dense'][0] > 0: #add a dense if the number of neurons is higher than 0
        ML.add(Dense(control['LSTM']['Dense'][0]))
    if len(control["LSTM"]["Neurons"]) > 1: #if there is another LSTM coming afterwards we need the true
        return_seq = True
    else: #if a dense comes afterwards we need the false
        return_seq = False
    ML.add(LSTM(control["LSTM"]["Neurons"][0], input_shape=(train_data.shape[1], train_data.shape[2]),
                activation = control["LSTM"]["ActFun"],
                bias_initializer = "zeros", kernel_initializer = "random_uniform",
                return_sequences=return_seq)) #LSTM layer

    for index in range(1,len(control["LSTM"]["Neurons"])): #add the missing LSTM layers and a Dense after the last one
        if index != len(control["LSTM"]["Neurons"])-1:
            ML.add(LSTM(control["LSTM"]["Neurons"][index], activation = control["LSTM"]["ActFun"], return_sequences=True)) #LSTM layers
        else:
            ML.add(LSTM(control["LSTM"]["Neurons"][index], activation = control["LSTM"]["ActFun"], return_sequences=False)) #LSTM layers
            if control['LSTM']['Dense'][1] > 0: #we only add the dense one if the number of neurons is higher than 0
                ML.add(Dense(control['LSTM']['Dense'][1], activation = 'relu'))
    ML.add(Dense(hor)) #this is the output layer
    ### mean-absolute-error (MSAE) loss function & Adam version of stochastic gradient descent
    ML.compile(loss=control["LSTM"]["LossFun"], optimizer=control["LSTM"]["Optimizer"]) #, metrics=['mse', 'mae', 'mape', 'cosine']
    ML.summary()
    
    callbacks = _early_stopping_callback(control["LSTM"].get("EarlyStopping"))
    ANN_training = ML.fit(train_data, train_target,
                epochs = control["LSTM"]["epo_num"], 
                batch_size = control["LSTM"]["n_batch"],
                validation_data = (validation_data, validation_target), #fed here the validation data
                # validation_split
                verbose=2, #0-> shows nothing, 1-> shows progress bar, 2-> shows the number of epoch.
                shuffle=False,
                callbacks=callbacks)

    return ML, ANN_training, #data, 


def train_CNN(data, control):
    
    """
    data -> includes ML_DATA, that is the train, validation and test sets, without applying reshapes as a dict of dataframes
    Control_Var -> generic control variables as it brings stuff from LSTM as a dictionary
    pre -> previous number of samples to use in the prediction as an int
    hor -> prediction horizon as an int
    features -> basic feature and complements as a list of strings
    
    returns 
    ML -> the trained LSTM model
    ANN_training -> the history of the fit
    """
    pre=control["PRE"]
    hor=control["H"]

    train_data = data['X_TRAIN']#.values.reshape(len(data['X_TRAIN'].index), pre+1, len(features))
    validation_data = data['X_VAL']#.values.reshape(len(data['X_VAL'].index), pre+1, len(features))
    
    train_target = data['Y_TRAIN']#.values.reshape(len(data['Y_TRAIN'].index),hor)
    validation_target = data['Y_VAL']#.values.reshape(len(data['Y_VAL'].index),hor)
    
    ##### Designing Neuronal Network #######
    ML = Sequential() #initialize
    ML.add(Conv1D(filters=control["CNN"]["filters"], kernel_size=control["CNN"]["kernel_size"], padding='same',
                  activation=control["CNN"]["ActFun"], input_shape=(train_data.shape[1], train_data.shape[2])))
    if control['CNN']['Dense'][0] > 0: #add a dense if the number of neurons is higher than 0
        ML.add(Dense(control['CNN']['Dense'][0]))
    ML.add(MaxPooling1D(pool_size=control["CNN"]["pool_size"],  padding='same'))
    if control['CNN']['Dense'][1] > 0: #we only add the dense one if the number of neurons is higher than 0
                ML.add(Dense(control['CNN']['Dense'][1], activation = 'relu'))
    ML.add(Flatten())
    ML.add(Dense(hor, activation = control["CNN"]["ActFun"]))
    ML.compile(loss=control["CNN"]["LossFun"], optimizer=control["CNN"]["Optimizer"]) #, metrics=['mse', 'mae', 'mape', 'cosine']
    ML.summary()
    
    callbacks = _early_stopping_callback(control["CNN"].get("EarlyStopping"))
    ANN_training = ML.fit(train_data, train_target,
                epochs = control["CNN"]["epo_num"], 
                batch_size = control["CNN"]["n_batch"],
                validation_data = (validation_data, validation_target), #fed here the validation data
                # validation_split
                verbose=2, #0-> shows nothing, 1-> shows progress bar, 2-> shows the number of epoch.
                shuffle=False,
                callbacks=callbacks)
    
    return ML, ANN_training,

def train_CNN_LSTM(data, control):
    
    """
    data -> includes ML_DATA, that is the train, validation and test sets, without applying reshapes as a dict of dataframes
    Control_Var -> generic control variables as it brings stuff from LSTM as a dictionary
    pre -> previous number of samples to use in the prediction as an int
    hor -> prediction horizon as an int
    features -> basic feature and complements as a list of strings
    
    returns 
    ML -> the trained LSTM model
    ANN_training -> the history of the fit
    """
    pre=control["PRE"]
    hor=control["H"]

    train_data = data['X_TRAIN']#already a correctly-shaped (samples, pre+1, n_features) array from PreProcessDataset
    validation_data = data['X_VAL']

    train_target = data['Y_TRAIN']#already a correctly-shaped (samples, hor) array from PreProcessDataset
    validation_target = data['Y_VAL']
    
    ##### Designing Neuronal Network #######
    ML = Sequential() #initialize
    ML.add(Conv1D(filters=control["CNN_LSTM"]["filters"], kernel_size=control["CNN_LSTM"]["kernel_size"], 
                  activation=control["CNN_LSTM"]["CNNActFun"], padding = 'causal', input_shape=(train_data.shape[1], train_data.shape[2])))
    if control['CNN_LSTM']['Dense'][0] > 0: #add a dense if the number of neurons is higher than 0
        ML.add(Dense(control['CNN_LSTM']['Dense'][0]))   
    ML.add(MaxPooling1D(pool_size=control["CNN_LSTM"]["pool_size"],  padding='same'))
    if control['CNN_LSTM']['Dense'][1] > 0: #we only add the dense one if the number of neurons is higher than 0
                ML.add(Dense(control['CNN_LSTM']['Dense'][1], activation = 'relu'))
    
    #no Masking layer: PreProcessDataset drops NaN rows outright, and Conv1D/MaxPooling1D don't propagate masks anyway
    ML.add(LSTM(control["CNN_LSTM"]["Neurons"][0], input_shape=(train_data.shape[1], train_data.shape[2]),
                activation = control["CNN_LSTM"]["LSTMActFun"],
                bias_initializer = "zeros", kernel_initializer = "random_uniform",
                return_sequences=True)) #LSTM layer
    ML.add(LSTM(control["CNN_LSTM"]["Neurons"][1], activation = control["CNN_LSTM"]["LSTMActFun"], return_sequences=False)) #LSTM layers
    
    ML.add(Dense(hor, activation = 'sigmoid')) #control["CNN_LSTM"]["LSTMActFun"]))
    ML.compile(loss=control["CNN_LSTM"]["LossFun"], optimizer=control["CNN_LSTM"]["Optimizer"]) #, metrics=['mse', 'mae', 'mape', 'cosine']
    ML.summary()
    
    callbacks = _early_stopping_callback(control["CNN_LSTM"].get("EarlyStopping"))
    ANN_training = ML.fit(train_data, train_target,
                epochs = control["CNN_LSTM"]["epo_num"], 
                batch_size = control["CNN_LSTM"]["n_batch"],
                validation_data = (validation_data, validation_target), #fed here the validation data
                # validation_split
                verbose=2, #0-> shows nothing, 1-> shows progress bar, 2-> shows the number of epoch.
                shuffle=False,
                callbacks=callbacks)
    
    return ML, ANN_training,


def TestMLmodel(control, data, ml, scaler):
    """
    Takes DATA and ML_DATA to test the trained model in the testing set

    Parameters
    ----------
    control : dict
        Control_Var
    data : DataFrame
        SOLETE dataset
    ml : trained ML model
        ML_DATA, the sets cointaining training, validation and testing
    scaler : dict
        if an scaler has been applied it is neccesary to invert it aftewards

    Returns
    -------
    Results : dict of dataframe
        Includes different dataframes:
            -Forecasted: predicted values
            -Observed: measured values (true ones)
            -Persistence: benchmark forecasting method, naive persistence
            -Persistence24: #Variant of persistence using the 24 hours previous value instead of latest available value

    """
    
    print("Testing " + control['MLtype'] + " This can take a while...")    

    predictions = ml.predict(data['X_TEST'])
    
    print("...Done!\n")
    
    
    
    print("Building Results dictionary of Dataframes...")
    #here we undo the scaler transformation to facilitate comparing the results with the orignal dataset
    Results={
        "Forecasted": pd.DataFrame(scaler['Y_data'].inverse_transform(predictions)), 
        "Observed": pd.DataFrame(scaler['Y_data'].inverse_transform(data["Y_TEST"].reshape(predictions.shape))),
        #PERSISTENCE_TEST (built in PreProcessDataset straight from the raw series) is already in physical
        #units, so no inverse_transform here -- avoids assuming Xscaler and Yscaler agree on how they
        #scaled the intrinsic feature, which the old generate_persistence()-based approach relied on.
        "Persistence": pd.DataFrame(np.asarray(data["PERSISTENCE_TEST"]).reshape(predictions.shape)),
        #"Persistence24": #Variant of persistence with 24 hours previous instead of latest available value #TODO
        }
        
    print("...Done\n")
    
    #Save the results
    filename = str(output_path("Results_" + control['MLtype'] + ".h5"))
    print("Saving Results as: ", filename)
    
    i=0 #counter to check first iteration
    for key in Results.keys():
        if i == 0:
            Results[key].to_hdf(filename, index = True, mode= 'w', key = key)
            i=i+1
        else:
            Results[key].to_hdf(filename, index = True, mode= 'a', key = key)
        
    print("...Done\n")
    
    return Results

