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
import numpy as np

import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import load_model


def PrepareMLmodel(control, ml_data):
    """
    Parameters
    ----------
    control : dict
        Control_Var
    ml_data : dict of dataframes
        It contains the training, validation and testing sets

    Returns
    -------
    ML : keras object or sci-kit learn object
        This is the trained model, it depends on which specific ML-method you are requesting

    """
    
    filename = "Trained_" + control['MLtype']
    
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
            plt.savefig('Training_Evaluation_' + control['MLtype'], dpi=500)
            plt.show() 

        if control['trainVSimport'] and control['saveMLmodel']:
            if control['MLtype'] in ['RF', "SVM"]:
                print("Saving Trained Model with name: " + filename)
                joblib.dump(ML, filename)
                
            elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]:
                print("Saving Trained Model with name: " + filename)
                ML.save(filename)
            print("...Done\n")
        else:
            print(control['MLtype'] + " was NOT saved.\n")

    else: #we dont train but import the ML
        if control['MLtype'] in ['RF', "SVM"]:
            filename = filename + ".joblib"
            print("Importing " + filename + "...")
            ML=joblib.load(filename)
            
        elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]: 
            filename = filename + ".h5"
            print("Importing " + filename + "...")
            ML=load_model(filename)
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
    features= control["PossibleFeatures"].copy()
    features.remove(control["IntrinsicFeature"])
    
    train_data = data['X_TRAIN']#already a correctly-shaped (samples, pre+1, n_features) array from PreProcessDataset
    validation_data = data['X_VAL']

    train_target = data['Y_TRAIN']#already a correctly-shaped (samples, hor) array from PreProcessDataset
    validation_target = data['Y_VAL']
    
    ##### Designing Neuronal Network #######
    ML = Sequential() #initialize
    ML.add(Masking(mask_value=999, input_shape=(train_data.shape[1], train_data.shape[2]))) #add the mask so 999 = nan and are not taken into account
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
    
    
    ANN_training = ML.fit(train_data, train_target,
                epochs = control["LSTM"]["epo_num"], 
                batch_size = control["LSTM"]["n_batch"],
                validation_data = (validation_data, validation_target), #fed here the validation data
                # validation_split
                verbose=2, #0-> shows nothing, 1-> shows progress bar, 2-> shows the number of epoch.
                shuffle=False)
                # callbacks=[tbGraph])

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
    features= control["PossibleFeatures"].copy()
    features.remove(control["IntrinsicFeature"])
    
    train_data = data['X_TRAIN']#.values.reshape(len(data['X_TRAIN'].index), pre+1, len(features))
    validation_data = data['X_VAL']#.values.reshape(len(data['X_VAL'].index), pre+1, len(features))
    
    train_target = data['Y_TRAIN']#.values.reshape(len(data['Y_TRAIN'].index),hor)
    validation_target = data['Y_VAL']#.values.reshape(len(data['Y_VAL'].index),hor)
    
    ##### Designing Neuronal Network #######
    ML = Sequential() #initialize
    #ML.add(TimeDistributed(Masking(mask_value=999, input_shape=(pre+1, len(features))))) #add the mask so 999 = nan and are not taken into account
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
    
    
    ANN_training = ML.fit(train_data, train_target,
                epochs = control["CNN"]["epo_num"], 
                batch_size = control["CNN"]["n_batch"],
                validation_data = (validation_data, validation_target), #fed here the validation data
                # validation_split
                verbose=2, #0-> shows nothing, 1-> shows progress bar, 2-> shows the number of epoch.
                shuffle=False)
                # callbacks=[tbGraph])
    
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
    features= control["PossibleFeatures"].copy()
    features.remove(control["IntrinsicFeature"])
    
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
    
    ML.add(Masking(mask_value=999, input_shape=(train_data.shape[1], train_data.shape[2]))) #add the mask so 999 = nan and are not taken into account
    ML.add(LSTM(control["CNN_LSTM"]["Neurons"][0], input_shape=(train_data.shape[1], train_data.shape[2]),
                activation = control["CNN_LSTM"]["LSTMActFun"],
                bias_initializer = "zeros", kernel_initializer = "random_uniform",
                return_sequences=True)) #LSTM layer
    ML.add(LSTM(control["CNN_LSTM"]["Neurons"][1], activation = control["CNN_LSTM"]["LSTMActFun"], return_sequences=False)) #LSTM layers
    
    ML.add(Dense(hor, activation = 'sigmoid')) #control["CNN_LSTM"]["LSTMActFun"]))
    ML.compile(loss=control["CNN_LSTM"]["LossFun"], optimizer=control["CNN_LSTM"]["Optimizer"]) #, metrics=['mse', 'mae', 'mape', 'cosine']
    ML.summary()
    
    ANN_training = ML.fit(train_data, train_target,
                epochs = control["CNN_LSTM"]["epo_num"], 
                batch_size = control["CNN_LSTM"]["n_batch"],
                validation_data = (validation_data, validation_target), #fed here the validation data
                # validation_split
                verbose=2, #0-> shows nothing, 1-> shows progress bar, 2-> shows the number of epoch.
                shuffle=False)
                # callbacks=[tbGraph])
    
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

    if control['MLtype'] in ['RF', "SVM"]:
        predictions=ml.predict(data['X_TEST'])
        persistence = generate_persistence(meas=data["X_TEST"][:,data["xcols"].index(control["IntrinsicFeature"]+"_(t)"),],
                                           hor=control['H'])
            
    elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]:        
        predictions = ml.predict(data['X_TEST'])
        persistence = generate_persistence(meas=data["X_TEST"][:,-1,data["xcols"].index(control["IntrinsicFeature"])],
                                           hor=control['H'])
    
    print("...Done!\n")
    
    
    
    print("Building Results dictionary of Dataframes...")
    #here we undo the scaler transformation to facilitate comparing the results with the orignal dataset
    Results={
        "Forecasted": pd.DataFrame(scaler['Y_data'].inverse_transform(predictions)), 
        "Observed": pd.DataFrame(scaler['Y_data'].inverse_transform(data["Y_TEST"].reshape(predictions.shape))),
        "Persistence": pd.DataFrame(scaler['Y_data'].inverse_transform(persistence)),
        #"Persistence24": #Variant of persistence with 24 hours previous instead of latest available value #TODO
        }
        
    print("...Done\n")
    
    #Save the results
    filename = "Results_" + control['MLtype'] + ".h5"
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

def generate_persistence(meas, hor,):
    """
    Short function that creates the naive persistance forecasters

    Parameters
    ----------
    meas : numpy array
        True values, observations to be repeated
    hor : int
        Horizon length in number of samples

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the naive Persistence forecaster

    """
    
    meas = meas.reshape(len(meas),1) #it is neccessary to reshape otherwise 
    #it doesnt allow to concatenate on the columns axis
    df = meas.copy()
    for i in range(1, hor): #it starts in 1 because it has the first copy already inside
        df = np.concatenate((df, meas), axis=1)
    
    return df


