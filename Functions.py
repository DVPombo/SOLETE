# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:35:08 2021

@author: Daniel Vázquez Pombo
email: dvapo@elektro.dtu.dk
LinkedIn: https://www.linkedin.com/in/dvp/
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo

The purpose of this script is to give you, dear user, a gentle hand and, hopefully, kick start your work with the Solete dataset.
It should run without errors simply by placing all the files in the same location.
This file only contains functions called by the rest of the scripts.

The lincensing of this work is pretty chill, just give credit: https://creativecommons.org/licenses/by/4.0/

Dependencies: Python 3.8.10, and Pandas 1.2.4
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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


def ExpandSOLETE(data, info, Control_Var):
    """
    
    Parameters
    ----------
    data : DataFrame
        Variable including all the data from the Solete dataset
    info : list
        Contains PVinfo and WTinfo which are dicts
    Control_Var : dict
        Holds information regarding what to do
    
    Returns
    -------
    Adds columns to data with new metrics. Some from the PV performance model [1], others from potentially useful metrics.

    """
    ncol=len(data.columns)       
        
    print("Expanding SOLETE with King's PV Performance Model")
    data['Pac'], data['Pdc'], data['TempModule'], data['TempCell'] = PV_Performance_Model(data, info[0])
    print("Cleaning noise and curtailment from active power production")
    data['P_Solar[kW]'] =  np.where(data['Pac'] >= 1.5*data['P_Solar[kW]'],
                                    data['Pac'], data['P_Solar[kW]'])
    print("Smoothing zeros")
    data['P_Solar[kW]'] =  np.where(data['P_Solar[kW]'] <= 0.001,
                                      0, data['P_Solar[kW]'])
    data['Pac'] =  np.where(data['Pac'] <= 0.001,
                                      0, data['Pac'])
    
    print("\nAdding new Types with codes: " + str(Control_Var['PossibleFeatures']))
    
    
    if 'HoursOfDay' in Control_Var['PossibleFeatures']: #time of the day hours
        data['HoursOfDay'] =  data.index.hour
    if 'MeanPrevH' in Control_Var['PossibleFeatures']: #mean previous horizon
        data['MeanPrevH'] =  data[ Control_Var['IntrinsicFeature']].rolling(Control_Var['H']).mean()
    if 'StdPrevH' in Control_Var['PossibleFeatures']: #std previous horizon
        data['StdPrevH'] =  data[Control_Var['IntrinsicFeature']].rolling(Control_Var['H']).std()
    if 'MeanWindSpeedPrevH' in Control_Var['PossibleFeatures']: #wind speed mean of the previous horizon
        data['MeanWindSpeedPrevH'] =  data['WIND_SPEED[m1s]'].rolling(Control_Var['H']).mean()
    if 'StdWindSpeedPrevH' in Control_Var['PossibleFeatures']: #wind speed std of the previous horizon
        data['StdWindSpeedPrevH'] =  data['WIND_SPEED[m1s]'].rolling(Control_Var['H']).std()
    

    print("\nSOLETE has been successfully expanded from:", ncol, "to:", len(data.columns), "features.\n\n")
    
    
    pass


def PV_Performance_Model(data, PVinfo, colirra='POA Irr[kW1m2]'):
    """
    Parameters
    ----------
    data : DataFrame
        Variable including all the data from the Solete dataset
    PVinfo : dict
        A bunch of parameters extracted from the datasheet and other supporting documents
        Check function: import_PV_WT_data for further details
    colirra : string
        holds Epoa, that is the irradiance in the plane of the array.
        If you reuse this code, make sure you are feeding Epoa and not GHI
        The default is 'POA Irr[kW1m2]'.

    Returns
    -------
    DataFrames
        Pac, Pdc, Tm, and Tc. [1]

    """
    
    
    # Obtains the expected solar production based on irradiance, temperature, pv parameters, etc
    DATA_PV = pd.DataFrame({'Pmp_stc' : PVinfo["Pmp_stc"],
                            'ganma_mp' : PVinfo['ganma_mp'],
                            'Ns': PVinfo['Ns'],
                            'Np': PVinfo['Np'],
                            'a' : PVinfo['a'],
                            'b' : PVinfo['b'],
                            'D_T' : PVinfo['D_T'],
                            'eff_P' : PVinfo['eff_P'],
                            'eff_%' : PVinfo['eff_%'],
                            }, 
                           index = PVinfo["index"])
    
    DATA_PV['eff_max_%'] = [max(DATA_PV['eff_%'].loc['A']), max(DATA_PV['eff_%'].loc['B'])] #maximum inverter efficiency in %
    DATA_PV['eff_max_P'] = [max(DATA_PV['eff_P'].loc['A']), max(DATA_PV['eff_P'].loc['B'])] #W maximum power output of the inverter
    
    Results = pd.DataFrame(index = data.index)
    
    for pv in DATA_PV.index:
        #Temperature Module
        Results['Tm_' + pv] = data['TEMPERATURE[degC]'] + data[colirra]*1000 *np.exp(DATA_PV.loc[pv,'a']+DATA_PV.loc[pv,'b']*data['WIND_SPEED[m1s]']) 
        #Temperature Cell
        Results['Tc_' + pv] = Results['Tm_' + pv] + data[colirra]*1000/PVinfo["Estc"] * DATA_PV.loc[pv,'D_T']
        #power produced in one single pannel
        Results['Pmp_panel_' + pv] = data[colirra]*1000/PVinfo["Estc"] * DATA_PV.loc[pv, 'Pmp_stc'] * (1+DATA_PV.loc[pv, 'ganma_mp'] * (Results['Tc_' + pv] - PVinfo["Tstc"]) )
        #power produced by all the panels in the array
        Results['Pmp_array_' + pv] = DATA_PV.loc[pv, 'Ns'] * DATA_PV.loc[pv, 'Np'] * Results['Pmp_panel_' + pv]
        #efficiency of the inverter corresponding to the instantaneous power output
        Results['eff_inv_' + pv] =  np.interp(Results['Pmp_array_' + pv], DATA_PV.loc[pv, 'eff_P'], DATA_PV.loc[pv, 'eff_%'], left=0)/100
        
        
        Results['Pac_' + pv] =  DATA_PV.loc[pv, 'eff_max_%']/100 * Results['Pmp_array_' + pv]
        Results[Results['Pac_' + pv]>DATA_PV.loc[pv, 'eff_max_P']]=DATA_PV.loc[pv, 'eff_max_P'] #If any of the Pac is > than the maximum capacity of the inverter 
        # then use the max capacity of the inverter
        Results[Results['Pac_' + pv]<0]=0
        
    return Results[['Pac_A', 'Pac_B']].sum(axis=1)/1000, Results[['Pmp_array_A', 'Pmp_array_B']].sum(axis=1)/1000, Results[['Tm_A', 'Tm_B']].mean(axis=1), Results[['Tc_A', 'Tc_B']].mean(axis=1)


def TimePeriods(data, control):
    """
    A function that will split the time series to input and output for training 
    of the forecast problem with supervised learning

    Parameters
    ----------
    data : DataFrame
        Variable including all the data from the Solete dataset
    control : dict
        Control_Var.

    Returns
    -------
    dik : dict of DataFrames
        cotains the train and testing sets for RF and SVM
        or the train, validation and testing for ANN
        
    Arguments
    ---------
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        base: Is the basic variable that will be shifted back and forward in time, e.g. Pestimated
        additions: are the other variables that will tag along base to complete the dataset
        train_val_test = division of timestamps in the three blocks

    """
    n_in=control["PRE"]
    n_out=control["H"]
    base=control["IntrinsicFeature"]
    additions= control["PossibleFeatures"].copy()
    additions.remove(control["IntrinsicFeature"])
    train_val_test = control['Train_Val_Test']
    
    if control['MLtype'] in ['RF', 'SVM']: 
        data.fillna(0, inplace=True)
           
        BASE = series_to_forecast(data[base], n_in, n_out, dropnan=False)
        
        col_loc_base = []
        for addition in additions:
            BASE[addition] = data[addition]
            col_loc_base.append(BASE.columns.get_loc(addition))
        
        
        BASE.dropna(inplace=True)
            
            
        X_COLS = [*range(0,n_in+1)]+col_loc_base  #because they are the PRE+t0 current sample
        Y_COLS = [*range(n_in+1,n_in+n_out+1)]
        
        X=BASE.iloc[:, X_COLS]
        Y=BASE.iloc[:, Y_COLS]    
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=train_val_test[-1]/100, shuffle=False, random_state=None)
        
        Scaler = {
                  'X_data' : 1,
                  'Y_data' : 1,
                  }
    
        
        ML_DATA = {
            "X_TRAIN": X_TRAIN.sort_index(),
            "X_TEST": X_TEST.sort_index(),
            "Y_TRAIN": Y_TRAIN.sort_index(),
            # "Y_TEST": Y_TEST.sort_index(),
            "Y_TEST_abs": Y_TEST.sort_index(),
            }
    
    
    elif control['MLtype'] in ['LSTM', 'CNN', 'CNN_LSTM']:
        Xscaler = MinMaxScaler(feature_range=(0, 1)) #initialise the scaler 
        Yscaler = MinMaxScaler(feature_range=(0, 1)) #initialise the scaler   
            
        X = data[additions] 
        Y = data[base]
                   
        X = series_to_forecast(X, n_in, 0, dropnan=False)#.drop(X.index[-(n_out):])
        Y = series_to_forecast(Y, 0, n_out, dropnan=False).drop(base+'_(t)', axis=1)#.drop(Y.index[:(n_in)])
           
        X_TRAIN, X_VAL_TEST, Y_TRAIN, Y_VAL_TEST = train_test_split(X, Y, train_size=train_val_test[0]/100, shuffle=False, random_state=None)
        
        del X, Y
        
        X_VAL, X_TEST, Y_VAL, Y_TEST = train_test_split(X_VAL_TEST, Y_VAL_TEST, train_size=train_val_test[1]/(100-train_val_test[0]), shuffle=False, random_state=None)
            
        del X_VAL_TEST, Y_VAL_TEST
            
        #apply scaler to keep all values between 0 and 1
        X_TRAIN = pd.DataFrame(Xscaler.fit_transform(X_TRAIN), index=X_TRAIN.index, columns = X_TRAIN.columns)
        X_VAL = pd.DataFrame(Xscaler.transform(X_VAL), index=X_VAL.index, columns = X_VAL.columns)
        X_TEST = pd.DataFrame(Xscaler.transform(X_TEST), index=X_TEST.index, columns = X_TEST.columns)
        
        Y_TRAIN = pd.DataFrame(Yscaler.fit_transform(Y_TRAIN), index=Y_TRAIN.index, columns = Y_TRAIN.columns)
        Y_VAL = pd.DataFrame(Yscaler.transform(Y_VAL), index=Y_VAL.index, columns = Y_VAL.columns)
        Y_TEST_abs = pd.DataFrame(Y_TEST, index=Y_TEST.index, columns = Y_TEST.columns)
        Y_TEST = pd.DataFrame(Yscaler.transform(Y_TEST), index=Y_TEST.index, columns = Y_TEST.columns)
        
        
        # apply masking as to substitute a complete row by 999 if any of its values is NAN (aka missing value)
        X_TRAIN.mask(X_TRAIN.isna().any(axis=1), other=999, inplace=True)
        X_VAL.mask(X_VAL.isna().any(axis=1), other=999, inplace=True)
        X_TEST.mask(X_TEST.isna().any(axis=1), other=999, inplace=True)
        
        Y_TRAIN.mask(Y_TRAIN.isna().any(axis=1), other=999, inplace=True)
        Y_VAL.mask(Y_VAL.isna().any(axis=1), other=999, inplace=True)
        Y_TEST.mask(Y_TEST.isna().any(axis=1), other=999, inplace=True)
        Y_TEST_abs.mask(Y_TEST_abs.isna().any(axis=1), other=999, inplace=True)
        
        
        Scaler = {
                  'X_data' : Xscaler,
                  'Y_data' : Yscaler,
                  }
       
        ML_DATA = {
            "X_TRAIN": X_TRAIN.sort_index(),
            "X_VAL": X_VAL.sort_index(),
            "X_TEST": X_TEST.sort_index().drop(X_TEST.tail(n_out).index, axis=0),
            "Y_TRAIN": Y_TRAIN.sort_index(),
            "Y_VAL": Y_VAL.sort_index(),
            "Y_TEST": Y_TEST.sort_index().drop(Y_TEST.tail(n_out).index, axis=0),
            "Y_TEST_abs": Y_TEST_abs.sort_index().drop(Y_TEST.tail(n_out).index, axis=0),
            }
            
    else:
        print("\n\n\n WARNING: Your ML method is not supported by the 'TimePeriods' function.\n\n")
    
    return ML_DATA, Scaler


def series_to_forecast(data, n_in, n_out, dropnan=True):
    """
    A function that will split the time series to input and output for training 
    of the forecast problem with supervised learning
    Arguments:
        data: Sequence of observations as a list, NumPy array or pandas series
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    
    # n_vars = data.shape[1] 
    df = pd.DataFrame(data)
    cols, names = list(), list()
    COLUMNS = df.columns
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(col_name + '_(t-%d)' % (i)) for col_name in COLUMNS]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out+1):
        cols.append(df.shift(-i))

        if i == 0:
            names += [(col_name + '_(t)') for col_name in COLUMNS]
        else:
            names += [(col_name + '_(t+%d)' % (i)) for col_name in COLUMNS]
    # put it all together (aggregate)
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



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
                print("Saving Trained Model with name:" + filename)
                joblib.dump(ML, filename)
                
            elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]:
                print("Saving Trained Model with name:" + filename)
                ML.save(filename)
            print("...Done")
        else:
            print(control['MLtype'] + " was NOT saved.")

    else: #we dont train but import the ML
        if control['MLtype'] in ['RF', "SVM"]:
            filename = filename + ".joblib"
            print("Importing " + filename + "...")
            ML=joblib.load(filename)
            
        elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]: 
            filename = filename + ".h5"
            print("Importing " + filename + "...")
            ML=load_model(filename)
        print("...Done")
    
        
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
    
    train_data = data['X_TRAIN'].values.reshape(len(data['X_TRAIN'].index), pre+1, len(features))
    validation_data = data['X_VAL'].values.reshape(len(data['X_VAL'].index), pre+1, len(features))
    
    train_target = data['Y_TRAIN'].values.reshape(len(data['Y_TRAIN'].index),hor)
    validation_target = data['Y_VAL'].values.reshape(len(data['Y_VAL'].index),hor)
    
    ##### Designing Neuronal Network #######
    ML = Sequential() #initialize
    ML.add(Masking(mask_value=999, input_shape=(pre+1, len(features)))) #add the mask so 999 = nan and are not taken into account
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
    # pass
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
    
    train_data = data['X_TRAIN'].values.reshape(len(data['X_TRAIN'].index), pre+1, len(features))
    validation_data = data['X_VAL'].values.reshape(len(data['X_VAL'].index), pre+1, len(features))
    
    train_target = data['Y_TRAIN'].values.reshape(len(data['Y_TRAIN'].index),hor)
    validation_target = data['Y_VAL'].values.reshape(len(data['Y_VAL'].index),hor)
    
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
    # pass
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
    
    train_data = data['X_TRAIN'].values.reshape(len(data['X_TRAIN'].index), pre+1, len(features))
    validation_data = data['X_VAL'].values.reshape(len(data['X_VAL'].index), pre+1, len(features))
    
    train_target = data['Y_TRAIN'].values.reshape(len(data['Y_TRAIN'].index),hor)
    validation_target = data['Y_VAL'].values.reshape(len(data['Y_VAL'].index),hor)
    
    ##### Designing Neuronal Network #######
    ML = Sequential() #initialize
    ML.add(Conv1D(filters=control["CNN_LSTM"]["filters"], kernel_size=control["CNN_LSTM"]["kernel_size"], 
                  activation=control["CNN_LSTM"]["CNNActFun"], padding = 'causal', input_shape=(train_data.shape[1], train_data.shape[2])))
    if control['CNN_LSTM']['Dense'][0] > 0: #add a dense if the number of neurons is higher than 0
        ML.add(Dense(control['CNN_LSTM']['Dense'][0]))   
    ML.add(MaxPooling1D(pool_size=control["CNN_LSTM"]["pool_size"],  padding='same'))
    if control['CNN_LSTM']['Dense'][1] > 0: #we only add the dense one if the number of neurons is higher than 0
                ML.add(Dense(control['CNN_LSTM']['Dense'][1], activation = 'relu'))
    
    ML.add(Masking(mask_value=999, input_shape=(pre+1, len(features)))) #add the mask so 999 = nan and are not taken into account
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
    # pass
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
    ml : dict of dataframes
        ML_DATA, the sets cointaining training, validation and testing
    scaler : dict
        if an scaler has been applied it is neccesary to invert it aftewards

    Returns
    -------
    predictions : dataframe
        Predicted values

    """
    
    print("Testing " + control['MLtype'] + " This can take a while...")    

    if control['MLtype'] in ['RF', "SVM"]:
        predictions=ml.predict(data['X_TEST'])
            
    elif control['MLtype'] in ["LSTM", "CNN", "CNN_LSTM"]: 
        features= control["PossibleFeatures"].copy()
        features.remove(control["IntrinsicFeature"])
        
        test_data = data['X_TEST'].values.reshape(len(data['X_TEST'].index), control["PRE"]+1, len(features))
        predictions = ml.predict(test_data)
    
        predictions = scaler['Y_data'].inverse_transform(predictions)

    print("...Done")
    return predictions


def get_results(control, data, ml_data, predictions):
    """
    Builds a DataFrame with the results. It is a very inefficient function.
    Each set of three columns corresponds to one set of predictions, observations and timestamps.

    Parameters
    ----------
    control : dict
        Control_Var
    data : DataFrame
        SOLETE dataset
    ml_data : dict of dataframes
        ML_DATA, the sets cointaining training, validation and testing.
    predictions : TYPE
        DESCRIPTION.

    Returns
    -------
    RESULTS : dataframe
        Predicted values.

    """
    print("Building Results Dataframe...")
    RESULTS=pd.DataFrame([], index=range(0,control["H"]+1))
    for i in range(0,len(ml_data['X_TEST'].index)):
        t0 = ml_data['Y_TEST_abs'].index[i]-datetime.timedelta(seconds=3600)
        RESULTS['Forecasted_' + str(i)] = np.insert(predictions[i], 0, np.nan, axis=0) #retrieves predictions and adds a nan in the t0
        RESULTS['Observed_' + str(i)] = np.insert(ml_data['Y_TEST_abs'].iloc[i].values, 0, data.loc[t0][control["IntrinsicFeature"]], axis=0) 
        RESULTS['Time_' + str(i)] = pd.date_range(start=t0, periods = control["H"]+1, freq = '3600s' )
    print("...Done")
    
    #Save the results
    filename = "Results_" + control['MLtype'] + ".h5"
    print("Saving Results as: ", filename)
    RESULTS.to_hdf(filename, index = True, mode= 'w', key = 'DATA')
    print("...Done")

    return RESULTS

def post_process(control, RESULTS, data):
    """
    Computes errors and plots RMSE.

    Parameters
    ----------
    control : dict
        Control_Var
    RESULTS : DataFrame
        The horrible DataFrame containing three columns per forecasted horizon.
    data : DataFrame
        SOLETE Dataset

    Returns
    -------
    RMSE : DataFrame
        Contains the results in terms of RMSE

    """
    
    print("Post-processing results...")
    ERROR =pd.DataFrame([], index=RESULTS.index[1:], columns = range(0,int(len(RESULTS.columns)/3)))
    Persistence = pd.DataFrame([], index=ERROR.index)
    Persistence24 = pd.DataFrame([], index=ERROR.index)
    RMSE =pd.DataFrame([], index=ERROR.index)
        
    for i in range(0,int(len(RESULTS.columns)/3)-1):
        valor = data.loc[(RESULTS.loc[0,'Time_'+str(i)]-datetime.timedelta(hours=24)), control["IntrinsicFeature"]]
        
        for t,v in zip(RESULTS.loc[1:,'Time_'+str(i)].index, RESULTS.loc[1:,'Time_'+str(i)]):
                        
            if RESULTS.loc[t,'Time_'+str(i)].hour not in range(7,19):# and RESULTS.loc[t,'Observed_'+str(i)] == 0:
                RESULTS.loc[t,'Observed_'+str(i)] = np.nan
                
        
        Persistence[i] =  RESULTS.loc[1:,'Observed_'+str(i)] - RESULTS.loc[0,'Observed_'+str(i)]
        ERROR[i] = RESULTS['Observed_'+str(i)]-RESULTS['Forecasted_'+str(i)]
        
        
        Persistence24[i] = RESULTS.loc[1:,'Observed_'+str(i)] - valor
    
    print("Computing RMSE...")
    RMSE = np.sqrt((ERROR**2).mean(axis=1)).to_frame(name='Forecaster')
    RMSE['Persistence'] = np.sqrt((Persistence**2).mean(axis=1))
    RMSE['Persistence24'] = np.sqrt((Persistence24**2).mean(axis=1))
    
    fig = plt.figure()
    plt.plot(RMSE)
    plt.grid()
    plt.xlim((RMSE.index[0], RMSE.index[-1]))
    plt.ylim((0, max(RMSE.max())*1.1)) 
    plt.ylabel( "RMSE" )
    plt.xlabel( "Time Horizon" )
    plt.title("Avgs: For=" + str(round(RMSE.Forecaster.mean(), 3)) + " Per = " + str(round(RMSE.Persistence.mean(), 3))\
              +" Per24 = "+ str(round(RMSE.Persistence24.mean(), 3)))
    plt.legend(RMSE.columns)
    
    filename = "RMSE_" + control["MLtype"]
    print("Saving RMSE plot as: ", filename)
    plt.savefig(filename, dpi=500)
    print("...Done")
    
    print("\n\nThe End!")
    
    return RMSE

































































































