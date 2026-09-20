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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .physics import PV_Performance_Model, Rincon_Pombo_ThermodynamicModel
from .qc import (
    apply_qc_flags,
    build_raw_value_qc_rules,
    build_substitution_qc_rule,
    QC_VALID,
    QC_FLAG_PRECEDENCE,
)


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
    Adds columns to data with new metrics. Some from the PV performance model [2, 3, 4], 
    others from potentially useful metrics.

    """
    # ncol=len(data.columns)   
    list_expansion=Control_Var['OriginalFeatures'].copy()
    all_expansions=Control_Var['PossibleFeatures'].copy()
    
        
    print("Expanding SOLETE with King's PV Performance Model")
    data['Pac'], data['Pdc'], data['TempModule'], data['TempCell'] = PV_Performance_Model(data, info[0])
    list_expansion.append('Pac')
    list_expansion.append('Pdc')
    list_expansion.append('TempModule')
    list_expansion.append('TempCell')
    
    print("    Cleaning noise and curtailment from active power production")
    #flags every row where the measured P_Solar[kW] is about to be silently replaced by King's
    #model estimate (Pac), so the substitution is traceable downstream instead of being invisible.
    data['P_Solar_model_substituted'] = data['Pac'] >= 1.5*data['P_Solar[kW]']
    list_expansion.append('P_Solar_model_substituted')
    #Phase 2: fold the substitution boolean into the unified QC schema
    #(finding #6, QC_SCHEMA.md section 3) without changing the substitution
    #logic above -- same boolean, just also exposed as P_Solar[kW]_qc.
    _, qc_counts = apply_qc_flags(data, [build_substitution_qc_rule()])
    list_expansion.append('P_Solar[kW]_qc')
    print("    QC flag (substitution):", qc_counts)
    data['P_Solar[kW]'] =  np.where(data['P_Solar_model_substituted'],
                                    data['Pac'], data['P_Solar[kW]'])
    print("    Smoothing zeros")
    data['P_Solar[kW]'] =  np.where(data['P_Solar[kW]'] <= 0.001,
                                      0, data['P_Solar[kW]'])
    data['Pac'] =  np.where(data['Pac'] <= 0.001,
                                      0, data['Pac'])

    print("    Adding derived hybrid wind+solar column (Phase 6, Task 6.1)")
    #P_hybrid[kW] = P_Solar[kW] + P_Gaia[kW], computed AFTER the substitution/
    #zero-smoothing above so it uses the same finalized P_Solar[kW] everything
    #else in this dataset sees -- not the raw pre-cleaning sensor value.
    #
    #IMPORTANT CAVEAT (see splits/README.md and KNOWN_ISSUES.md finding #10):
    #P_Gaia[kW] is genuinely nonzero on only ~0.4-0.8% of rows across the real
    #record (depending on split block) -- this column is overwhelmingly
    #P_Solar[kW] in practice, not a balanced wind+solar hybrid signal. That
    #limitation is documented at the data level, not hidden here.
    data['P_hybrid[kW]'] = data['P_Solar[kW]'] + data['P_Gaia[kW]']
    list_expansion.append('P_hybrid[kW]')

    #QC inheritance (Task 6.1, QC_SCHEMA.md section 8): P_hybrid[kW]_qc takes
    #whichever constituent's flag is higher-precedence per QC_FLAG_PRECEDENCE
    #(same tie-break apply_qc_flags itself uses), tagged in a companion
    #P_hybrid[kW]_qc_source column so a reader can tell which constituent (or
    #neither) produced the flag. P_Gaia[kW] has no QC detection rule of its own
    #today -- checked against build_raw_value_qc_rules() / QC_SCHEMA.md section
    #3, neither exists -- so in practice this currently only ever reflects
    #P_Solar[kW]_qc. The combination is still written generically against
    #whatever '<constituent>_qc' columns exist, so it starts covering wind too
    #the moment a wind QC rule is added, with no further code change here.
    solar_qc = data['P_Solar[kW]_qc'].to_numpy()
    if 'P_Gaia[kW]_qc' in data.columns:
        wind_qc = data['P_Gaia[kW]_qc'].to_numpy()
    else:
        wind_qc = np.full(len(data), QC_VALID)

    def _qc_rank(flag):
        return (QC_FLAG_PRECEDENCE.index(flag)
                if flag in QC_FLAG_PRECEDENCE else len(QC_FLAG_PRECEDENCE))
    rank = np.vectorize(_qc_rank)
    solar_rank, wind_rank = rank(solar_qc), rank(wind_qc)

    solar_wins = (solar_qc != QC_VALID) & (solar_rank <= wind_rank)
    wind_wins = (wind_qc != QC_VALID) & ~solar_wins

    data['P_hybrid[kW]_qc'] = np.where(
        solar_wins, solar_qc, np.where(wind_wins, wind_qc, QC_VALID))
    data['P_hybrid[kW]_qc_source'] = np.where(
        solar_wins, 'P_Solar[kW]', np.where(wind_wins, 'P_Gaia[kW]', 'none'))
    list_expansion.append('P_hybrid[kW]_qc')
    list_expansion.append('P_hybrid[kW]_qc_source')
    print(f"    P_hybrid[kW] QC: "
          f"{int((data['P_hybrid[kW]_qc'] != QC_VALID).sum())} rows flagged "
          f"(inherited from constituents)")

    if 'TempModule_RP' in Control_Var['PossibleFeatures']: #advanced thermodynamic model
        data['TempModule_RP'] = Rincon_Pombo_ThermodynamicModel(data, info[0])
        list_expansion.append('TempModule_RP')
    
    #Phase 2: also register the raw-value QC columns computed earlier in
    #import_SOLETE_data() (before ExpandSOLETE ran), so the "features added"
    #accounting below reflects them too, same treatment as Pac/Pdc/etc.
    list_expansion.extend(c for c in data.columns
                           if c.endswith('_qc') and c not in list_expansion)

    for expansion in list_expansion: #this is than simply to print a nice statement about which types are added
        if expansion in all_expansions: all_expansions.remove(expansion)
    
    print("\nAdding new Types with codes: " + str(all_expansions))
    
    
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
    

    print("\nSOLETE has been successfully expanded from:", len(Control_Var['OriginalFeatures']), "to:", len(data.columns), "features.\n\n")
    
    
    pass


def PreProcessDataset(data, control):
    """
    A function that does two things:
        1-It adapts the time series to a forecasting problem with supervised 
        learning. This is done by dividing the main dataset into training, 
        validation, and testing subsets.
        2-Summons and trains a scaler according to user input. This scaler helps
        the learning process as it keeps all values within the same range.

    Parameters
    ----------
    data : DataFrame
        Variable including all the data from the SOLETE dataset
    control : dict
        Control_Var.

    Returns
    -------
    ML_DATA : dict of DataFrames
        cotains the train and testing sets for RF and SVM
        or the train, validation and testing sets for ANN
        
    Scaler : dict
        data of the scaling method applied to the data. This scaler is used 
        later by other functions in order to undo the transformation, thus 
        recovering the actual values.
        
    Arguments
    ---------
        n_var_in: Number of variables going into the ML model
        n_var_out: Number of variables predicted/outputed by the ML model
        base: Is the basic or intrinsic variable that will be shifted back and forward in time.
        additions: are the other variables that will tag along base to complete the dataset
        train_val_test = division of timestamps in the three blocks

    """
    #we define this dummy variables simply to ease the reading of the code
    base=control["IntrinsicFeature"]
    additions= control["PossibleFeatures"]
    H=control["H"] #number of samples of each test, corresponds to time
    PRE = control["PRE"] #number of previous samples
    train_val_test = control['Train_Val_Test'] #spliting ratio of the dataset
    n_var_in = len(additions) #number of variables going into the model    
    n_var_out = len(additions) #number of variables predicted by the model

    if type(base) == str:
        n_var_out = 1
    else:
        n_var_out = len(base)
         
    X = data[additions] #input data
    Y = pd.DataFrame(data[base]) #output data
    
    #Scaler selection        
    if control['Scaler'] == 'MinMax01':
        #scales from 0 to 1 each feature independently
        Xscaler = MinMaxScaler(feature_range=(0, 1)) #initialise the scaler 
        Yscaler = MinMaxScaler(feature_range=(0, 1)) #initialise the scaler 
    elif control['Scaler'] == 'MinMax11':
        #scales from -1 to 1 each feature independently
        Xscaler = MinMaxScaler(feature_range=(-1, 1)) #initialise the scaler 
        Yscaler = MinMaxScaler(feature_range=(-1, 1)) #initialise the scaler 
    elif control['Scaler'] == 'Standard':
        #Standardize features by removing the mean and scaling to unit variance
        #might behave badly if each features does look like standard normally 
        #distributed data: Gaussian with zero mean and unit variance.
        Xscaler = StandardScaler() #initialise the scaler 
        Yscaler = StandardScaler() #initialise the scaler 

    X.reset_index(inplace=True)  # (samples, PRE+1, n_var_in)
    Y.reset_index(inplace=True)  # (samples, H, n_var_out)

    # with this loop we append nans in the rows we need to complete the first PRE
    for i in range(0, int(np.ceil(X.shape[0] / (PRE + 1)) * (PRE + 1)) - X.shape[0]):
        X = pd.concat([X, pd.DataFrame([np.nan])], axis=0, ignore_index=True)
        X = X.drop(0, axis=1)

    # with this loop we append nans in the rows we need to complete the last H
    for i in range(0, int(np.ceil(Y.shape[0] / H) * H) - Y.shape[0]):
        Y = pd.concat([Y, pd.DataFrame([np.nan])], axis=0, ignore_index=True)
        Y = Y.drop(0, axis=1)

    # either do this or drop the index column
    X.set_index('index', inplace=True)
    Y.set_index('index', inplace=True)
    
    #the column names are needed to id what are the input and output var names
    xcols = X.columns
    ycols = Y.columns
    
    X_dict = {} #convert time series data into supervised learning compatible
    for col in X.columns: #for the input is based on the number of previous samples
        X_dict[col] = X[col]
        X_dict[col] = series_to_forecast(X_dict[col], PRE, 0, dropnan=False)
        if control['MLtype'] in ['LSTM', 'CNN', 'CNN_LSTM']:
            #in the case of ANN it is neccessary to use 3D vectors
            X_dict[col] = np.ravel(X_dict[col])
            X_dict[col] = X_dict[col].reshape(int(X.shape[0]), PRE+1, n_var_out)  # (samples, PRE+1, n_variables_in)
    
    Y_dict = {} #convert time series data into supervised learning compatible
    for col in Y.columns: #for the output is based on the horizon length
        Y_dict[col] = Y[col]
        Y_dict[col] = series_to_forecast(Y_dict[col], 0, H, dropnan=False).drop(col + '_(t)', axis=1)
        if control['MLtype'] in ['LSTM', 'CNN', 'CNN_LSTM']:
            #in the case of ANN it is neccessary to use 3D vectors
            Y_dict[col] = np.ravel(Y_dict[col])
            Y_dict[col] = Y_dict[col].reshape(int(Y.shape[0]), H, n_var_out)  # (samples, H, n_variables_predicted)
        
        
        #There are a number of operations diverging from RF and SVM compared to ANN
    if control['MLtype'] in ['RF', 'SVM']:        
        
        # First we concatenate all arrays by the number of variables as columns
        X = pd.concat([X_dict[x] for x in X_dict], axis=1)
        Y = pd.concat([Y_dict[x] for x in Y_dict], axis=1)
        
        #the column names must be updated 
        xcols = X.columns
        ycols = Y.columns
        
        #Then we contact the arrays into a single DataFrame in order to remove al rows with nans
        #X and Y are reset to a unique positional index first: the row order between
        #them is already correct (both were built by shifting the same underlying
        #time series), but their original DatetimeIndex can carry duplicate
        #timestamps (a known SOLETE data quirk) and the padding step upstream can
        #append multiple NaN-labelled rows, either of which pandas now refuses to
        #align/reindex during concat.
        XY=pd.concat([X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1).dropna(axis=0, how='any')
        
        del X, Y #done to release memory
        
        # split the dataset into training and testing. In general, ensemble methods do not require validation set (unlike ANN)
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(XY[xcols], XY[ycols], test_size=train_val_test[-1] / 100, shuffle=False,
                                                            random_state=None)
        del XY #done to release memory

        # apply the scaler: note that we fit/train it into the training set only
        #and then we apply it on the remaining sets. If we were to fit_transform 
        #on the whole dataset, we would be introducing informantion from the test
        #into the training, hence, introducing bias.
        X_TRAIN = Xscaler.fit_transform(X_TRAIN) 
        X_TEST = Xscaler.transform(X_TEST) 
        
        #same reasoning for the outputs
        Y_TRAIN = Yscaler.fit_transform(Y_TRAIN)
        Y_TEST = Yscaler.transform(Y_TEST)

        #these are the two variables to be output by this function
        Scaler = {
            'X_data': Xscaler,
            'Y_data': Yscaler,
        }

        ML_DATA = {
            "X_TRAIN": X_TRAIN,
            "X_TEST": X_TEST,
            "Y_TRAIN": Y_TRAIN,
            "Y_TEST": Y_TEST,
            "xcols": list(xcols),
            "ycols": list(ycols),
        }

    elif control['MLtype'] in ['LSTM', 'CNN', 'CNN_LSTM']:
        
        # First we concatenate all arrays by the number of variables as columns
        X = np.concatenate([X_dict[x] for x in X_dict], 2)
        Y = np.concatenate([Y_dict[x] for x in Y_dict], 2)
                
        if H > PRE:
            #Now we remove the nans from the begining of the dataset (caused by the number previous samples)
            X= X[PRE:-PRE-1,:,:]
            Y = Y[PRE:-PRE - 1, :, :]
            
            #Then, we do the same with the nans in the end of the dataset (caused by horizon)
            X = X[0:-(H), :, :]
            Y=Y[0:-(H-PRE)-1,:,:]
        elif PRE > H:
            X= X[PRE:-PRE-1,:,:]
            Y = Y[PRE:-PRE-1, :, :]
            
        #match lenght of arrays
        dif = X.shape[0]-Y.shape[0]
        if dif > 0:
            X = X[0:-dif, :, :]
        elif dif < 0:
            Y = Y[0:dif, :, :]
            
        #remove nans
        id_nans=np.argwhere(np.isnan(X))
        if id_nans.any():
            id_nans=np.unique(id_nans[:,0])
            X=np.delete(X, id_nans, axis=0)
            Y=np.delete(Y, id_nans, axis=0)
        
        id_nans=np.argwhere(np.isnan(Y))
        if id_nans.any():
            id_nans=np.unique(id_nans[:,0])
            X=np.delete(X, id_nans, axis=0)
            Y=np.delete(Y, id_nans, axis=0)
        
        #again, we split the dataset in training, validation, and testing. We use a scikit learn function
        #thus we must use it twice in order to split it as we want it
        X_TRAIN, X_VAL_TEST, Y_TRAIN, Y_VAL_TEST = train_test_split(X, Y, train_size=train_val_test[0]/100, shuffle=False, random_state=None)
        
        del X, Y #done to release memory
        
        X_VAL, X_TEST, Y_VAL, Y_TEST = train_test_split(X_VAL_TEST, Y_VAL_TEST, train_size=train_val_test[1]/(100-train_val_test[0]), shuffle=False, random_state=None)
            
        del X_VAL_TEST, Y_VAL_TEST #done to release memory
        
        #the scaler can only take 2D arrays, thus we must undo the reshape of the splat sets 
        X_TRAIN=X_TRAIN.reshape(int(X_TRAIN.shape[0]*X_TRAIN.shape[1]), n_var_in)
        X_VAL=X_VAL.reshape(int(X_VAL.shape[0]*X_VAL.shape[1]), n_var_in)
        X_TEST=X_TEST.reshape(int(X_TEST.shape[0]*X_TEST.shape[1]), n_var_in)
        
        #same reasoning for the outputs
        Y_TRAIN=Y_TRAIN.reshape(int(Y_TRAIN.shape[0]*Y_TRAIN.shape[1]), n_var_out)
        Y_VAL=Y_VAL.reshape(int(Y_VAL.shape[0]*Y_VAL.shape[1]), n_var_out)
        Y_TEST=Y_TEST.reshape(int(Y_TEST.shape[0]*Y_TEST.shape[1]), n_var_out)
        
        # apply the scaler: note that we fit/train it into the training set only
        #and then we apply it on the remaining sets. If we were to fit_transform 
        #on the whole dataset, we would be introducing informantion from the test
        #into the training, hence, introducing bias.
        X_TRAIN = Xscaler.fit_transform(X_TRAIN)
        X_VAL = Xscaler.transform(X_VAL)
        X_TEST = Xscaler.transform(X_TEST)
        
        #same reasoning for the outputs
        Y_TRAIN = Yscaler.fit_transform(Y_TRAIN)
        Y_VAL = Yscaler.transform(Y_VAL)
        Y_TEST = Yscaler.transform(Y_TEST)
        
        #redo the reshape so that they end being 3D vectors since the ML models take them like this
        X_TRAIN=X_TRAIN.reshape(int(X_TRAIN.shape[0]/(PRE+1)), PRE+1, n_var_in)
        X_VAL=X_VAL.reshape(int(X_VAL.shape[0]/(PRE+1)), PRE+1, n_var_in)
        X_TEST=X_TEST.reshape(int(X_TEST.shape[0]/(PRE+1)), PRE+1, n_var_in)
        
        Y_TRAIN=Y_TRAIN.reshape(int(Y_TRAIN.shape[0]/H), H, n_var_out)
        Y_VAL=Y_VAL.reshape(int(Y_VAL.shape[0]/H), H, n_var_out)
        Y_TEST=Y_TEST.reshape(int(Y_TEST.shape[0]/H), H, n_var_out)
        
        #these are the two variables to be output by this function
        Scaler = {
                  'X_data' : Xscaler,
                  'Y_data' : Yscaler,
                  }
       
        ML_DATA = {
            "X_TRAIN": X_TRAIN,
            "X_VAL": X_VAL,
            "X_TEST": X_TEST,
            "Y_TRAIN": Y_TRAIN,
            "Y_VAL": Y_VAL,
            "Y_TEST": Y_TEST,
            "xcols": list(xcols),
            "ycols": list(ycols),
            }
            
    else:
        print("\n\n\n WARNING: Your ML method is not supported by the 'PreProcessDataset' function.\n\n")
    
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



