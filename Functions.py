# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:35:08 2021
Latest edit April 2023

author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com
LinkedIn: https://www.linkedin.com/in/dvp/
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo

The purpose of this script is to collect all functions called by the rest of the scripts.

The licensing of this work is pretty chill, just give credit: https://creativecommons.org/licenses/by/4.0/
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

from CoolProp.HumidAirProp import HAPropsSI
import sys
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
#yep, bad practice, see function get_results to understand why is this here :) 

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
        
        ExpandSOLETE(df, [PVinfo, WTinfo], Control_Var)
        
        if Control_Var["SOLETE_save"]==True:
            df.to_hdf(name_import, 'name', mode='w')
            
    elif Control_Var["SOLETE_builvsimport"]=='Import':
        
        try:
            df=pd.read_hdf(name_import)
        except FileNotFoundError:
            error_msg(key = "missing_expanded_SOLETE")
        
        print("SOLETE was imported:")
        print("    -resolution: ", Control_Var['resolution'])
        print("    -version: Expanded. ")
        
        for col in Control_Var['PossibleFeatures']: #if the possiblefeature includes
        #something that was not in the import file, execution is killed with an error message
            if col not in df.columns: 
                error_msg(key = "missing_feature_expanded_SOLETE")
        
        for col in df.columns: #the undesired columns are removed
        #undersired columns are those within the imported file not appearing in Control_Var['PossibleFeatures']
            if col not in Control_Var['PossibleFeatures']: 
                df = df.drop(col, axis=1)
                print("Dropped col: ", col)           
        
        print("\n")
        
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
    data['P_Solar[kW]'] =  np.where(data['Pac'] >= 1.5*data['P_Solar[kW]'],
                                    data['Pac'], data['P_Solar[kW]'])
    print("    Smoothing zeros")
    data['P_Solar[kW]'] =  np.where(data['P_Solar[kW]'] <= 0.001,
                                      0, data['P_Solar[kW]'])
    data['Pac'] =  np.where(data['Pac'] <= 0.001,
                                      0, data['Pac'])
    
    if 'TempModule_RP' in Control_Var['PossibleFeatures']: #advanced thermodynamic model
        data['TempModule_RP'] = Rincon_Pombo_ThermodynamicModel(data, info[0])
        list_expansion.append('TempModule_RP')
    
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


def PV_Performance_Model(data, PVinfo, colirra='POA Irr[kW1m2]', coltemp='TEMPERATURE[degC]',colwindspeed='WIND_SPEED[m1s]'):
    """
    This function implements King's PV performance model. More info in [2].
    
    Parameters
    ----------
    data : DataFrame
        Variable including all the data from the Solete dataset
    PVinfo : dict
        A bunch of parameters extracted from the datasheet and other supporting documents
        Check function: import_PV_WT_data for further details
    colirra : string, optional
        holds Epoa, that is the irradiance in the plane of the array in kW/m2
        If you reuse this code, make sure you are feeding Epoa and not GHI
        The default is 'POA Irr[kW1m2]'.
    coltemp : string, optional
        holds the ambient temperature in Celsius.
        The default is 'TEMPERATURE[degC]'.
    colwindspeed : string, optional 
        holds the wind speed in m/sec
        The default is'WIND_SPEED[m1s]'

    Returns
    -------
    DataFrames
        Pac, Pdc, Tm, and Tc. 

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
        Results['Tm_' + pv] = data[coltemp] + data[colirra]*1000 *np.exp(DATA_PV.loc[pv,'a']+DATA_PV.loc[pv,'b']*data[colwindspeed]) 
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

def Rincon_Pombo_ThermodynamicModel(data, pv, verbose=0):
    """
    This function implements section 4.4 from [4]. That is, an advanced thermodynamic 
    performance model for photovoltaic pannels. All credit for the coding goes to my
    good friend Mario Javier Rincón Pérez (mjrp@mpe.au.dk). I simply adapted it to fit
    in the SOLETE platform. If you are into fluid and thermodynamics reach out to him.
    
    Note that this function is not very pythonee, which makes it slow. 
    Could it be that it was originally coded in Matlab and then poorly ported? Maybe. 
    Will there be a future release making it faster? Maybe.

    Parameters
    ----------
    data : DataFrame
            Variable including all the data from the Solete dataset
    verbose : int, optional
        The default is 0.
        If a 1 is fed, the iterations are shown. 

    Returns
    -------
    df : DataFrame
        Temperature of the modules according to the Rincón-Pombo method [4].

    """

    print("")    
    print("Expanding SOLETE with the Rincon-Pombo thermodynamic model")    
    print("    This is going to take a while be patient.")    
    # Hardcoded INPUTS
    g = 9.81  # gravity m/s^2
    psi = 0.1  # Gradient limiter factor
    gradLimiter = 5  # max gradT allowed without limiter applied
    # PV cells
    # E_STC = 1000  # Solar irradiance W/m^2
    E_POA = data['POA Irr[kW1m2]'] * 1000  # Solar irradiance W/m^2
    # E_POA = data['GHI[kW/m2]'] * E_STC
    epsilon = 0.3  # radiative emissivity (glass)
    SB = 5.670374419e-8  # stefan boltzmann constant
    reflectitivy = 0.6  # light that is reflected by the module to the atmoshpere \reflactance
    transmittance = 0.1  # these three sum  \tau trasmittance
    absorption = 1 - reflectitivy - transmittance  # \alpha absorptance
    IRratio = 0.53  # Infra Red light factor = contributes to heating
        
    # Air
    R = 8.31432e3  # ideal gas constant
    R_a = 285.9  # dry air gas constant
    R_w = 461.5  # water vapour constant
    T = 273.15 + data['TEMPERATURE[degC]']  # dry bulb temperature K
    p = data['Pressure[mbar]'] * 100  # pressure Pa
    
    # Initial variables declaration
    Nu = 0
    T_plot = np.array([])
    T_PV = 273.15 + data['TempModule'][0]  # initialise temperature of PV cell
    A = pv["L"] * pv["W"]  # PV area
    
    gradT = np.array([])
    index = 0
    
    for i in range(len(data)):  
        # RADIATION
        if T_PV >= T[i]:
            q_epsilon = -epsilon * SB * A * (T_PV ** 4 - T[i] ** 4)
        else:
            q_epsilon = 0
    
        q_absorbed = E_POA[i] * A * IRratio * absorption
        # CONVECTION
        # Thermophysical properties of air and fluid mechanics variables
        rho_a = p[i] / (R_a * T[i])  # density of dry air
        if data['HUMIDITY[%]'][i] > 1:
            data['HUMIDITY[%]'][i] = 1.0
    
    
        rho = rho_a * (1 + data['HUMIDITY[%]'][i]) / (1 + R_w / R_a * data['HUMIDITY[%]'][i])  # density of mixture
        mu = HAPropsSI('mu', 'P', p[i], 'T', T[i], 'R', data['HUMIDITY[%]'][i])  # dynamic viscosity
        cp = HAPropsSI('cp_ha', 'P', p[i], 'T', T[i], 'R', data['HUMIDITY[%]'][i])  # specific heat per unit of humid air
        k = HAPropsSI('k', 'P', p[i], 'T', T[i], 'R', data['HUMIDITY[%]'][i])  # thermal conductivity
        nu = mu / rho
        beta = 1 / T[i]  # thermal expansion coefficient for ideal gases
    
        # Nusslet number correlations for humidity changes
        Re = rho * abs(data['WIND_SPEED[m1s]'][i]) * pv["L"] / mu  # Reynolds number
        Pr = cp * mu / k  # Prandtl number
        Gr = g * beta * abs((T_PV - T[i])) * (A / (2 * pv["W"] + 2 * pv["L"])) ** 3 / nu ** 2  # Grashof number
        Ra = Gr * Pr  # Rayleigh number
        hx = np.array([])
        for x in np.linspace(0, pv["L"], num=100): #PVdiscretisation
            Rex = rho * abs(data['WIND_SPEED[m1s]'][i]) * x / mu
            if Rex <= 1e5:  # Laminar, similarity solutions
                if Pr < 0.6:
                    print('Correlation does not satisfy')
                Nux = 0.453 * Rex ** (1 / 2) * Pr ** (1 / 3)
                Nu = 0.680 * Re ** (1 / 2) * Pr ** (1 / 3)
            elif Rex > 1e5:  # Turbulent, empirical correlations
                Nux = 0.0308 * Rex ** (4 / 5) * Pr ** (1 / 3)
                Nu = (0.037 * Re ** (4 / 5) - 871) * Pr ** (1 / 3)
    
            if x == 0:
                hx = np.append(hx, 0)
            else:
                hx = np.append(hx, Nux * k / x)
    
        h = np.mean(hx)  # mean convective heat transfer coefficient (W/m^2/K)
    
        if 1e4 < Ra < 1e7 and Re < 1e3:  # Natural convection, empirical correlations
            Nu = 0.54 * Ra ** (1 / 4)
            h = Nu * k / pv["L"]  # mean convective heat transfer coefficient from correlations (W/m^2/K)
    
        elif 1e7 < Ra < 1e11 and Re < 1e3:  # Natural convection, empirical correlations
            Nu = 0.15 * Ra ** (1 / 3)
            h = Nu * k / pv["L"]  # mean convective heat transfer coefficient from correlations (W/m^2/K)
    
        if h == 0:  # numerical solution for problems in convection or inputs
            h = 1e-16
            R = pv["d"] / (pv["k_r"] * A * 2)  # Thermal resistance of the system
        else:
            R = 1 / (h * A) + pv["d"] / (pv["k_r"] * A * 2)  # Thermal resistance of the system
    
        q_convection = h * A * (T[i] - T_PV)
       
        # Heat balance
        q = (q_absorbed + q_convection + q_epsilon)
        T_PV = T_PV + q * R
    
        if i == 0:
            gradT = np.append(gradT, 0)
        else:
            gradT = np.append(gradT, T_PV - T_plot[-1])
            if abs(gradT[-1]) >= gradLimiter:  # Gradient limiter function
                T_PV = T_plot[-1] + psi * gradT[-1]
                T_plot[-1] = T_PV
                gradT[-1] = T_PV - T_plot[-1]
    
        T_plot = np.append(T_plot, T_PV)
    
        if i % 500 == 0: #output progress every 500 samples
            msg='    Progress: ' + str(round(index/len(data)*100)) + ' %'
            sys.stdout.write('\r'+msg)

    
        index = index + 1
    
    df = T_plot - 273.15
    
    msg='    Progress: ' + str(100) + ' %'
    sys.stdout.write('\r'+msg)    
    print("")
    
    
    return df

def PreProcessDataset(data, control):
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
        
    Scaler : dict
        data of the scaling method applied to the data. This must be inverted
        afterwards in oder functions.
        
    Arguments
    ---------
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        base: Is the basic variable that will be shifted back and forward in time, e.g. Pestimated
        additions: are the other variables that will tag along base to complete the dataset
        train_val_test = division of timestamps in the three blocks

    """

    base=control["IntrinsicFeature"]
    additions= control["PossibleFeatures"].copy()
    n_var_out = len(additions)
    additions.remove(base)
    
    
    n_var_in = len(additions) #number of variables going into the model
    n_var_out = n_var_out - len(additions) #number of variables as output to the model
    H=control["H"] #number of samples of each test, corresponds to time
    PRE = control["PRE"] #number of previous samples
    train_val_test = control['Train_Val_Test']
    
    # PRE=0
    
    train_val_test = control['Train_Val_Test']
            
    X = data[additions] 
    Y = pd.DataFrame(data[base])
        
    
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
        #might behave badly if each features does look like standard normally distributed data: Gaussian with zero mean and unit variance.
        Xscaler = StandardScaler() #initialise the scaler 
        Yscaler = StandardScaler() #initialise the scaler 
    
    if control['MLtype'] in ['LSTM', 'CNN', 'CNN_LSTM']:
        
        #we do this in order to build 3D vectors of shapes
        X.reset_index(inplace=True) #(samples, PRE+1, n_variables_in) 
        Y.reset_index(inplace=True) #(samples, H, n_variables_predicted)
        
        #with this loop we append nans in the rows we need to complete the last H
        for i in range(0,int(np.ceil(X.shape[0]/(PRE+1))*(PRE+1))-X.shape[0]):
            X = pd.concat([X, pd.DataFrame([np.nan])], axis=0, ignore_index=True)
            X = X.drop(0, axis=1) 
        
        #the same must happen to Y
        for i in range(0,int(np.ceil(Y.shape[0]/H)*H)-Y.shape[0]):
            Y = pd.concat([Y, pd.DataFrame([np.nan])], axis=0, ignore_index=True)
            Y = Y.drop(0, axis=1) 
        
        #this is unnecesary since the reshape makes you lose the index anyway
        X.set_index('index',inplace=True) 
        Y.set_index('index',inplace=True)
        
        #change the shape into 3D vectors 
        X=X.values.reshape(int(X.shape[0]/(PRE+1)), PRE+1, n_var_in) #(samples, PRE+1, n_variables_in)
        Y=Y.values.reshape(int(Y.shape[0]/H), H, n_var_out) #(samples, H, n_variables_predicted)
        
        
        #TODO: de aqui para abajo no corre. Más allá de eso el reshaping de X e Y tiene más miga ya que el número de samples
        #tiene que ser igual y coherente. Jaleito guapo. Toca pensar como montar tanto X como Y. Y quizás salga fácil con la
        #fcn de machine learning mastery para timeseries más el reshape que ya está hecho ahora. Para la X no lo tengo tan claro.
        #buenas noches y buena suerte :D
        #Dormi mal y se me ocurrio lo siguiente. Pillar Y, aplicar timeseries como en machinelearning mastery (samples, h),
        #eso se puede reshape en (samples, H, n_variables_predicted) creo
        #en el caso de X, o si Y tiene más de una variable. Creo que habría que dividir dicha función en una dataframe por variable
        #repetir el proceso y luego concat las matrices.
        
        #split dataset into training, validation, and testing sets. This is done with the 3D vectors in order to
        #apply shuffle why not lossing the temporal correlations (include 'Time' in possible features to see how it works)
        X_TRAIN, X_VAL_TEST, Y_TRAIN, Y_VAL_TEST = train_test_split(X, Y, train_size=train_val_test[0]/100, shuffle=True, random_state=None)
        
        del X, Y
        
        X_VAL, X_TEST, Y_VAL, Y_TEST = train_test_split(X_VAL_TEST, Y_VAL_TEST, train_size=train_val_test[1]/(100-train_val_test[0]), shuffle=True, random_state=None)
            
        del X_VAL_TEST, Y_VAL_TEST
        
        #undoing the reshape into the suffled sets to be able to use the scaler function
        X_TRAIN=X_TRAIN.reshape(int(X_TRAIN.shape[0]*X_TRAIN.shape[1]), n_var_in)
        X_VAL=X_VAL.reshape(int(X_VAL.shape[0]*X_VAL.shape[1]), n_var_in)
        X_TEST=X_TEST.reshape(int(X_TEST.shape[0]*X_TEST.shape[1]), n_var_in)
        
        Y_TRAIN=Y_TRAIN.reshape(int(Y_TRAIN.shape[0]*Y_TRAIN.shape[1]), n_var_out)
        Y_VAL=Y_VAL.reshape(int(Y_VAL.shape[0]*Y_VAL.shape[1]), n_var_out)
        Y_TEST=Y_TEST.reshape(int(Y_TEST.shape[0]*Y_TEST.shape[1]), n_var_out)
        
        #apply the scaler
        X_TRAIN = Xscaler.fit_transform(X_TRAIN)
        X_VAL = Xscaler.transform(X_VAL)
        X_TEST = Xscaler.transform(X_TEST)
        
        Y_TRAIN = Yscaler.fit_transform(Y_TRAIN)
        Y_VAL = Yscaler.transform(Y_VAL)
        Y_TEST = Yscaler.transform(Y_TEST)
        
        #redo the reshape so that they end being 3D vectors since the ML models take them like this
        X_TRAIN=X_TRAIN.reshape(int(X_TRAIN.shape[0]/H), H, n_var_in)
        X_VAL=X_VAL.reshape(int(X_VAL.shape[0]/H), H, n_var_in)
        X_TEST=X_TEST.reshape(int(X_TEST.shape[0]/H), H, n_var_in)
        
        Y_TRAIN=Y_TRAIN.reshape(int(Y_TRAIN.shape[0]/H), H, n_var_out)
        Y_VAL=Y_VAL.reshape(int(Y_VAL.shape[0]/H), H, n_var_out)
        Y_TEST=Y_TEST.reshape(int(Y_TEST.shape[0]/H), H, n_var_out)
        
        
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
            }
            
    else:
        print("\n\n\n WARNING: Your ML method is not supported by the 'TimePeriods' function.\n\n")
    
    return ML_DATA, Scaler

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
        
    Scaler : dict
        data of the scaling method applied to the data. This must be inverted
        afterwards in oder functions.
        
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
    
    Note: This is a shitty way of implementing this for several reasons, for instance it results 
    in a PerformanceWarning due to the DataFrame RESULTS being fragmented due to using inser too 
    many times. However, and even worse than that is the fact that RESULTS is an horizontal 
    DataFrame rather than a vertical one, which is a big nono. I may fix it in a future release.

    Parameters
    ----------
    control : dict
        Control_Var
    data : DataFrame
        SOLETE dataset
    ml_data : dict of dataframes
        ML_DATA, the sets cointaining training, validation and testing.
    predictions : DataFrame
        Predicted values

    Returns
    -------
    RESULTS : DataFrame
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
    
    if key == "resolution":
        print("ERROR: You have selected a resolution that is not available.\n")
        print("Worry not, it is easy to fix:")
        print("     1-Go to the code section: Control The Script")
        print("     2-Edit the dict Control_Var[resolution]")
        print("     Available options: 5min or 1h")
    
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
        
        
    sys.exit("Did I do that? ¯\_(ツ)_/¯ \n\n\nCheck at the top for hints on what went wrong!!!")
    #yes, that was a reference to good old Steve Urkel
    pass






