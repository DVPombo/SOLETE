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

The licensing of this work is pretty chill, just give credit: https://creativecommons.org/licenses/by/4.0/


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
    1- Go to section: "Control the Script"
        In the dictionary called Control_Var you can modify different values, like the
        horizon to forecast, the division between training, validation and testing, what
        is the metric to be predicted, which should be used as extrinsic features, etc.
               
        Modify "SOLETE_builvsimport" from "Build" to "Import". This avoid expanding the dataset
        all the time and simply imports last built version. However, note that each version is
        exclusive of a resolution. By default that is set to 60min, you can try 5min as well.
    2- Go to section: Define Machine Learning Configuration and Hyperparameters
        Use those dictionaries to define different topologies for the 5 ML methods 
        covered in references [1,2,3]. However, be careful with ANN, they have a 
        particular shape, you can't control each layer's position from there. 
        But you can edit that yourself.
    3- After setting those parameters just run it. Take into account the size of 
    the data, it might take a while for the algorithm to finish.
    4- Now that you have some time to spare while it runs: 
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

from Functions import import_SOLETE_data, import_PV_WT_data, PreProcessDataset  
from Functions import PrepareMLmodel, TestMLmodel, post_process

#%% Control The Script:

Control_Var = {
    '_description_' : 'Holds all the variables that define the behaviour of the algoritm',
    'resolution' : '60min', #either 1sec, 1min, 5min or 60min
    'SOLETE_builvsimport': 'Build', # Takes 'Build' or 'Import'. The first expands the dataset, the second imports a existing expansion
    'SOLETE_save': True, # saves the built SOLETE. Only works if SOLETE_builvsload=='Build' 
    'trainVSimport' : True, #True - trains the ML model, False - imports the model
    'saveMLmodel' : True, #saves the trained model if True, but also trainVSimport must be True, otherwise does nothing.
    'Train_Val_Test' : [70, 20, 10], #train validation test division of the available DATA
    'Scaler': 'MinMax01', #scaling technique, choose: 'MinMax01', 'MinMax11' 'Standard' see notes in PreProcessDataset()
    'IntrinsicFeature' : 'P_Solar[kW]', #feature to be predicted
    'PossibleFeatures': ['TEMPERATURE[degC]', 'HUMIDITY[%]', 'WIND_SPEED[m1s]', 'WIND_DIR[deg]',
                        'GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Gaia[kW]', 'P_Solar[kW]', 'Pressure[mbar]', 
                        'Pac', 'Pdc','TempModule', 'TempCell', #'TempModule_RP', 
                        'HoursOfDay', 'MeanPrevH', 'StdPrevH', 'MeanWindSpeedPrevH', 'StdWindSpeedPrevH',
                        ],
    'MLtype' : 'CNN', # RF SVM LSTM CNN CNN_LSTM
    'H' : 10, #horizon length in number of samples
    'PRE' : 5, #previous samples to be used in the predictor
    }  

#%% Define Machine Learning Configuration and Hyperparameters
# edit this dictionaries to tailor-made the ML model that you want 

RF = {'_description_' : 'Holds the values related to Random Forest',
        'n_trees' : 1, #number of trees
        'random_state' : 32, #initialization number, can be removed for random seed generation
        }

SVM = {'_description_' : 'Holds the values related to Support Vector Machine',
       'kernel' : 'rbf', #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ <--> default=’rbf’
        'degree' : 3, # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.  
        'gamma' : 'scale', # ‘scale’, ‘auto’ -> if you don't know what you are doing leave it as scale
        'coef0' : 0, # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’. 
        'C' : 3, # Regularization parameter. The strength of the regularization is inversely proportional to C. 
                # Must be strictly positive. The penalty is a squared l2 penalty.
        'epsilon' : 0.1, # Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty 
                        # is associated in the training loss function with points predicted within a distance epsilon from the actual value.
        }

LSTM = {'_description_' : 'Holds the values related to LSTM ANN design',
        'n_batch' : 16, #int <-> # number of samples fed together - helps with paralelization  (smaller takes longer, improves performance carefull with overfitting)
        'epo_num' : 1000, # 5 - epoc number of iterations of each batch - same reasoning as for the batches'
        'Neurons' : [15,15,15], #number of neurons per layer <-> you can feed up to three layers using list e.g. [15, 10] makes two layers of 15 and 10 neurons, respectively.
        'Dense'  : [0, 0], #number of dense layers and neurons in them. If left as 0 they are not created.
        'ActFun' : 'tanh', #sigmoid, tanh, elu, relu - activation function as a str 
        'LossFun' : 'mean_absolute_error', #mean_absolute_error or mean_squared_error
        'Optimizer' : 'adam' # adam RMSProp - optimization method adam is the default of the guild 
        }

CNN = {'_description_' : 'Holds the values related to LSTM NN design',
        'n_batch' : 16, #see note in LSTM
        'epo_num' : 3, #see note in LSTM
        'filters' : 32, #number of nodes per layer, usually top layers have higher values
        'kernel_size' : 2, #size of the filter used to extract features
        'pool_size' : 3, #down sampling feature maps in order to gain robustness to changes
        'Dense'  : [10, 10],#see note in LSTM
        'ActFun' : 'tanh', #see note in LSTM
        'LossFun' : 'mean_absolute_error', #see note in LSTM
        'Optimizer' : 'adam' #see note in LSTM
        }

CNN_LSTM = {'_description_' : 'Holds the values related to LSTM NN design',
        'n_batch' : 16, #see note in LSTM
        'epo_num' : 1000, #see note in LSTM        
        'filters' : 32, #see note in CNN
        'kernel_size' : 3, #see note in CNN
        'pool_size' : 2, #see note in CNN
        'Dense'  : [0, 0], #see note in LSTM
        'CNNActFun' : 'tanh', #see note in CNN
        
        'Neurons' : [10,15,10], #see note in LSTM
        'LSTMActFun' : 'sigmoid', #see note in LSTM
        
        'LossFun' : 'mean_absolute_error', #see note in LSTM
        'Optimizer' : 'adam' #see note in LSTM
        }

Control_Var['RF'] = RF
Control_Var['SVM'] = SVM
Control_Var['LSTM'] = LSTM
Control_Var['CNN'] = CNN
Control_Var['CNN_LSTM'] = CNN_LSTM
del RF, SVM, LSTM, CNN, CNN_LSTM

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























