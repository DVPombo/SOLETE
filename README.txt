Title of the dataset: The SOLETE Platform
_______________________________________________________________
Creators:
Daniel Vázquez Pombo https://orcid.org/0000-0001-5664-9421
_______________________________________________________________
Contributors:
Mario Javier Rincon Perez https://orcid.org/0000-0003-3239-6612
_______________________________________________________________
Related publications:
https://github.com/DVPombo/SOLETE -> GitHub repository.
https://doi.org/10.11583/DTU.17040767 -> The SOLETE dataset, which is required to run the code
https://doi.org/10.1016/j.dib.2022.108046 -> paper describing the dataset [1]
https://doi.org/10.3390/s22030749 -> paper using the dataset [2]
https://doi.org/10.1016/j.egyr.2022.05.006 -> paper using the dataset [3]
https://doi.org/10.1016/j.segan.2022.100943 -> paper using the dataset [4]
All of them are open access so you should be able to get them easily.
_______________________________________________________________
Description:
The SOLETE platform consists of a number of scripts facilitating the ussage of the homonimous
dataset. Furthermore, this item is a Machine Learning platform aimed at time-series forecasting 
focused on PV power. The different scripts have various functions. One allows to import SOLETE 
and show some plots. Another is a platform where you can play with different Machine Learning 
models for time series forecasting. The application focuses on predicting PV power, but it can 
be easily edited by the user. 

For more information, dependencies, etc. Refer to the git https://github.com/DVPombo/SOLETE

The platform should be useful as a learning tool in the machine learning field as it allows to 
train, test, and validate different machine learning methods targeting time-series forecasting.

The SOLETE dataset (see Related publications above) was originally disclosed in [1] to increase
the transparency and replicability of [2], [3], and [4]. Yet, SOLETE's enormous potential to 
facilitate derived research motivated the development of the SOLETE platform. Said platform 
started as a few Python scripts allowing to import and plot pieces of the dataset. But nowadays 
is a comprehensive platform facilitating training of physics-informed machine learning models for 
time-series forecasting.
_______________________________________________________________
Keywords:
#MachineLearning #SolarPowerForecasting #PVforecasting #SOLETE
#Humidity #Irradiance #Gaia #HybridPowerPlant #ColocatedWindSolar #HybridPowerSystem
#WindPower #SolarPower #PVpower #Meteorology #Temperature #WindDirection #WindSpeed
_______________________________________________________________
This dataset contains the following files:
README.txt: contains general information and links to related documents
README.md: similar to README.txt but employed by Git to compile text.
RunMe.py: fast sneak peek into SOLETE's contents. Shows how to import and plot.
MLForecasting.py: Main file of the platform allowing to configure different machine learning models
Functions.py: compilation of all functions required by the rest of the scripts.
_______________________________________________________________
Methods, materials and software:
100% Python, although we might release something for Matlab soon.
_______________________________________________________________
How to cite:
Pombo, Daniel Vazquez (2023): The SOLETE platform. GitHub. Dataset. https://doi.org/10.11583/DTU.17040767

-version 3
@misc{Pombo2023SOLETEplatform,
author = "Daniel Vazquez Pombo",
title = "{The SOLETE Platform}", 
year = "2023",
month = "Mar",
url = "https://doi.org/10.11583/DTU.17040626",
doi = "10.11583/DTU.17040626",
note = {Retrieved from: \url{https://doi.org/10.11583/DTU.17040626}, {DOI}: {https://doi.org/10.11583/DTU.17040626}},
}

You can also refer to [1]
_______________________________________________________________
References:
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
_______________________________________________________________
This platform is published under the MIT license. https://opensource.org/license/mit/

