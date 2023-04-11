# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:35:08 2021
Latest edit April 2023

author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com
LinkedIn: https://www.linkedin.com/in/dvp/
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo

The purpose of this script is to give you, dear user, a gentle hand and, 
hopefully, kick start your work with the one and only, the SOLETE dataset.
It should run without errors simply by placing all the files in the same location.
We have checked the hdf5 file (the actual dataset) compatibility with Mathlab, Python and R. 
If you encounter any troubles using it with other software let me know and I will see what I can do.

The licensing of this work is pretty chill, just give credit: https://creativecommons.org/licenses/by/4.0/
"""

import pandas as pd
import matplotlib.pyplot as plt
from Functions import import_PV_WT_data

#%% Import Data
# DATA=pd.read_hdf('SOLETE_short.h5')
# DATA=pd.read_hdf('SOLETE_Pombo_1sec.h5') #WARNING, this file is huge, consider only reading a piece of it.
# DATA=pd.read_hdf('SOLETE_Pombo_1min.h5')
# DATA=pd.read_hdf('SOLETE_Pombo_5min.h5')
DATA=pd.read_hdf('SOLETE_Pombo_60min.h5')
PVinfo, WTinfo = import_PV_WT_data()

#%% Plot a bit of data
fig, ax = plt.subplots(nrows=2, ncols=2)
DATA['POA Irr[kW1m2]'].plot(ax=ax[0,0], legend=True)
DATA['P_Solar[kW]'].plot(ax=ax[0,1], legend=True)
DATA['WIND_SPEED[m1s]'].plot(ax=ax[1,0], legend=True)
DATA['P_Gaia[kW]'].plot(ax=ax[1,1], legend=True)
plt.show()

#%% Plot the power curve of the WT and the efficiency curve of the PV array
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(WTinfo["CWs"], WTinfo["CP"])
fig.subplots_adjust(hspace=0.4)
ax[1].plot(PVinfo["eff_P"][0], PVinfo["eff_%"][0])
ax[1].plot(PVinfo["eff_P"][1], PVinfo["eff_%"][1])

ax[0].set_xlim(0, 25)
ax[0].set_ylim(0, 12)
ax[0].set_xlabel('Wind Speed [m/s]')
ax[0].set_ylabel('Power [kW]')
ax[0].grid(True)

ax[1].set_xlim(0, 10000)
ax[1].set_ylim(0, 110)
ax[1].set_xlabel('Power [W]')
ax[1].set_ylabel('Efficiency [%]')
ax[1].grid(True)

plt.show()

print("Hey! Would you look at that? \nSOLETE ran, \nSOLETE plotted something, \nSOLETE works... \n\nSOLETE ROCKS!")

"""
Thank you for using the SOLETE dataset. Have fun with it, share it, spread the word.

Remember that there is an Open Access paper describing the data: https://doi.org/10.1016/j.dib.2022.108046

Also, I would like to mention again that the field data comes from Risø. Google it.
It is a fantastic research facility in Denmark, where I had the privilege of pursuing
my PhD in the Technical University of Denmark (DTU).
"""