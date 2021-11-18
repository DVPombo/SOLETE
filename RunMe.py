# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:35:08 2021

@author: Daniel Vázquez Pombo
email: dvapo@elektro.dtu.dk
LinkedIn: https://www.linkedin.com/in/dvp/
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo

The purpose of this script is to give you, dear user, a gentle hand and, hopefully, kick start your work with the Solete dataset.
It should run without errors simply by placing all the files in the same location.
We have checked the hdf5 file (the actual dataset) compatibility with both Python and R. 
If you encounter any troubles using it with other software let me know and I will see what I can do.

The lincensing of this work is pretty chill, just give credit: https://creativecommons.org/licenses/by/4.0/

Dependencies: Python 3.8.10, and Pandas 1.2.4
"""


import pandas as pd
import matplotlib.pyplot as plt


def import_PV_WT_data():
    """
    Returns
    -------
    PV : dict
        Holds data regarding the PV string in SYSLAB 715
    WT : dict
        Holds data regarding the Gaia wind turbine

    """

    PV = {
        "Type": "Poly-cristaline",
        "Az": 60,  # deg
        "Estc": 1000,  # W/m**2
        "Tstc": 25,  # C
        'Pmp_stc': [165, 125],  # W
        'ganma_mp': [-0.478 / 100, -0.45 / 100],  # 1/K
        'Ns': [18, 6],  # int
        'Np': [2, 2],  # int
        'a': [-3.56, -3.56],  # module material construction parameters a, b and D_T
        'b': [-0.0750, -0.0750],
        'D_T': [3, 3],  # represents the difference between the module and cell temperature
        # these three parameters correspond to glass/cell/polymer sheet with open rack
        # they are extracted from Sandia document King, Boyson form 2004 page 20
        'eff_P': [[0, 250, 400, 450, 500, 600, 650, 750, 825, 1000, 1200, 1600, 2000, 3000, 4000, 6000, 8000, 10000],
                  [0, 250, 400, 450, 500, 600, 650, 750, 825, 1000, 1200, 1600, 2000, 3000, 4000, 6000, 8000, 10000]],
        'eff_%': [[0, 85.5, 90.2, 90.9, 91.8, 92, 92.3, 94, 94.4, 94.8, 95.6, 96, 97.3, 97.7, 98, 98.1, 98.05, 98],
                  [0, 85.5, 90.2, 90.9, 91.8, 92, 92.3, 94, 94.4, 94.8, 95.6, 96, 97.3, 97.7, 98, 98.1, 98.05, 98]],
        "index": ['A', 'B'],  # A and B refer to each channel of the inverter, which has connected a different string.
    }

    WT = {
        "Type": "Asynchronous",
        "Mode": "Passive, downwind vaning",
        "Pn": 11,  # kW
        "Vn": 400,  # V
        'CWs': [3.5, 6, 8, 10, 10.5, 11, 12, 13, 13.4, 14, 16, 18, 20, 22, 24, 25, ],  # m/s
        'CP': [0, 5, 8.5, 10.9, 11.2, 11.3, 11.2, 10.5, 10.5, 10, 8.8, 8.7, 8, 7.3, 6.6, 6.3, ],  # kW
        "Cin": 3.5,  # m/s
        "Cout": 25,  # m/s
        "HH": 18,  # m
        "D": 13,  # m
        "SA": 137.7,  # m**2
        "B": 2,  # int
    }

    return PV, WT


#%% Import Data
DATA=pd.read_hdf('SYSLAB715_Solete_Pombo.h5')
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